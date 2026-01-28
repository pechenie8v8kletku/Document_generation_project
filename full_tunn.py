import torch, transformers
from datasets import load_dataset
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)

# ===== Параметры =====
MODEL_ID      = "model-out2/checkpoint-434"
TRAIN_PATH    = "train_prepped.jsonl"
OUTPUT_DIR    = "model-out3"#_adamw
FREEZE_UP_TO  = 2
START_LEN     = 8192
TARGET_LEN    = 16384
USE_ADAFACTOR = True

cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
orig_mpe = getattr(cfg, "max_position_embeddings", 4096)
factor = TARGET_LEN / float(orig_mpe)
'''
cfg.rope_scaling = {
    "type": "yarn",
    "factor": factor,
    "original_max_position_embeddings": orig_mpe
}
cfg.max_position_embeddings = TARGET_LEN
'''
cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    config=cfg,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="sdpa",
    low_cpu_mem_usage=True,
).to("cuda")


tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"
def freeze_prefix_layers(m, upto=10):
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        layers = m.model.layers
    elif hasattr(m, "gpt_neox") and hasattr(m.gpt_neox, "layers"):
        layers = m.gpt_neox.layers
    elif hasattr(m, "transformer") and hasattr(m.transformer, "h"):
        layers = m.transformer.h
    else:
        raise RuntimeError("Не нашёл список блоков слоёв у модели")

    for _, p in m.named_parameters():
        p.requires_grad = False

    for i in range(upto + 1, len(layers)):
        for _, p in layers[i].named_parameters():
            p.requires_grad = True

    for n, p in m.named_parameters():
        if n.startswith("lm_head") or "norm" in n or "layernorm" in n:
            p.requires_grad = True

freeze_prefix_layers(model, FREEZE_UP_TO)
model.config.use_cache = False
model.gradient_checkpointing_enable()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable/1e6:.1f}M / {total/1e6:.1f}M "
      f"({100*trainable/total:.1f}%)")

ds = load_dataset("json", data_files={"train": TRAIN_PATH})["train"]

def build_prompt_and_target(messages):
    last_ass_idx = None
    for idx in range(len(messages)-1, -1, -1):
        if messages[idx].get("role") == "assistant":
            last_ass_idx = idx
            break
    if last_ass_idx is None:
        return None
    ctx = messages[:last_ass_idx]
    target = messages[last_ass_idx]["content"]
    prompt = tok.apply_chat_template(
        ctx,
        tokenize=False,
        add_generation_prompt=True
    )
    return prompt, target

def tokenize(example):
    bt = build_prompt_and_target(example["messages"])
    if bt is None:
        return {}
    prompt, target = bt
    full_text = prompt + target
    tok_full = tok(
        full_text,
        truncation=True,
        max_length=TARGET_LEN,
        padding=False,
        return_attention_mask=True
    )
    n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])

    labels = tok_full["input_ids"][:]
    labels[:n_prompt] = [-100] * min(n_prompt, len(labels))
    tok_full["labels"] = labels
    return tok_full

ds_tok = ds.map(tokenize, remove_columns=ds.column_names)
ds_tok = ds_tok.filter(lambda ex: "input_ids" in ex and len(ex["input_ids"]) > 0)

class DynamicCLMCollator:
    def __init__(self, tokenizer, start_len=2048, pad_to_multiple_of=8):
        self.current_max_len = start_len
        self.inner = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False, pad_to_multiple_of=pad_to_multiple_of
        )
    def __call__(self, features):
        trimmed = []
        L = self.current_max_len
        for f in features:
            trimmed.append({
                "input_ids":      f["input_ids"][:L],
                "attention_mask": f["attention_mask"][:L],
            })
        return self.inner(trimmed)

collator = DynamicCLMCollator(tok, start_len=START_LEN, pad_to_multiple_of=8)

from transformers import TrainerCallback
class LengthWarmupCallback(TrainerCallback):
    def __init__(self, collator, milestones):
        self.collator = collator
        self.milestones = milestones
    def on_step_begin(self, args, state, control, **kwargs):
        progress = (state.global_step / state.max_steps) if state.max_steps else (state.epoch / max(1e-9, args.num_train_epochs))
        for frac, L in self.milestones:
            if progress >= frac:
                self.collator.current_max_len = L

milestones = [
    (0.00, 8192),
    (0.1, 12288),
    (0.2, 16384),
]
callbacks = [LengthWarmupCallback(collator, milestones)]

# ===== Тренировка ====="adamw_torch_fused"
optim_choice = "adafactor" if USE_ADAFACTOR else "adamw_bnb_8bit"

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    warmup_ratio=0.02,
    lr_scheduler_type="cosine",
    weight_decay=0.0005,
    bf16=True,
    fp16=False,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=3,
    report_to="none",
    optim="adamw_torch_fused",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds_tok,
    data_collator=collator,
    tokenizer=tok,
    callbacks=callbacks,
)

trainer.train()
