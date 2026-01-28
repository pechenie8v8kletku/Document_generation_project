import torch, transformers
from datasets import load_dataset,Dataset
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
import json
# ===== Параметры =====
MODEL_ID      = "YandexGPT-5-Lite-8B-instruct"
TRAIN_PATH    = "self_supervised.json"
GOSTS_PATH="just_md_ibm.json"
OUTPUT_DIR    = "model-out2"#_adamw
FREEZE_UP_TO  = 2
TARGET_LEN    = 16384
USE_ADAFACTOR = True


with open(TRAIN_PATH, "r", encoding="utf-8") as f:
    data1 =json.load(f)
with open(GOSTS_PATH, "r", encoding="utf-8") as f:
    data2=json.load(f)
data3=data1+data2

ALIASES = {
    "input":  ["input", "код", "code", "вход", "x"],
    "output": ["output", "макет", "layout", "y", "target"]
}

def norm(rec):
    out = {}
    for std_key, candidates in ALIASES.items():
        for k in candidates:
            if k in rec and rec[k] not in (None, ""):
                out[std_key] = rec[k]
                break
    return out
DATA_FINAL=[norm(r) for r in data3]


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


ds = Dataset.from_list(DATA_FINAL)



def format_chat(txt):
    return txt["output"]
def tokenize(batch):
    text = format_chat(batch)
    return tok(
        text,
        truncation=True,
        max_length=TARGET_LEN,
        padding="max_length"
    )

ds_tok = ds.map(tokenize, batched=True)

optim_choice = "adafactor" if USE_ADAFACTOR else "adamw_bnb_8bit"

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_ratio=0.02,
    lr_scheduler_type="cosine",
    weight_decay=0.003,
    bf16=True, fp16=False,
    logging_steps=10,
    save_steps=100, save_strategy="steps", save_total_limit=3,
    report_to="none",
    optim="adamw_torch_fused",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds_tok,
    tokenizer=tok,
    data_collator=transformers.DataCollatorForLanguageModeling(tok, mlm=False),
)

trainer.train()
