# train_lora_fixed.py
import json
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from tqdm import tqdm

train_path = "../train.jsonl"
base_model_id = "YandexGPT-5-Lite-8B-instruct"
output_dir = "../lora-out"
MAX_LENGTH = 4000

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype="float16"
)

# ==== LoRA ====
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
)

# ==== TrainingArguments ====
train_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=2,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=20,
    save_steps=10,
    fp16=False,   # важно: отключаем mixed precision
    bf16=False,
    optim="adamw_torch",
    weight_decay=0.01,
    max_grad_norm=1.0,
    report_to="none"
)

max_memory = {0: "6GiB", "cpu": "16GiB"}

# ==== ДАННЫЕ ====
ds = load_dataset("json", data_files=train_path, split="train")

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def format_example(example):
    messages = example["messages"]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

print("Форматирование текста...")
ds = ds.map(format_example, remove_columns=ds.column_names, desc="Formatting examples")

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, max_length=MAX_LENGTH)

print("Токенизация данных...")
tokenized_ds = ds.map(tokenize_function, batched=True, remove_columns=ds.column_names, desc="Tokenizing examples")

# ==== МОДЕЛЬ ====
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    offload_folder="offload",
    max_memory=max_memory
)

# отключаем кеш и gradient checkpointing для совместимости с adamw_torch
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.gradient_checkpointing_disable()

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ==== Callback для tqdm ====
class TQDMProgressCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self.pbar = tqdm(total=state.max_steps or 0, desc="Training progress", leave=True)

    def on_step_end(self, args, state, control, **kwargs):
        if hasattr(self, "pbar") and state.max_steps:
            self.pbar.n = state.global_step
            self.pbar.refresh()

    def on_train_end(self, args, state, control, **kwargs):
        if hasattr(self, "pbar"):
            self.pbar.close()

# ==== TRAINER ====
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_ds,
    args=train_args,
    data_collator=collator,
    callbacks=[TQDMProgressCallback]
)

# ==== TRAIN ====
trainer.train()
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"LoRA-адаптер сохранён в {output_dir}")
