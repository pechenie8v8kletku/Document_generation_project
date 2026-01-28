
import torch
import transformers
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

train_path = "train_prepped.jsonl"
model_id = "YandexGPT-5-Lite-8B-instruct"
output_dir = "lora-out"
MAX_LENGTH = 11000


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_enable_fp32_cpu_offload=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",  # распределение весов по GPU/CPU
    trust_remote_code=True,
    offload_folder="offload",  # при нехватке VRAM
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()
config = LoraConfig(
    r=256,
    lora_alpha=512,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj","gate_proj","up_proj","down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)

dataset = load_dataset("json", data_files={"train": train_path})
def format_chat(messages):
    text = ""
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role == "system":
            text += f"[SYSTEM]: {content}\n"
        elif role == "user":
            text += f"[USER]: {content}\n"
        elif role =="assistant":
            text += f"[ASSISTANT]: {content}\n"
    return text


def tokenize(batch):
    texts = [format_chat(m) for m in batch["messages"]]
    return tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )


tokenized = dataset.map(tokenize, batched=True)

training_args = transformers.TrainingArguments(
    num_train_epochs=2,
    per_device_train_batch_size=1,
    learning_rate=4e-5,
    bf16=True,
    save_total_limit=5,
    logging_steps=5,
    output_dir=output_dir,
    save_strategy='epoch', gradient_accumulation_steps=1,
)
trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized["train"],
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False
trainer.train()
