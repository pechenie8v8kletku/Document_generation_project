import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch, transformers
from datasets import load_dataset
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
# Пути
base_model_id = "model-out3/checkpoint-1530"
test_path = "test_prepped_last.jsonl"
output_json = "results2.json"

max_new_tokens = 700




model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    config= AutoConfig.from_pretrained(base_model_id, trust_remote_code=True),
    dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="sdpa",
    low_cpu_mem_usage=True,
).to("cuda")

model.eval()


# Токенизатор
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.pad_token = tokenizer.eos_token


def format_chat(messages):
    text = ""
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role == "system":
            text += f" {content}\n"
        elif role == "user":
            text += f" {content}\n"
        elif role == "assistant":
            text += f" {content}\n"
    return text


# Читаем тестовый датасет
with open(test_path, "r", encoding="utf-8") as f:
    examples = [json.loads(line) for line in f]

batch_size = 2
results = []

for i in range(9,len(examples) , batch_size):
    batch = examples[i:i+batch_size]
    prompts = [format_chat(ex["messages"]) for ex in batch]
    a = 0
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    if i%10==6 or i%10==7:
        a=2500

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max(max_new_tokens, a),
            do_sample=False,
            temperature=None,
            top_p=None, top_k=None,
            num_beams=1,
            repetition_penalty=1.05,
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            early_stopping=True,

        )

    for j, ex in enumerate(batch):
        generated_text = tokenizer.decode(outputs[j], skip_special_tokens=True)
        results.append({
            "id": i+j,
            "messages": ex["messages"],
            "generated": generated_text
        })
        print(f"[{i+j}] готово")

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n✅ Результаты сохранены в {output_json}")
