import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Пути
lora_checkpoint = "lora-out/checkpoint-1699"
base_model_id = "YandexGPT-5-Lite-8B-instruct"
test_path = "test_prepped_last.jsonl"
output_json = "results2489.json"

max_new_tokens = 500

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)


model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    offload_folder="offload",
)

model = PeftModel.from_pretrained(model, lora_checkpoint)
model.eval()


# Токенизатор
tokenizer = AutoTokenizer.from_pretrained(lora_checkpoint)
tokenizer.pad_token = tokenizer.eos_token


def format_chat(messages):
    text = ""
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role == "system":
            text += f"[SYSTEM]: {content}\n"
        elif role == "user":
            text += f"[USER]: {content}\n"
        elif role == "assistant":
            text += f"[ASSISTANT]: {content}\n"
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
    # Токенизация с паддингом только в пределах батча
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    if i%10==6 or i%10==7:
        a=1400

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max(max_new_tokens, a),
            do_sample=False,  # <- НЕТ сэмплинга
            temperature=None,  # игнорируется при do_sample=False
            top_p=None, top_k=None,  # игнорируется
            num_beams=1,  # greedy
            repetition_penalty=1.05,  # помягче
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
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
