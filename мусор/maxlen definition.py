from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
train_path= "../test_prepped1.jsonl"
model_id="sixteen_girl"
dataset = load_dataset("json", data_files={"train": train_path})
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Функция для форматирования сообщений, как у тебя
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
lengths = []
examples_texts = []

for batch in dataset["train"]:
    text = format_chat(batch["messages"])
    tokenized = tokenizer(text, truncation=False, padding=False)
    lengths.append(len(tokenized["input_ids"]))
    examples_texts.append(text)

# Максимальная длина
max_len = max(lengths)
# Средняя длина
avg_len = sum(lengths)/len(lengths)

print(f"Максимальная длина токенов в датасете: {max_len}")
print(f"Средняя длина токенов: {avg_len:.2f}")

# Топ-5 самых длинных примеров
top5_indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)[:40]
print("\nТоп 5 самых длинных примеров:")
for i in top5_indices:
    print(f"{i+1}. Длина: {lengths[i]} токенов")
    print(f"   Пример текста (первые 500 символов): {examples_texts[i]!r}\n")