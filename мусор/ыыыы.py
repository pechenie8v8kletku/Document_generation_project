import json

input_file = "../train.jsonl"  # твой исходный файл
output_file = "../train_prepped.jsonl"  # куда сохранять подготовленные примеры

with open(input_file, "r", encoding="utf-8") as f_in, \
     open(output_file, "w", encoding="utf-8") as f_out:

    for line in f_in:
        data = json.loads(line)

        # формируем prompt: объединяем system + user
        system_msgs = [m["content"] for m in data["messages"] if m["role"] == "system"]
        user_msgs = [m["content"] for m in data["messages"] if m["role"] == "user"]
        assistant_msgs = [m["content"] for m in data["messages"] if m["role"] == "assistant"]

        prompt = "\n".join(system_msgs + user_msgs)
        response = "\n".join(assistant_msgs)

        out_obj = {
            "prompt": prompt,
            "response": response
        }

        f_out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

print(f"Готово! Сохранено в {output_file}")
