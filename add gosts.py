import json

json_file = "new_train_fixed.json"
txt_file = "направление госты.txt"
output_file = "new_train_fixed.json"

with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

data_by_code = {item["код"]: item for item in data}

with open(txt_file, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if not parts:
            continue

        code = (parts[0])
        print(code)
        gosts = parts[1:]

        if code in data_by_code:
            print("pisya")
            data_by_code[code]["Госты"] = gosts

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(list(data_by_code.values()), f, ensure_ascii=False, indent=2)

print(f"Файл обновлен и сохранён как {output_file}")
