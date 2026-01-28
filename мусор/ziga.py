import re
import json

def group_sections(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    text = re.sub(r'\s+', ' ', text)
    pattern = re.compile(r'\b1\.(\d+)\b')
    matches = list(pattern.finditer(text))

    sections = {}
    expected = 1

    for i in range(len(matches)):
        current_num = int(matches[i].group(1))
        if current_num != expected:
            continue

        key = f"1.{expected}"
        start = matches[i].start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        sections[key] = content
        expected += 1

    return sections

def append_to_json_file(json_path, new_entries):
    try:
        with open(json_path, "r", encoding="utf-8") as jf:
            data = json.load(jf)
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        data = []

    data.extend(new_entries)

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(data, jf, ensure_ascii=False, indent=4)

def make_json_entries(sections):
    entries = []
    for content in sections.values():
        before_semicolon = content.split(";")[0].strip()
        input_text = f"Выдай ГОСТ: {before_semicolon}"
        output_text = content.strip()
        entries.append({
            "input": input_text,
            "output": output_text
        })
    return entries

txt_path = "../cop 3.txt"
json_path = "../sample.json"

sections = group_sections(txt_path)
json_entries = make_json_entries(sections)
append_to_json_file(json_path, json_entries)
