from docling.document_converter import DocumentConverter
import re
import html
import os
import json
from pathlib import Path


def clean_text_basic(text):
    text = html.unescape(text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'[^\S\r\n]+', ' ', text)
    text = re.sub(r'(?m)^[^A-Za-zА-Яа-я0-9]+$', '', text)

    def is_garbage(line):
        line = line.strip()
        return len(line) <= 15 and sum(c.isalnum() for c in line) < 8

    lines = text.splitlines()
    cleaned = [line for line in lines if not is_garbage(line)]
    return '\n'.join(cleaned).strip()


# Папки, где лежат документы
base_folders = {
    "направления": "Направление в мгсу/направления",
    "доп_данные": "Направление в мгсу/доп хуйня если есть",
    "макет": "Направление в мгсу/шаблоны"
}

# Хранилище всех документов по коду
docs_by_code = {}

converter = DocumentConverter()

import win32com.client as win32
from pathlib import Path

def convert_doc_to_docx(doc_path):
    word = win32.Dispatch("Word.Application")
    doc_path = Path(doc_path).resolve()
    doc = word.Documents.Open(str(doc_path))
    new_path = doc_path.with_suffix(".docx")
    doc.SaveAs(str(new_path), FileFormat=16)  # 16 = wdFormatDocumentDefault (.docx)
    doc.Close()
    word.Quit()
    return new_path





for section, folder_path in base_folders.items():
    if not os.path.exists(folder_path):
        print(f"⚠ Папка {folder_path} не найдена, пропускаем")
        continue

    for file_name in os.listdir(folder_path):
        file_path = Path(folder_path) / file_name

        if not file_path.is_file():
            continue

        # Берём первые 3 цифры из имени файла
        match = re.match(r"(\d{3})", file_name)
        if not match:
            print(f"⚠ Не удалось выделить код из {file_name}")
            continue

        code = match.group(1)

        try:
            if file_path.suffix.lower() == ".doc":
                file_path = convert_doc_to_docx(file_path)

            result = converter.convert(file_path.as_posix())
            text = result.document.export_to_markdown()
            text = clean_text_basic(text)
        except Exception as e:
            print(f"❌ Ошибка при обработке {file_path}: {e}")
            continue

        # Если такого кода ещё нет — создаём заготовку
        if code not in docs_by_code:
            docs_by_code[code] = {
                "код": code,
                "направления": "",
                "доп_данные": "",
                "макет": ""
            }

        docs_by_code[code][section] = text

# Превращаем в список для сохранения
output_data = list(docs_by_code.values())

# Сохраняем в JSON
json_path = "../documents_by_code.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"✅ Сохранено {len(output_data)} документов в {json_path}")
