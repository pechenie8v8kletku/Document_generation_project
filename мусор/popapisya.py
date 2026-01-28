from docling.document_converter import DocumentConverter
import re
import html
import os
from мусор.ziga import append_to_json_file
from pathlib import Path
def clean_text_basic(text):
    text = html.unescape(text)

    text = re.sub(r'[ \t]+', ' ', text)

    text = re.sub(r'\n{2,}', '\n', text)

    text = re.sub(r'[^\S\r\n]+', ' ', text)

    text = re.sub(r'(?m)^[^A-Za-zА-Яа-я0-9]+$', '', text)

    def is_garbage(line):
        line = line.strip()
        return (
            len(line) <= 15 and
            sum(c.isalnum() for c in line) < 8
        )

    lines = text.splitlines()
    cleaned = [line for line in lines if not is_garbage(line)]

    return '\n'.join(cleaned).strip()





input_folder="downloaded_gosts"
files=os.listdir(input_folder)
json_path = "../just_md_ibm.json"
entries=[]
for a in files:
    file_path = Path(input_folder).joinpath(a)
    converter=DocumentConverter()
    result=converter.convert(file_path.as_posix())
    text=result.document.export_to_markdown()
    text=clean_text_basic(text)
    Prompt = f" Гост {os.path.splitext(a)[0]}: "
    entr = {
        "input": Prompt,
        "output": text
    }
    entries.append(entr)

append_to_json_file(json_path,entries)