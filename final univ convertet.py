import re
import html
import os
import json
from pathlib import Path
from pdf2image import convert_from_path
from fpdf import FPDF
import tempfile

from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat

def clean_text_basic(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'[^\S\r\n]+', ' ', text)
    text = re.sub(r'(?m)^[^A-Za-zА-Яа-я0-9]+$', '', text)

    def is_garbage(line):
        line = line.strip()
        return len(line) <= 5 and sum(c.isalnum() for c in line) < 8

    lines = text.splitlines()
    cleaned = [line for line in lines if not is_garbage(line)]
    return '\n'.join(cleaned).strip()

base_folders = {
    "направления": "дополнение в датасет/направления",
    "доп_данные": "дополнение в датасет/доп хуйня если есть",
    "макет": "дополнение в датасет/шаблоны"
}


json_path = "train_addition_part.json"
if os.path.exists(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        try:
            existing_data = json.load(f)
        except json.JSONDecodeError:
            existing_data = []
else:
    existing_data = []
docs_by_code = {item["код"]: item for item in existing_data}

pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.do_table_structure = True
pipeline_options.ocr_options = EasyOcrOptions()
pipeline_options.ocr_options.lang = ["en", "ru"]

pdf_converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)

import win32com.client as win32

def convert_doc_to_docx(doc_path):
    word = win32.Dispatch("Word.Application")
    doc_path = Path(doc_path).resolve()
    doc = word.Documents.Open(str(doc_path))
    new_path = doc_path.with_suffix(".docx")
    doc.SaveAs(str(new_path), FileFormat=16)  # docx
    doc.Close()
    word.Quit()
    return new_path

doc_converter = DocumentConverter()
def extract_tables_markdown(pdf_path: str, pages="1-end"):
    try:
        import camelot
    except ImportError:
        return ""

    md_parts = []
    for flavor in ("lattice", "stream"):
        try:
            tables = camelot.read_pdf(pdf_path, pages=pages, flavor=flavor)
        except Exception:
            continue
        if tables.n <= 0:
            continue
        for i, t in enumerate(tables):
            try:
                md_parts.append(f"\n\n**Таблица ({flavor}) #{i+1}**\n\n" + t.df.to_markdown(index=False))
            except Exception:
                pass
        if md_parts:
            break
    return "\n".join(md_parts)

def pdf_to_image_pdf(original_pdf_path):
    images = convert_from_path(original_pdf_path, dpi=400)
    temp_pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    pdf = FPDF(unit="pt", format=[images[0].width, images[0].height])
    for img in images:
        pdf.add_page()
        img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        img.save(img_path, "PNG")
        pdf.image(img_path, 0, 0)
        os.remove(img_path)
    pdf.output(temp_pdf_path)
    return temp_pdf_path

for section, folder_path in base_folders.items():
    if not os.path.exists(folder_path):
        print(f"⚠ Папка {folder_path} не найдена, пропускаем")
        continue

    for file_name in os.listdir(folder_path):
        file_path = Path(folder_path) / file_name
        if not file_path.is_file():
            continue

        match = re.match(r"(\d{3})", file_name)
        if not match:
            print(f"⚠ Не удалось выделить код из {file_name}")
            continue
        code = match.group(1)

        try:
            text = ""

            # === PDF обработка ===
            if file_path.suffix.lower() == ".pdf":
                img_pdf_path = pdf_to_image_pdf(file_path.as_posix())
                result = pdf_converter.convert(img_pdf_path)
                text = result.document.export_to_markdown()
                tables_md = extract_tables_markdown(file_path.as_posix())
                if tables_md:
                    text = text + "\n\n" + tables_md
                os.remove(img_pdf_path)

            # === DOC/DOCX обработка ===
            elif file_path.suffix.lower() in [".doc", ".docx"]:
                if file_path.suffix.lower() == ".doc":
                    file_path = convert_doc_to_docx(file_path)
                result = doc_converter.convert(file_path.as_posix())
                text = result.document.export_to_markdown()

            else:
                print(f"⚠ Пропускаем неподдерживаемый формат: {file_path.suffix}")
                continue

            text = clean_text_basic(text)

        except Exception as e:
            print(f"❌ Ошибка при обработке {file_path}: {e}")
            continue

        if code not in docs_by_code:
            docs_by_code[code] = {
                "код": code,
                "направления": "",
                "доп_данные": "",
                "макет": ""
            }
        docs_by_code[code][section] = text

output_data = list(docs_by_code.values())
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"✅ Обновлено/сохранено {len(output_data)} документов в {json_path}")
