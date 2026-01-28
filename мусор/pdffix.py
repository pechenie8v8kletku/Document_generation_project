import re
import html
import os
import json
from pathlib import Path
from pdf2image import convert_from_path
from fpdf import FPDF
import tempfile

def clean_text_basic(text: str) -> str:
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

# Папки с документами
base_folders = {
    "направления": "Направление в мгсу/направления",
    "доп_данные": "Направление в мгсу/доп хуйня если есть",
    "макет": "Направление в мгсу/шаблоны"
}

# Путь к JSON
json_path = "../documents_by_code.json"

# Загружаем существующую базу
if os.path.exists(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        try:
            existing_data = json.load(f)
        except json.JSONDecodeError:
            existing_data = []
else:
    existing_data = []

docs_by_code = {item["код"]: item for item in existing_data}

# Настройка OCR для PDF
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat

pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.do_table_structure = True
pipeline_options.ocr_options = EasyOcrOptions()
pipeline_options.ocr_options.lang = ["en", "ru"]

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

def pdf_to_image_pdf(original_pdf_path):
    """Конвертирует PDF в PDF-картинку без текстового слоя."""
    images = convert_from_path(original_pdf_path, dpi=300)
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

        if not file_path.is_file() or file_path.suffix.lower() != ".pdf":
            continue

        match = re.match(r"(\d{3})", file_name)
        if not match:
            print(f"⚠ Не удалось выделить код из {file_name}")
            continue
        code = match.group(1)

        try:
            # Убираем текстовый слой
            img_pdf_path = pdf_to_image_pdf(file_path.as_posix())

            # Кормим в docling
            result = converter.convert(img_pdf_path)
            text = result.document.export_to_markdown()
            text = clean_text_basic(text)

            # Удаляем временный PDF
            os.remove(img_pdf_path)

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

# Сохраняем обратно
output_data = list(docs_by_code.values())
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"✅ Обновлено/сохранено {len(output_data)} документов в {json_path}")
