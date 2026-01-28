import re
import html
import os
import json
from pathlib import Path
from pdf2image import convert_from_path
from fpdf import FPDF
import tempfile
import fitz
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
import unicodedata

# ---------- текстовые утилиты ----------

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

_HEX_REPS = [
    r'(?i)/?uni([0-9a-f]{4})',
    r'(?i)/?u([0-9a-f]{4})',
    r'(?i)\\u([0-9a-f]{4})',
    r'(?i)&#x([0-9a-f]{2,6});',
]
def _hex_to_char(m: re.Match) -> str:
    try:
        return chr(int(m.group(1), 16))
    except Exception:
        return m.group(0)

def decode_pdf_glyph_names(text: str) -> str:
    prev = None
    s = text
    while prev != s:
        prev = s
        for pat in _HEX_REPS:
            s = re.sub(pat, _hex_to_char, s)
    return unicodedata.normalize("NFC", s)

_LAT2CYR = str.maketrans({
    'A':'А','B':'В','C':'С','E':'Е','H':'Н','K':'К','M':'М','O':'О','P':'Р','T':'Т','X':'Х',
    'a':'а','e':'е','o':'о','p':'р','c':'с','x':'х','y':'у'
})
def fix_lookalikes_if_cyrillic(text: str) -> str:
    cyr = sum('А' <= ch <= 'я' or ch in 'ёЁ' for ch in text)
    lat = sum('A' <= ch <= 'z' or 'A' <= ch <= 'Z' for ch in text)
    if cyr > lat:
        return text.translate(_LAT2CYR)
    return text

def is_pdf_textual(pdf_path: str, max_pages_check: int = 3) -> bool:
    try:
        with fitz.open(pdf_path) as doc:
            n = min(len(doc), max_pages_check)
            for i in range(n):
                if doc[i].get_text("text").strip():
                    return True
    except Exception:
        pass
    return False

def pdf_to_image_pdf(original_pdf_path: str, dpi: int = 400) -> str:
    images = convert_from_path(original_pdf_path, dpi=dpi)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp_pdf_path = tmp.name
    tmp.close()
    pdf = FPDF(unit="pt", format=[images[0].width, images[0].height])
    for img in images:
        pdf.add_page()
        img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        img.save(img_path, "PNG")
        pdf.image(img_path, 0, 0)
        os.remove(img_path)
    pdf.output(tmp_pdf_path)
    return tmp_pdf_path


base_folders = {
    "макет": "только макеты для self supervised"
}

json_path = "self_supervised.json"
if os.path.exists(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        try:
            existing_data = json.load(f)
        except json.JSONDecodeError:
            existing_data = []
else:
    existing_data = []
docs_by_code = {item["код"]: item for item in existing_data}


# 1) без OCR
opts_no_ocr = PdfPipelineOptions()
opts_no_ocr.do_ocr = False
opts_no_ocr.do_table_structure = True
pdf_conv_no_ocr = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts_no_ocr)}
)

# 2) с EasyOCR (ru+en)
opts_easy = PdfPipelineOptions()
opts_easy.do_ocr = True
opts_easy.do_table_structure = True
opts_easy.ocr_options = EasyOcrOptions()
opts_easy.ocr_options.lang = ["ru", "en"]
pdf_conv_easyocr = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts_easy)}
)

from docling.document_converter import DocumentConverter as DocConverter
doc_converter = DocConverter()

import win32com.client as win32
def convert_doc_to_docx(doc_path):
    word = win32.Dispatch("Word.Application")
    doc_path = Path(doc_path).resolve()
    doc = word.Documents.Open(str(doc_path))
    new_path = doc_path.with_suffix(".docx")
    doc.SaveAs(str(new_path), FileFormat=16)
    doc.Close()
    word.Quit()
    return new_path

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

for section, folder_path in base_folders.items():
    if not os.path.exists(folder_path):
        print(f"⚠ Папка {folder_path} не найдена, пропускаем")
        continue

    for file_name in os.listdir(folder_path):
        file_path = Path(folder_path) / file_name
        if not file_path.is_file():
            continue

        m = re.match(r"^(\d{1,3})(?=\D|$)", file_name)
        if not m:
            print(f"⚠ Не удалось выделить код из {file_name}")
            continue
        code = m.group(1).zfill(3)

        try:
            text = ""

            if file_path.suffix.lower() == ".pdf":
                textual = is_pdf_textual(file_path.as_posix())

                if textual:
                    result = pdf_conv_no_ocr.convert(file_path.as_posix())
                    text = result.document.export_to_markdown()
                    if len(re.findall(r'(?i)/uni[0-9a-f]{4}', text)) >= 3:
                        try:
                            result = pdf_conv_easyocr.convert(file_path.as_posix())
                            text = result.document.export_to_markdown()
                        except Exception as e:
                            print(f"↪ OCR по исходному PDF не удался ({e}), пробуем image-PDF…")
                            tmp_img_pdf = pdf_to_image_pdf(file_path.as_posix(), dpi=400)
                            try:
                                result = pdf_conv_easyocr.convert(tmp_img_pdf)
                                text = result.document.export_to_markdown()
                            finally:
                                try: os.remove(tmp_img_pdf)
                                except: pass
                    else:
                        text = decode_pdf_glyph_names(text)
                        text = fix_lookalikes_if_cyrillic(text)

                    tables_md = extract_tables_markdown(file_path.as_posix())
                    if tables_md:
                        text += "\n\n" + tables_md

                else:
                    try:
                        result = pdf_conv_easyocr.convert(file_path.as_posix())
                        text = result.document.export_to_markdown()
                    except Exception as e:
                        print(f"↪ OCR по исходному PDF не удался ({e}), пробуем image-PDF…")
                        tmp_img_pdf = pdf_to_image_pdf(file_path.as_posix(), dpi=400)
                        try:
                            result = pdf_conv_easyocr.convert(tmp_img_pdf)
                            text = result.document.export_to_markdown()
                        finally:
                            try: os.remove(tmp_img_pdf)
                            except: pass

            elif file_path.suffix.lower() in [".doc", ".docx"]:
                src = file_path
                if src.suffix.lower() == ".doc":
                    src = convert_doc_to_docx(src)
                result = doc_converter.convert(src.as_posix())
                text = result.document.export_to_markdown()
                print("word")

            else:
                print(f"⚠ Пропускаем неподдерживаемый формат: {file_path.suffix}")
                continue

            text = text

        except Exception as e:
            print(f"❌ Ошибка при обработке {file_path}: {e}")
            continue

        if code not in docs_by_code:
            docs_by_code[code] = {"код": code, "макет": ""}

        docs_by_code[code][section] = text

output_data = list(docs_by_code.values())
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"✅ Обновлено/сохранено {len(output_data)} документов в {json_path}")
