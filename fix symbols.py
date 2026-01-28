# fix_json_text.py
import json, re, unicodedata
from typing import Any

INPUT_JSON  = r"new_train.json"
OUTPUT_JSON = r"new_train_fixed.json"

PROCESS_ALL_FIELDS = True
FIELDS_WHITELIST = {
    "направления", "доп_данные", "макет"
}

# --- декод /uniXXXX, /uXXXX, \uXXXX, &#xXXXX; ---
HEX_PATTERNS = [
    re.compile(r'(?i)/?uni([0-9a-f]{4})'),
    re.compile(r'(?i)/?u([0-9a-f]{4})'),
    re.compile(r'(?i)\\u([0-9a-f]{4})'),
    re.compile(r'(?i)&#x([0-9a-f]{2,6});'),
]
def _hex_to_char(m: re.Match) -> str:
    try: return chr(int(m.group(1), 16))
    except Exception: return m.group(0)

def decode_glyph_names(s: str) -> str:
    prev = None
    while prev != s:
        prev = s
        for rx in HEX_PATTERNS:
            s = rx.sub(_hex_to_char, s)
    return unicodedata.normalize("NFC", s)

# --- замена латинских look-alikes на кириллицу, если кириллицы больше ---
LAT2CYR = str.maketrans({
    'A':'А','B':'В','C':'С','E':'Е','H':'Н','K':'К','M':'М','O':'О','P':'Р','T':'Т','X':'Х',
    'a':'а','e':'е','o':'о','p':'р','c':'с','x':'х','y':'у'
})
def fix_lookalikes_if_cyrillic(s: str) -> str:
    cyr = sum('А' <= ch <= 'я' or ch in 'ёЁ' for ch in s)
    lat = sum('A' <= ch <= 'z' or 'A' <= ch <= 'Z' for ch in s)
    return s.translate(LAT2CYR) if cyr >= lat else s

# --- базовая чистка ---
def tidy_spaces(s: str) -> str:
    s = s.replace('\ufeff', '').replace('\u00A0', ' ')
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'\n{2,}', '\n', s)
    s = re.sub(r'(?m)^[^\w#]+$', '', s)
    return s.strip()

def fix_string(s: str) -> str:
    s = decode_glyph_names(s)
    s = fix_lookalikes_if_cyrillic(s)
    s = tidy_spaces(s)
    return s

# --- рекурсивная обработка ---
def fix_obj(obj: Any, parent_key: str | None = None) -> Any:
    if isinstance(obj, dict):
        return {k: fix_obj(v, k) for k, v in obj.items()}
    if isinstance(obj, list):
        return [fix_obj(x, parent_key) for x in obj]
    if isinstance(obj, str):
        if PROCESS_ALL_FIELDS or (parent_key in FIELDS_WHITELIST):
            return fix_string(obj)
        else:
            return obj
    return obj

def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    fixed = fix_obj(data)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(fixed, f, ensure_ascii=False, indent=2)
    print(f"✔ saved: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
