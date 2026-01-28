#кусочки:
# титульник
# 1. Основание для проведения испытаний:
# 2. Объект испытаний
# 3. Заказчик
# Изготовитель:
# Идентификационные сведения о представленной на испытания продукции
# Методы испытаний:
# Условия проведения испытаний:
# Процедура испытаний:
# Испытательное оборудование и средства измерений:
# Результаты испытаний:
#
import json
import re
from pathlib import Path
from multiprocessing import Pool, cpu_count

json_path = Path("new_train_fixed.json")

sections = [
    ("титульник", None),
    ("Основание для проведения испытаний:", "Основание для проведения испытаний:"),
    ("Объект испытаний", "Объект испытаний"),
    ("Заказчик", "Заказчик"),
    ("Изготовитель", "Изготовитель:"),
    ("Идентификационные сведения о представленной на испытания продукции",
        "Идентификационные сведения о представленной на испытания продукции"),
    ("Методы испытаний:", "Методы испытаний:"),
    ("Условия проведения испытаний:", "Условия проведения испытаний:"),
    ("Процедура испытаний:", "Процедура испытаний:"),
    ("Испытательное оборудование и средства измерений:",
        "Испытательное оборудование и средства измерений:"),
    ("Результаты испытаний:", "Результаты испытаний:")
]


def split_maket(text):
    if not text:
        return {name: "" for name, _ in sections}

    parts = {}
    positions = {}

    for name, marker in sections:
        if marker is None:
            positions[name] = 0
        else:
            match = re.search(re.escape(marker), text)
            positions[name] = match.start() if match else None

    ordered = [name for name, _ in sections if positions[name] is not None]
    ordered_positions = sorted(ordered, key=lambda n: positions[n])

    for i, sec in enumerate(ordered_positions):
        start = positions[sec]
        end = positions[ordered_positions[i + 1]] if i + 1 < len(ordered_positions) else len(text)
        parts[sec] = text[start:end].strip()
    if "титульник" not in parts:
        first_marker_pos = min([p for p in positions.values() if p is not None and p > 0], default=None)
        if first_marker_pos is not None:
            parts["титульник"] = text[:first_marker_pos].strip()
        else:
            parts["титульник"] = text.strip()
    for name, _ in sections:
        if name not in parts:
            parts[name] = ""

    return parts


def process_doc(doc):
    maket_text = doc.get("макет", "")
    doc["макет_разделы"] = split_maket(maket_text)
    return doc


if __name__ == "__main__":
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with Pool(cpu_count()) as pool:
        data = pool.map(process_doc, data)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ Разделы макетов добавлены в JSON ({len(data)} документов обработано)")

