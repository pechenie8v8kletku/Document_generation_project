
import json
from pathlib import Path
import re
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText
from sentence_transformers import SentenceTransformer

INPUT_JSON = "documents_by_code_updated.json"
OUTPUT_JSONL = "train_prepped.jsonl"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION = "standards_collection"

EMB_MODEL = "sbert_large_nlu_ru"
TOP_M_PER_GOST = 6
MAX_CHUNK_LEN = 600

FIELD_CODE = "код"
FIELD_DIRECTION = "направления"
FIELD_EXTRA = "доп_данные"
FIELD_GOSTS = "Госты"
FIELD_SECTIONS = "макет_разделы"

TARGET_SECTIONS = [
    "Основание для проведения испытаний:",
    "Объект испытаний",
    "Идентификационные сведения о представленной на испытания продукции",
    "Заказчик",
    "Изготовитель",
    "Методы испытаний:",
    "Испытательное оборудование и средства измерений:",
"Результаты испытаний:",
    "Условия проведения испытаний:",
    "Процедура испытаний:"
]

SYSTEM_MSG_TPL = (
    "Ты модель, которая генерирует раздел «{section}». "
    "Используй предоставленные данные: направление, дополнительные сведения и фрагменты ГОСТов. "
    "Пиши официальным стилем, деловым языком, опираясь на суть, а не на прямую копипасту."
)

def clean_text(s: str, max_len: int) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_len:
        s = s[:max_len] + "..."
    return s

def fetch_passages(client, model, query_text, gost_id, top_m):
    query_vec = model.encode(query_text).tolist()
    flt = Filter(must=[FieldCondition(key="source", match=MatchText(text=str(gost_id)))] )
    results = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_vec,
        limit=top_m,
        with_payload=True,
        query_filter=flt
    )
    return [clean_text(r.payload.get("text", ""), MAX_CHUNK_LEN) for r in results]

def make_user_prompt(section, direction, extra, gost_ids, ctx_map):
    lines = []
    lines.append(f"Задача: Сгенерируй раздел «{section}».\n")
    if direction:
        lines.append(f"Направление: {direction}")
    if extra:
        lines.append(f"Дополнительные данные: {extra}")
    if gost_ids:
        lines.append(f"Перечень ГОСТов: {', '.join(gost_ids)}")
    lines.append("\nКонтекст по ГОСТам:")
    for gid in gost_ids:
        frags = ctx_map.get(gid, [])
        if not frags:
            continue
        lines.append(f"[ГОСТ {gid}]")
        for i, ch in enumerate(frags, 1):
            lines.append(f"  ({i}) {ch}")
    lines.append("\nТребования: оформи связный раздел, указывай номера ГОСТов по смыслу, избегай выдумок.")
    return "\n".join(lines)

def build_messages(system_msg, user_msg, assistant_msg):
    return {
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg}
        ]
    }

def main():
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    model = SentenceTransformer(EMB_MODEL)

    data = json.loads(Path(INPUT_JSON).read_text(encoding="utf-8"))
    out = open(OUTPUT_JSONL, "w", encoding="utf-8")

    for obj in data:
        direction = obj.get(FIELD_DIRECTION, "").strip()
        extra = obj.get(FIELD_EXTRA, "").strip()
        gost_ids = [str(x) for x in obj.get(FIELD_GOSTS, [])]

        if not direction or not gost_ids:
            continue

        ctx_map = {}
        for gid in gost_ids:
            try:
                ctx_map[gid] = fetch_passages(client, model, direction, gid, TOP_M_PER_GOST)
            except Exception:
                ctx_map[gid] = []

        for section in TARGET_SECTIONS:
            target = obj.get(FIELD_SECTIONS, {}).get(section, "").strip()
            if not target:
                continue

            user_prompt = make_user_prompt(section, direction, extra, gost_ids, ctx_map)
            system_msg = SYSTEM_MSG_TPL.format(section=section)

            record = build_messages(system_msg, user_prompt, target)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    out.close()
    print(f"Готово: {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()
