# build_dataset.py
import json
from pathlib import Path
import re
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText
from sentence_transformers import SentenceTransformer

# ==== НАСТРОЙКИ ====
INPUT_JSON = "documents_by_code_updated.json"         # твой исходный файл
OUTPUT_JSONL = "train_prepped.jsonl"           # датасет для обучения
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION = "standards_collection"

EMB_MODEL = "sbert_large_nlu_ru"
TOP_M_PER_GOST = 6
MAX_CHUNK_LEN = 600

# Поля в твоём JSON
FIELD_CODE = "код"
FIELD_DIRECTION = "направления"
FIELD_EXTRA = "Доп_данные"
FIELD_GOSTS = "Госты"
FIELD_SECTIONS = "макет_разделы"
TARGET_SECTION = "Основание для проведения испытаний:"

SYSTEM_MSG = (
    "Ты модель, которая генерирует раздел «Основания для проведения испытаний». "
    "Используй предоставленные данные: направление, дополнительные сведения и фрагменты ГОСТов. "
    "Пиши официальным стилем, деловым языком, опираясь на суть, а не на прямую копипасту."
)

# ==== ВСПОМОГАТЕЛЬНО ====
def clean_text(s: str, max_len: int) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_len:
        s = s[:max_len] + "..."
    return s

def fetch_passages(client, model, query_text, gost_id, top_m):
    query_vec = model.encode(query_text).tolist()
    flt = Filter(must=[FieldCondition(key="source", match=MatchText(text=str(gost_id)))])
    results = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_vec,
        limit=top_m,
        with_payload=True,
        query_filter=flt
    )
    return [clean_text(r.payload.get("text", ""), MAX_CHUNK_LEN) for r in results]

def make_user_prompt(direction, extra, gost_ids, ctx_map):
    lines = []
    lines.append("Задача: Сгенерируй раздел «Основания для проведения испытаний».")
    lines.append("")
    lines.append(f"Направление: {direction}")
    if extra:
        lines.append(f"Дополнительные данные: {extra}")
    lines.append(f"Перечень ГОСТов: {', '.join(gost_ids)}")
    lines.append("")
    lines.append("Контекст по ГОСТам:")
    for gid in gost_ids:
        frags = ctx_map.get(gid, [])
        if not frags:
            continue
        lines.append(f"[ГОСТ {gid}]")
        for i, ch in enumerate(frags, 1):
            lines.append(f"  ({i}) {ch}")
    lines.append("")
    lines.append("Требования: оформи связный раздел, указывай номера ГОСТов по смыслу, избегай выдумок.")
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
        target = obj.get(FIELD_SECTIONS, {}).get(TARGET_SECTION, "").strip()
        print(gost_ids,target)
        if not direction or not gost_ids or not target:
            continue

        ctx_map = {}
        print(gost_ids)
        for gid in gost_ids:
            try:
                ctx_map[gid] = fetch_passages(client, model, direction, gid, TOP_M_PER_GOST)

            except Exception:
                ctx_map[gid] = []

        user_prompt = make_user_prompt(direction, extra, gost_ids, ctx_map)
        record = build_messages(SYSTEM_MSG, user_prompt, target)
        out.write(json.dumps(record, ensure_ascii=False) + "\n")

    out.close()
    print(f"Готово: {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()
