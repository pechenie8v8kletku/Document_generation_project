import json, re, random
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

INPUT_JSON = "new_train_fixed.json"
OUTPUT_JSONL = "train_prepped.jsonl"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION = "mgsu_collection"

EMB_MODEL = "sbert_large_nlu_ru"
RERANKER_MODEL = "jinaai/jina-reranker-v2-base-multilingual"
TOP_M_PER_GOST = 6
FETCH_PER_GOST = TOP_M_PER_GOST * 3
MAX_CHUNK_TOKENS = 600

FIELD_CODE = "код"
FIELD_DIRECTION = "направления"
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
    "Используй направление и фрагменты ГОСТов. "
    "Пиши официальным деловым стилем, опираясь на смысл без прямой копипасты."
)

PROMPT_VARIANTS = [
    "Сгенерируй раздел «{section}».",
    "Подготовь раздел «{section}».",
    "Составь раздел «{section}».",
    "Сформируй раздел «{section}».",
    "Опиши раздел «{section}».",
    "Напиши раздел «{section}».",
    "Собери раздел «{section}».",
    "Сверстай раздел «{section}».",
    "Скомпонуй раздел «{section}».",
    "Сделай раздел «{section}».",
]

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except Exception:
    def count_tokens(text: str) -> int:
        return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))

def trim_to_tokens(text: str, max_tokens: int) -> str:
    text = normalize_ws(text)
    n = count_tokens(text)
    if n <= max_tokens:
        return text
    ratio = max_tokens / max(n, 1)
    cut = max(1, int(len(text) * ratio))
    return text[:cut].rstrip() + "…"

def fetch_passages(client, enc_model, reranker, section, direction, gost_id, fetch_n, keep_top_m, max_tokens):
    query_text = f"{section}. {direction}" if direction else section
    query_vec = enc_model.encode(query_text).tolist()
    flt = Filter(must=[FieldCondition(key="source", match=MatchText(text=str(gost_id)))])
    results = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_vec,
        limit=fetch_n,
        with_payload=True,
        query_filter=flt
    )
    cand = []
    for r in results:
        txt = r.payload.get("text", "")
        if not txt:
            continue
        cand.append(normalize_ws(txt))

    if not cand:
        return []
    if reranker is not None and len(cand) > keep_top_m:
        pairs = [(query_text, c) for c in cand]
        scores = reranker.predict(pairs, convert_to_numpy=True)
        ranked = sorted(zip(cand, scores), key=lambda x: x[1], reverse=True)
        cand = [c for c, _ in ranked[:keep_top_m]]
    else:
        cand = cand[:keep_top_m]

    return [trim_to_tokens(c, max_tokens) for c in cand]

def make_user_prompt(section, direction, gost_ids, ctx_map):
    head = random.choice(PROMPT_VARIANTS).format(section=section)
    lines = [head, ""]
    if direction:
        lines.append(f"Направление: {direction}")
    if gost_ids:
        lines.append(f"ГОСТы: {', '.join(gost_ids)}")
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
    lines.append("Требования: оформи связный раздел; указывай номера ГОСТов по смыслу; избегай выдумок.")
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
    enc_model = SentenceTransformer(EMB_MODEL)
    try:
        reranker = CrossEncoder(RERANKER_MODEL, trust_remote_code=True)
    except Exception as e:
        print(f"[i] Реранкер не загружен ({e}). Будет только cosine top-k.")
        reranker = None

    data = json.loads(Path(INPUT_JSON).read_text(encoding="utf-8"))
    out = open(OUTPUT_JSONL, "w", encoding="utf-8")

    for obj in data:
        direction = (obj.get(FIELD_DIRECTION) or "").strip()
        gost_ids = [str(x) for x in (obj.get(FIELD_GOSTS) or [])]
        if not direction or not gost_ids:
            continue

        for section in TARGET_SECTIONS:
            target = (obj.get(FIELD_SECTIONS, {}) or {}).get(section, "")
            target = (target or "").strip()
            if not target:
                continue
            ctx_map = {}
            for gid in gost_ids:
                try:
                    ctx_map[gid] = fetch_passages(
                        client,
                        enc_model,
                        reranker,
                        section=section,
                        direction=direction,
                        gost_id=gid,
                        fetch_n=FETCH_PER_GOST,
                        keep_top_m=TOP_M_PER_GOST,
                        max_tokens=MAX_CHUNK_TOKENS
                    )
                except Exception as e:
                    print(f"[i] fetch_passages error for GOST {gid}: {e}")
                    ctx_map[gid] = []

            user_prompt = make_user_prompt(section, direction, gost_ids, ctx_map)
            system_msg = SYSTEM_MSG_TPL.format(section=section)

            record = build_messages(system_msg, user_prompt, target)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    out.close()
    print(f"Готово: {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()
