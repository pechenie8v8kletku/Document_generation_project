# build_qdrant_by_tokens.py
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from qdrant_client.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from tqdm import tqdm
import json

JSON_PATH = "just_md_ibm.json"
COLLECTION = "mgsu_collection"
BATCH = 64
MODEL_NAME = "sbert_large_nlu_ru"
CHUNK_SIZE_TOK = 500
CHUNK_OVERLAP_TOK = 50

def split_by_tokens(text, tok, size=500, overlap=50):
    ids = tok.encode(text, add_special_tokens=False)
    if not ids:
        return []
    step = max(1, size - overlap)
    chunks = []
    for st in range(0, len(ids), step):
        piece = ids[st:st+size]
        if not piece:
            break
        chunks.append(tok.decode(piece, skip_special_tokens=True))
    return chunks
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = SentenceTransformer(MODEL_NAME)
client = QdrantClient(host="localhost", port=6333)
if client.collection_exists(COLLECTION):
    client.delete_collection(COLLECTION)
dim = len(model.encode("тест", normalize_embeddings=True).tolist())
client.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
)

total_chunks = 0
for d in data:
    total_chunks += len(split_by_tokens(d.get("output",""), tok, CHUNK_SIZE_TOK, CHUNK_OVERLAP_TOK))

points, pid = [], 0
with tqdm(total=total_chunks, desc="Чанки", unit="chunk") as bar:
    for d in data:
        src = d.get("input","")
        txt = d.get("output","")
        chunks = split_by_tokens(txt, tok, CHUNK_SIZE_TOK, CHUNK_OVERLAP_TOK)
        if not chunks:
            continue
        vecs = model.encode(chunks, normalize_embeddings=True).tolist()
        for ch, vec in zip(chunks, vecs):
            points.append(PointStruct(
                id=pid,
                vector=vec,
                payload={"source": src, "text": ch}
            ))
            pid += 1
            bar.update(1)

            if len(points) >= BATCH:
                client.upsert(collection_name=COLLECTION, points=points)
                points = []

if points:
    client.upsert(collection_name=COLLECTION, points=points)

print(f"[✓] Всего добавлено точек: {pid}")
