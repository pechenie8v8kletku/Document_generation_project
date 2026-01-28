from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
client = QdrantClient(host="localhost", port=6333)
collections = client.get_collections()

query = "1. МЕТОД ОПРЕДЕЛЕНИЯ ПРЕДЕЛА ПРОЧНОСТИ ПРИ СЖАТИИ цов, их количество и точность изготовления  "
model = SentenceTransformer("sbert_large_nlu_ru")
query_vector = model.encode(query).tolist()
sub =" Гост ГОСТ 9623: "


from qdrant_client.models import Filter, FieldCondition, MatchText

filtered_results = client.search(
    collection_name="standards_collection",
    query_vector=query_vector,
    limit=3,
    with_payload=True,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="source",
                match=MatchText(text="9623")
            )
        ]
    )
)

print("\n=== Отфильтрованные результаты ===")
for res in filtered_results:
    print(f"Score: {res.score:.4f}")
    print(f"Source: {res.payload['source'][:100]}")
    print(f"Text: {res.payload['text'][:100]}...\n")