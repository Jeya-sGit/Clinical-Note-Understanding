from embeddings.gemini_embeddings import get_embedding
from vectorstore.mongo_vector import retrieve_chunks
from rag_retrival.generator import generate_answer


def run_rag(query):
    print("\n🔍 Query:", query)

    query_embedding = get_embedding(query)

    chunks = retrieve_chunks(query_embedding)

    print("\n📦 Retrieved Chunks:\n")
    for c in chunks:
        print(c["text"])
        print("Score:", c.get("score"))
        print("-----")

    answer = generate_answer(query, chunks)

    sources = list(set([c["source"] for c in chunks]))

    return answer, sources