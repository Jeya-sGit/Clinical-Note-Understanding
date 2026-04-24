from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import faiss
import ollama


# 🔹 Build index ONCE
def build_index(chunks):
    texts = [chunk["text"] for chunk in chunks]

    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(texts).toarray()

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))

    return index, vectorizer


# 🔹 Retrieve relevant chunks
def retrieve_chunks(query, index, vectorizer, chunks, k=3):
    query_embedding = vectorizer.transform([query]).toarray()

    distances, indices = index.search(
        np.array(query_embedding).astype('float32'), k
    )

    results = []
    for i, dist in zip(indices[0], distances[0]):
        if dist < 1.5:   # 🔥 threshold (tune if needed)
            results.append(chunks[i])

    # fallback if nothing passes filter
    if not results:
        results.append(chunks[indices[0][0]])

    return results


# 🔹 Generate answer using LLM
def generate_answer(query, retrieved_chunks):
    # structured context
    context = "\n\n".join([
        f"Source: {chunk['source']}\n{chunk['text']}"
        for chunk in retrieved_chunks
    ])

    # guardrail
    if len(context.strip()) < 20:
        return "I don't know"

    prompt = f"""
Extract ONLY medicine names from the context.

STRICT RULES:
- Return comma-separated values
- No explanation
- No sentences
- No extra text
- If none: I don't know

Context:
{context}

Question:
{query}

Answer:
"""

    response = ollama.chat(
        model='phi',
        messages=[
            {
                "role": "system",
                "content": "Answer ONLY from context. Do not generate extra text."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )


    answer = response['message']['content'].strip()
    answer = answer.split("\n")[0]

    return answer


# 🔹 Main RAG function
def run_rag(index, vectorizer, chunks, query):
    retrieved_chunks = retrieve_chunks(query, index, vectorizer, chunks)

    # debug (optional)
    print("\nRetrieved Chunks:\n")
    for c in retrieved_chunks:
        print(c["text"][:120], "...\n")

    answer = generate_answer(query, retrieved_chunks)

    sources = list(set([chunk["source"] for chunk in retrieved_chunks]))

    return {
        "answer": answer,
        "sources": sources
    }