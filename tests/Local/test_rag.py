from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import faiss
import ollama

# Sample clinical data
documents = [
    "Patient has fever and was prescribed Dolo 650",
    "Patient reports headache and nausea",
    "Blood pressure is normal"
]

# Create embeddings
vectorizer = TfidfVectorizer()
doc_embeddings = vectorizer.fit_transform(documents).toarray()
print(f"Document embeddings shape: {doc_embeddings.shape}")

# FAISS index
dimension = doc_embeddings.shape[1]
print(f"Embedding dimension: {dimension}")
index = faiss.IndexFlatL2(dimension)
print(f"Is the index trained? {index.is_trained}")  
index.add(np.array(doc_embeddings).astype('float32'))
print(f"Number of documents in index: {index.ntotal}")

# Query
query = "What medicine was given?"
query_embedding = vectorizer.transform([query]).toarray()

# Search
distances, indices = index.search(np.array(query_embedding).astype('float32'), k=2)

# Context
context = "\n".join([documents[i] for i in indices[0]])

# Prompt
prompt = f"""
You are a medical assistant.

STRICT INSTRUCTIONS:
- Answer ONLY from the given context
- Do NOT use prior knowledge
- Do NOT guess or assume anything
- If the answer is not explicitly present, reply exactly: "I don't know"

Return ONLY the final answer.
Do NOT explain.

Context:
{context}

Question:
{query}

Answer:
"""

# LLM
response = ollama.chat(
    model='phi',
    messages=[{'role': 'user', 'content': prompt}]
)

print("\nAnswer:\n", response['message']['content'])