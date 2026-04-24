from pymongo import MongoClient
import google.genai as genai
import time
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("API_KEY")
MONGO_URI = os.getenv("MONGO_URI")


client_genai = genai.Client(api_key=API_KEY)
mongo_client = MongoClient(MONGO_URI)

db = mongo_client["clinical_rag"]
collection = db["embeddings"]


def get_embedding(text):
    for i in range(3):
        try:
            response = client_genai.models.embed_content(
                model="gemini-embedding-001",
                contents=text,
                config={"output_dimensionality": 768}
            )

            '''for i in response.embeddings:
                print("Embedding response:", i)'''
            return response.embeddings[0].values
        
        except Exception as e:
            print(f"⚠️ Retry {i+1} failed:", e)
            time.sleep(2)

    raise Exception("❌ Embedding failed after retries")


def retrieve_chunks(query, k=2):
    query_embedding = get_embedding(query)

    results = collection.aggregate([
        {
            "$vectorSearch": {
                "index": "default",  # name that we provided while creatingthe vector search index
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 50,
                "limit": k
            }
        },
        {
            "$project": {
                "text": 1,
                "source": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ])
    return list(results)

def generate_answer(query, chunks):
    context = "\n\n".join([c["text"] for c in chunks])


    if len(context.strip()) < 20:
        return "I don't know"

    prompt = f"""
Extract ONLY medicine names from the context.

Rules:
- Comma-separated
- No explanation
- No extra text
- If not found: I don't know

Context:
{context}

Question:
{query}

Answer:
"""

    for i in range(3):
        try:
            response = client_genai.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            print(f"⚠️ LLM Retry {i+1} failed:", e)
            time.sleep(2)

    return "I don't know"


def run_rag(query):
    print("\n🔍 Query:", query)

    chunks = retrieve_chunks(query)

    print("\n📦 Retrieved Chunks:\n")
    for c in chunks:
        print(c["text"])
        print("Score:", c.get("score"))
        print("-----")

    answer = generate_answer(query, chunks)

    sources = list(set([c["source"] for c in chunks]))

    return answer, sources


if __name__ == "__main__":
    query = "What medicines were prescribed?"

    answer, sources = run_rag(query)

    print("\n✅ Final Answer:\n", answer)
    print("\n📄 Sources:\n", sources)