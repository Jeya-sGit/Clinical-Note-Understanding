from pymongo import MongoClient
import google.genai as genai
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

uri = MONGO_URI
client_genai = genai.Client(api_key=API_KEY)


client = MongoClient(uri)
db = client["clinical_rag"]
collection = db["embeddings"]


def retrieve_chunks(query, k=3):
    # 🔹 Query → embedding
    response = client_genai.models.embed_content(
    model="gemini-embedding-001",
    contents=query,
    config={
        "output_dimensionality": 768
    }
)
    print("Query embedding generated.",response)
    query_embedding = response.embeddings[0].values

    # 🔹 Vector search
    results = collection.aggregate([
        {
            "$vectorSearch": {
                "index": "default",
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


# 🔹 Test
query = "What medicines were prescribed?"

results = retrieve_chunks(query)

print("\nRetrieved Results:\n")

for r in results:
    print("Text:", r["text"][:120])
    print("Source:", r["source"])
    print("Score:", r["score"])
    print("-----")