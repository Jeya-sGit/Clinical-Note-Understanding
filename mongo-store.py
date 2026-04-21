from pymongo import MongoClient
import google.genai as genai
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

client_genai = genai.Client(api_key=API_KEY)

models = client_genai.models.list()

for m in models:
    print(m.name)
# 🔹 MongoDB
client = MongoClient(MONGO_URI)
db = client["clinical_rag"]
collection = db["embeddings"]

# 🔹 Your chunks
chunks = [
    {"text": "Prescribed Azithromycin 500 mg", "source": "doc1"},
    {"text": "Ibuprofen 400 mg twice daily", "source": "doc2"},
]

docs = []

for chunk in chunks:
    response = client_genai.models.embed_content(
    model="gemini-embedding-001",
    contents=chunk["text"],
    config={
        "output_dimensionality": 768
    }
)

    embedding = response.embeddings[0].values

    docs.append({
        "text": chunk["text"],
        "source": chunk["source"],
        "embedding": embedding
    })

collection.insert_many(docs)

print("✅ Stored embeddings")