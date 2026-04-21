from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

try:
    client = MongoClient(MONGO_URI)

    
    db = client["clinical_rag"]
    collection = db["test_collection"]

    collection.insert_one({"status": "connected"})

    print("✅ MongoDB connected successfully!")

except Exception as e:
    print("❌ Connection failed:", e)