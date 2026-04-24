import google.genai as genai
from dotenv import load_dotenv
import os
import time

load_dotenv()

client_genai = genai.Client(api_key=os.getenv("API_KEY"))

def get_embedding(text):
    for i in range(3):
        try:
            response = client_genai.models.embed_content(
                model="gemini-embedding-001",
                contents=text,
                config={"output_dimensionality": 768}
            )
            return response.embeddings[0].values
        except Exception as e:
            print(f"⚠️ Retry {i+1} failed:", e)
            time.sleep(2)

    raise Exception("❌ Embedding failed after retries")