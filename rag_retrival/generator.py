import google.genai as genai
import time
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("API_KEY")

client_genai = genai.Client(api_key=API_KEY)


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
