import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.0-flash"

client = genai.Client(api_key=API_KEY)


def llm(question, docs):
    # Build context
    context_blocks = []
    for i, doc in enumerate(docs, start=1):
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page", "unknown")

        text = doc.page_content.strip().replace("\n", " ")
        context_blocks.append(f"[Source {i}] ({source}, page {page}) {text}")

    context = "\n\n".join(context_blocks)

    prompt = f"""
You are a helpful assistant answering questions ONLY using the provided sources.

Rules:
- If the answer is not present in the sources, say: "Not found in the provided papers."
- Keep the answer short (5-8 lines max).
- After the answer, add citations like: (Source 1), (Source 2).
- Do NOT use outside knowledge.

SOURCES:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text.strip()
