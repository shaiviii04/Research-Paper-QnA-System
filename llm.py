import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.0-flash"

def _fallback_answer(question, docs):
    """No LLM. Just show best retrieved chunks + citations."""
    if not docs:
        return "Not found in the provided papers."

    lines = []
    lines.append("⚠️ Gemini API is currently unavailable (quota / rate-limit).")
    lines.append("Showing the most relevant extracted text from the papers instead.\n")

    lines.append(f"**Question:** {question}\n")
    lines.append("**Most relevant evidence:**\n")

    for i, doc in enumerate(docs[:3], start=1):
        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "?")

        text = doc.page_content.strip().replace("\n", " ")
        text = " ".join(text.split())
        text = text[:700]

        lines.append(f"[{i}] {source} (page {page})")
        lines.append(f"{text}\n")

    lines.append("Citations: [1] [2] [3]")
    return "\n".join(lines)


def llm(question, docs):
    if not API_KEY:
        return "❌ GEMINI_API_KEY not found. Add it in your .env file."

    # Build context for Gemini
    sources = []
    for i, doc in enumerate(docs, start=1):
        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "?")

        text = doc.page_content.strip().replace("\n", " ")
        text = " ".join(text.split())
        text = text[:900]

        sources.append(f"[{i}] {source} (page {page}): {text}")

    context = "\n\n".join(sources)

    prompt = f"""
You are a strict research assistant.

Rules:
- Answer ONLY using the SOURCES.
- If the answer is not present, say exactly: Not found in the provided papers.
- Keep it short (3–6 lines).
- End with: Citations: [1] [2] ...

SOURCES:
{context}

QUESTION:
{question}

ANSWER:
""".strip()

    try:
        client = genai.Client(api_key=API_KEY)

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        text = (response.text or "").strip()

        if not text:
            return _fallback_answer(question, docs)

        return text

    except Exception:
        # Any Gemini error (429, 403, etc.) → fallback
        return _fallback_answer(question, docs)
