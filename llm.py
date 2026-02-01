import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

MODEL_NAME = "gemini-2.5-flash"


def llm(question, docs):
    context = "\n\n".join([doc.page_content for doc in docs])
    print("CONTEXT PREVIEW:\n", context[:500])
    prompt = f"""
You are a research assistant.
Answer ONLY using the provided context.
If the answer is not in the context, say:
"Not found in the provided papers."

Context:
{context}

Question:
{question}

Answer in 3–4 clear sentences.
"""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text.strip()