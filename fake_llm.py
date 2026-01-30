from typing import List

def fake_llm(question: str, docs: List) -> str:
    """
    Simple Fake LLM for debugging RAG pipelines.
    It does NOT generate new information.
    It only formats retrieved documents in a readable way.
    """

    if not docs:
        return "No relevant information found in the uploaded research papers."

    response = []
    response.append(f"QUESTION:\n{question}\n")
    response.append("RETRIEVED EVIDENCE FROM PAPERS:\n")

    for idx, doc in enumerate(docs, start=1):
        text = doc.page_content.strip().replace("\n", " ")
        text = " ".join(text.split())  # normalize spacing

        response.append(f"[Source {idx}]")
        response.append(text)
        response.append("")  # empty line

    response.append("NOTE:")
    response.append(
        "This response is generated using a fake LLM. "
        "It only displays retrieved content and does not generate or infer new information."
    )

    return "\n".join(response)
