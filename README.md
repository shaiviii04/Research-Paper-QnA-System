# 📄 RAG Based Research Paper Q&A System

A **local, cost-free Research Paper Question Answering system** built with **LangChain, FAISS, HuggingFace embeddings, and Streamlit**.

This project implements the **retrieval part of a RAG pipeline** using a **Fake LLM** for safe, deterministic, and debuggable outputs.

---

## 🚀 What It Does

* Loads research paper PDFs
* Splits them into chunks
* Generates embeddings (HuggingFace)
* Stores embeddings in FAISS
* Retrieves relevant chunks for a question
* Displays exact evidence (no generation)

---

## 🧠 Why Fake LLM?

* No hallucinations
* No API cost
* Easy to debug retrieval quality

-> The Fake LLM only formats retrieved text. It does NOT generate answers.

---

## 🏗️ Project Structure

researchqa/
├── app.py
├── ingest.py
├── fake_llm.py
├── requirements.txt
├── data/        # PDF files
├── vectorstore/ # FAISS index
└── README.md
```

## ⚙️ Setup

1. pip install -r requirements.txt

2. Add PDFs to `data/`, then build the vector store:
   python ingest.py

3. Run the app:
   streamlit run app.py



## 🔄 Future Upgrades

* Replace Fake LLM with OpenAI / Claude / Gemini
* Add PDF upload support
* Dockerize and deploy


---

Runs fully on CPU • No GPU • No API keys
