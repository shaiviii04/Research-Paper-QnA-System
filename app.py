import os
import shutil
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from llm import llm  # your Gemini LLM function


VECTORSTORE_PATH = "vectorstore"
UPLOAD_DIR = "uploaded_pdfs"


st.set_page_config(page_title="ML Research Q&A", layout="centered")
st.title("📄 Machine Learning Research Q&A")
st.write("Ask questions based on research papers (preloaded OR uploaded).")


# -----------------------------
# Embeddings (same for both modes)
# -----------------------------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# -----------------------------
# Load prebuilt vectorstore
# -----------------------------
@st.cache_resource
def load_prebuilt_vectorstore():
    embeddings = get_embeddings()
    db = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return db


# -----------------------------
# Build vectorstore from uploaded PDFs
# -----------------------------
def build_vectorstore_from_uploads():
    # Clean upload folder
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # Save uploaded files
    for file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

    # Load PDFs
    loader = PyPDFDirectoryLoader(UPLOAD_DIR)
    documents = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    # Build FAISS
    embeddings = get_embeddings()
    db = FAISS.from_documents(chunks, embeddings)
    return db


# -----------------------------
# UI mode selector
# -----------------------------
mode = st.radio(
    "Choose document source:",
    ["Use preloaded papers", "Upload my own PDFs"]
)

db = None

if mode == "Use preloaded papers":
    db = load_prebuilt_vectorstore()

else:
    uploaded_files = st.file_uploader(
        "Upload one or more PDF research papers",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        with st.spinner("Indexing uploaded PDFs (this may take a bit)..."):
            db = build_vectorstore_from_uploads()
        st.success("✅ Uploaded PDFs indexed successfully!")


# -----------------------------
# Ask question
# -----------------------------
question = st.text_input("Ask a question:")

ask_btn = st.button("🔍 Ask")

if ask_btn:
    if not db:
        st.warning("No documents loaded yet.")
    elif not question.strip():
        st.warning("Please type a question first.")
    else:
        with st.spinner("Searching papers..."):
            retriever = db.as_retriever(search_kwargs={"k": 4})
            docs = retriever.invoke(question)

        with st.spinner("Generating answer..."):
            answer = llm(question, docs)

        st.subheader("✅ Answer")
        st.write(answer)

        st.subheader("📌 Sources")
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "Unknown file")
            page = doc.metadata.get("page", "Unknown page")

            st.markdown(f"**Source {i}:** `{os.path.basename(source)}` — page {page}")
            st.caption(doc.page_content[:400] + "...")
