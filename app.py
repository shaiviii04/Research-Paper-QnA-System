import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from fake_llm import fake_llm

VECTORSTORE_PATH = "vectorstore"

st.set_page_config(page_title="ML Research Q&A", layout="centered")

st.title("📄 Machine Learning Research Q&A")
st.write("Ask questions based on the preloaded machine learning research papers.")

# Load Vector Store
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    db = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return db

db = load_vectorstore()

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# User Question
question = st.text_input("What do you want to ask? :)")

if question:
    with st.spinner("Searching papers..."):
         docs = db.similarity_search(question, k=4)  # get top 4 chunks
         answer = fake_llm(question, docs)
         st.subheader("📄 Answer")
         st.text(answer)

