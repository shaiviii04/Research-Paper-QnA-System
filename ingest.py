from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Paths
DATA_PATH = "data/papers"       
VECTORSTORE_PATH = "vectorstore"  

# 1️⃣ Load PDFs
print("Loading PDFs...")
loader = DirectoryLoader(
    DATA_PATH,
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
    show_progress=True,
    silent_errors=True   
)
documents = loader.load()
print(f"Loaded {len(documents)} documents")

# 2️⃣ Split text into chunks
print("Splitting documents...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# 3️⃣ Create embeddings
print("Creating embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4️⃣ Build FAISS vector store
print("Building FAISS vector store...")
db = FAISS.from_documents(chunks, embeddings)

# 5️⃣ Save vector store locally
if not os.path.exists(VECTORSTORE_PATH):
    os.makedirs(VECTORSTORE_PATH)

db.save_local(VECTORSTORE_PATH)
print("✅ Vector store saved successfully!")
