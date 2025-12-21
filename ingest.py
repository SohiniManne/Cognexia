import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# 1. Load Secrets
load_dotenv()

def ingest():
    # Check if data.pdf exists
    if not os.path.exists("data.pdf"):
        print("❌ Error: 'data.pdf' not found in this folder.")
        return

    print("🚀 [Cognexia] Starting ingestion...")

    # 2. Load PDF
    loader = PyPDFLoader("data.pdf")
    raw_docs = loader.load()
    print(f"   - Loaded {len(raw_docs)} pages")

    # 3. Split Text
   text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    documents = text_splitter.split_documents(raw_docs)
    print(f"   - Split into {len(documents)} chunks")

    # 4. Embed & Store (Using Free HuggingFace Model)
    print("   - Downloading embedding model (This happens once)...")
    # This runs locally on your machine
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("   - Uploading vectors to Pinecone...")
    PineconeVectorStore.from_documents(
        documents, 
        embeddings, 
        index_name=os.getenv("PINECONE_INDEX_NAME")
    )
    
    print("✅ [Cognexia] Knowledge Base Updated Successfully!")

if __name__ == "__main__":
    ingest()
