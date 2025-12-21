import os
import shutil
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from pinecone import Pinecone

# Imports
from langchain_groq import ChatGroq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
app = FastAPI(title="Cognexia RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Components
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

vectorstore = PineconeVectorStore(
    index_name=os.getenv("PINECONE_INDEX_NAME"),
    embedding=embeddings
)

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

template = """
You are Cognexia. Answer based ONLY on the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def home():
    return {"status": "Cognexia System Online"}

@app.post("/chat")
async def chat(request: QueryRequest):
    # 1. Retrieve Docs
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(request.query)
    
    # 2. Format Context
    context_text = "\n\n".join([d.page_content for d in docs])
    
    # 3. Generate Answer
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context_text, "question": request.query})
    
    # 4. Extract Unique Citations (The Upgrade)
    # PyMuPDF pages are 0-indexed, so we add +1 to make it human-readable
    sources = sorted(list(set([f"Page {doc.metadata.get('page', 0) + 1}" for doc in docs])))
    
    return {"answer": answer, "sources": sources}

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    try:
        # Step 1: Wipe Memory (Safe)
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
        try:
            index.delete(delete_all=True)
        except Exception:
            pass

        # Step 2: Process File (Batch Mode)
        temp_filename = f"temp_{file.filename}"
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        loader = PyMuPDFLoader(temp_filename)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        
        batch = []
        total_chunks = 0
        
        # --- SMART BATCHING ---
        # We read the file ONE page at a time
        for doc in loader.lazy_load():
            # Convert just this 1 page into chunks
            page_chunks = text_splitter.split_documents([doc])
            batch.extend(page_chunks)
            
            # Whenever we have 50 chunks, send them to Pinecone and CLEAR RAM
            if len(batch) >= 50:
                vectorstore.add_documents(batch)
                total_chunks += len(batch)
                batch = []  # <--- This prevents the crash!
                
        # Send any remaining chunks at the end
        if batch:
            vectorstore.add_documents(batch)
            total_chunks += len(batch)
        # ----------------------

        os.remove(temp_filename)
        
        return {"status": "success", "chunks": total_chunks}
        
    except Exception as e:
        print(f"Error ingesting: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ingest failed: {str(e)}")
