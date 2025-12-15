import os
import shutil
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from pinecone import Pinecone

# --- NEW IMPORTS ---
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyMuPDFLoader # <--- THE FIX
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
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

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

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def home():
    return {"status": "Cognexia System Online"}

@app.post("/chat")
async def chat(request: QueryRequest):
    # Search for top 10 chunks (increased from 5 to catch Abstracts better)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    response = rag_chain.invoke(request.query)
    return {"answer": response}

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    try:
        # WIPE MEMORY
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
        try:
            index.delete(delete_all=True)
        except Exception:
            pass

        # PROCESS FILE
        temp_filename = f"temp_{file.filename}"
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # --- THE FIX: Use PyMuPDFLoader ---
        # This reads columns correctly
        loader = PyMuPDFLoader(temp_filename) 
        raw_docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(raw_docs)
        
        vectorstore.add_documents(documents)
        os.remove(temp_filename)
        
        return {"status": "success", "chunks": len(documents)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))