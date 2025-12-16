# 🧠 Cognexia: Dynamic Research Assistant (RAG)

**A Full-Stack AI Application for "Chatting" with Research Papers.** *Built for the Spazorlabs AI/ML Internship Assessment (Task 1).*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cognexia.streamlit.app)
[![Deployed on Render](https://img.shields.io/badge/Backend-Render-46E3B7)](https://cognexia-backend.onrender.com/docs)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)

---

## 📖 Overview

**Cognexia** is an end-to-end **Retrieval-Augmented Generation (RAG)** system that allows users to upload complex PDF documents (such as research papers) and ask questions about them. The system uses a decoupled client-server architecture to ensure scalability and separation of concerns.

> *"Build an end-to-end RAG system with document ingestion, chunking, embeddings, vector database search, and LLM-based answer generation **with citations**."*

### 🚀 Key Features
* **📄 PDF Ingestion:** Parses complex PDF structures using `PyMuPDF` to extract text while preserving page metadata.
* **🧩 Intelligent Chunking:** Uses Recursive Character Splitting to create context-aware text chunks.
* **⚡ High-Performance Embeddings:** Optimized for memory-constrained environments using **FastEmbed** (Quantized ONNX models), reducing RAM usage by ~60% compared to standard PyTorch transformers.
* **🔍 Vector Search:** Serverless vector storage and retrieval using **Pinecone** for low-latency queries (<200ms).
* **🤖 LLM Reasoning:** Powered by **Llama-3-8b-Instant** (via Groq) for rapid, accurate, and context-aware responses.
* **📍 Accurate Citations:** Every answer includes precise **Page Number references** so users can verify the source.
* **☁️ Cloud Native:** Fully deployed with a **FastAPI** backend on Render and a **Streamlit** frontend on Streamlit Cloud.

---

## 🛠️ Tech Stack

| Component | Technology | Reasoning |
| :--- | :--- | :--- |
| **Frontend** | Streamlit (Python) | Rapid UI development and interactive chat interface. |
| **Backend** | FastAPI | High-performance, asynchronous REST API capabilities. |
| **LLM** | Llama-3-8b (Groq) | State-of-the-art open-source model with extremely fast inference speed. |
| **Embeddings** | FastEmbed (`bge-small-en`) | Lightweight ONNX runtime (No-GPU required), perfect for free-tier deployments. |
| **Vector DB** | Pinecone | Managed serverless vector database for scalable similarity search. |
| **PDF Parser** | PyMuPDF (LangChain) | fast and accurate text extraction with metadata support. |
| **Hosting** | Render + Streamlit Cloud | Decoupled hosting to scale logic separate from UI. |

---

## 🏗️ Architecture

1.  **Ingestion:** User uploads a PDF -> Backend chunks text -> Generates Embeddings -> Stores in Pinecone.
2.  **Retrieval:** User asks a question -> Query is embedded -> Pinecone finds top 5 relevant chunks.
3.  **Generation:** Retrieved chunks + Question are sent to Llama-3 -> LLM generates an answer with citations.

---

## 💻 Local Setup

If you want to run this locally, follow these steps:

### 1. Clone the Repository
```bash
git clone [https://github.com/SohiniManne/Cognexia.git](https://github.com/SohiniManne/Cognexia.git)
cd Cognexia
2. Set Environment Variables
Create a .env file in the root directory and add your API keys:

Code snippet

GROQ_API_KEY=gsk_...
PINECONE_API_KEY=pc-...
PINECONE_INDEX_NAME=cognexia-index
3. Install Dependencies
Backend:

Bash

pip install -r backend.txt
Frontend:

Bash

pip install -r requirements.txt
4. Run the Application
You need two terminals running simultaneously:

Terminal 1 (Backend):

Bash

uvicorn main:app --reload
Terminal 2 (Frontend):

Bash

streamlit run app.py
🌐 API Documentation
The backend exposes the following endpoints (Auto-generated docs available at /docs):

GET /: Health check.

POST /ingest: Uploads a PDF, processes it, and indexes vectors into Pinecone.

Input: file (UploadFile)

Output: JSON with chunk count.

POST /chat: Asks a question to the RAG pipeline.

Input: query (string)

Output: JSON with answer and sources list.

🚧 Challenges & Optimizations
Memory Constraints: Initial deployment on Render (Free Tier 512MB RAM) failed with OOM (Out of Memory) errors due to heavy PyTorch dependencies.

Solution: Migrated to FastEmbed, which uses quantized models and the ONNX runtime, significantly lowering the memory footprint.

Cold Starts: Serverless backends sleep after inactivity.

Solution: Implemented a health-check endpoint and error handling in the frontend to alert users if the backend is waking up.

📄 License
This project is open-source and available under the MIT License.

Author: Sohini Manne
