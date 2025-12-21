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

        # Step 2: Process File
        temp_filename = f"temp_{file.filename}"
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        loader = PyMuPDFLoader(temp_filename)
        raw_docs = loader.load()

        # --- THE SAFETY FIX ---
        # Only take the first 5 pages. This prevents the Free Server from crashing.
        if len(raw_docs) > 5:
            raw_docs = raw_docs[:5]
        # ----------------------
        
        # Use a larger chunk size to reduce the number of vectors
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        documents = text_splitter.split_documents(raw_docs)
        
        vectorstore.add_documents(documents)
        os.remove(temp_filename)
        
        return {"status": "success", "chunks": len(documents)}
        
    except Exception as e:
        # This print will show up in your Render logs if it fails
        print(f"Error ingesting: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ingest failed: {str(e)}")
