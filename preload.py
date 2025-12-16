from langchain_huggingface import HuggingFaceEmbeddings

print("⬇️ Downloading AI Model for Cache...")
# This forces the download to happen now, saving it to the disk
HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("✅ Model downloaded successfully!")