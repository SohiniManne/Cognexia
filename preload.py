from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

print("⬇️ Downloading FastEmbed Model...")
# This forces the download of the lightweight model
FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
print("✅ FastEmbed Model Ready!")
