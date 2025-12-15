import streamlit as st
import requests

st.set_page_config(page_title="Cognexia AI", page_icon="🧠")
st.title("🧠 Cognexia: Dynamic Research Assistant")

# --- 1. SAFE URL HANDLING ---
# This function prevents the crash by handling the missing file error
def get_api_url():
    try:
        # Try to access secrets (works on Cloud)
        if "API_URL" in st.secrets:
            return st.secrets["API_URL"]
    except FileNotFoundError:
        # If file missing (Local), return localhost
        return "http://127.0.0.1:8000"
    except Exception:
        return "http://127.0.0.1:8000"
    return "http://127.0.0.1:8000"

API_BASE_URL = get_api_url()

# --- 2. SIDEBAR (Uploads) ---
with st.sidebar:
    st.header("📂 Knowledge Base")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Ingest Document"):
            with st.spinner("Processing..."):
                files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                try:
                    # Use the safe URL
                    response = requests.post(f"{API_BASE_URL}/ingest", files=files)
                    if response.status_code == 200:
                        st.success(f"✅ Indexed {response.json()['chunks']} chunks!")
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Failed to connect to Backend: {e}")

# --- 3. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your PDF..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("Thinking...")
        try:
            # Use the safe URL
            response = requests.post(f"{API_BASE_URL}/chat", json={"query": prompt})
            
            if response.status_code == 200:
                ans = response.json()["answer"]
                placeholder.markdown(ans)
                st.session_state.messages.append({"role": "assistant", "content": ans})
            else:
                placeholder.error(f"Error {response.status_code}")
        except Exception as e:
            placeholder.error(f"Backend is offline. Is uvicorn running?\n\n{e}")