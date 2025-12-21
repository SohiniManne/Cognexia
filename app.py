import streamlit as st
import requests

st.set_page_config(page_title="Cognexia AI", page_icon="🧠")
st.title("🧠 Cognexia: Dynamic Research Assistant")

def get_api_url():
    try:
        if "API_URL" in st.secrets:
            return st.secrets["API_URL"]
    except FileNotFoundError:
        return "http://127.0.0.1:8000"
    except Exception:
        return "http://127.0.0.1:8000"
    return "http://127.0.0.1:8000"

API_BASE_URL = get_api_url()

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 Knowledge Base")
    
    # 1. Add a meaningful Help Tooltip to the uploader
    uploaded_file = st.file_uploader(
        "Upload a PDF", 
        type="pdf", 
        help="The system will process as much of the file as possible within the Free Tier time limit (approx. 40s)."
    )
    
    # 2. Add a Caption to explain the "Smart Timer" logic explicitly
    st.caption("🟢 **Free Tier Mode:** Processing auto-stops after 40s to prevent crashing. Large files will be indexed partially.")
    
    if uploaded_file is not None:
        if st.button("Ingest Document"):
            with st.spinner("Processing (Smart Time-Limit Active)..."):
                files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                try:
                    response = requests.post(f"{API_BASE_URL}/ingest", files=files)
                    if response.status_code == 200:
                        data = response.json()
                        # Show the user exactly what happened
                        st.success(f"✅ {data.get('note', 'Indexing Complete!')}")
                        st.info(f"📄 Processed {data.get('pages_processed', '?')} pages successfully.")
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Failed to connect: {e}")

# --- CHAT ---
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
            response = requests.post(f"{API_BASE_URL}/chat", json={"query": prompt})
            
            if response.status_code == 200:
                data = response.json()
                ans = data["answer"]
                sources = data.get("sources", [])
                
                # Create the final answer with citations
                if sources:
                    full_response = f"{ans}\n\n**Sources:** {', '.join(sources)}"
                else:
                    full_response = ans
                
                placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            else:
                placeholder.error(f"Error {response.status_code}")
        except Exception as e:
            placeholder.error(f"Backend Error: {e}")
