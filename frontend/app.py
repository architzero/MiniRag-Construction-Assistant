import html

def sanitize_text(text):
    if not text or not isinstance(text, str):
        return ""
    return html.escape(text)

import streamlit as st
import sys
import os
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
src_dir = os.path.join(parent_dir, 'src')

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import Backend
try:
    from rag_pipeline import RAGPipeline
    from build_index import run_indexing_pipeline
except ImportError:
    st.error("Critical Error: System modules not found. Check 'src' folder.")
    st.stop()

# Import File Parsers
try:
    import pypdf
    from docx import Document
except ImportError:
    st.error("Missing libraries! Please run: pip install pypdf python-docx")
    st.stop()

st.set_page_config(
    page_title="Indecimal AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important;
    }

    .stApp { background-color: #0E1117; color: #E0E0E0; }
    section[data-testid="stSidebar"] { background-color: #262730; }
    
    p, .stMarkdown { 
        font-size: 1.05rem; 
        line-height: 1.7; 
        color: #E0E0E0;
    }

    h1, h2, h3 { color: #FFFFFF !important; font-weight: 700; }
    
    .stChatMessage[data-testid="stChatMessageUser"] {
        background-color: #1E293B; 
        border: 1px solid #334155; 
        border-radius: 12px;
    }

    .stChatMessage[data-testid="stChatMessageAssistant"] {
        background-color: #1F1F1F; 
        border: 1px solid #333; 
        border-radius: 12px;
    }

    .source-box {
        background-color: #2B2D31;
        border-left: 4px solid #4CAF50;
        padding: 12px; 
        margin-top: 8px; 
        border-radius: 4px;
        font-family: 'Consolas', 'Monaco', monospace; 
        font-size: 0.9em;
    }

    .source-header { 
        display: flex; 
        justify-content: space-between; 
        color: #4CAF50; 
        font-weight: bold; 
        margin-bottom: 5px; 
    }

    .source-text { color: #CCCCCC; }

    .info-box {
        background: #333; 
        padding: 10px; 
        border-radius: 5px;
        font-size: 0.85em; 
        color: #ccc; 
        margin-bottom: 10px; 
        border: 1px solid #444;
    }

    .github-btn {
        display: block; 
        width: 100%; 
        padding: 10px; 
        text-align: center;
        background: #24292e; 
        color: white !important; 
        border-radius: 6px; 
        text-decoration: none; 
        border: 1px solid #444; 
        font-weight: 600;
        margin-top: 15px;
    }

    .github-btn:hover { 
        background: #333; 
        border-color: #666; 
    }
</style>
""", unsafe_allow_html=True)
def extract_text_from_file(uploaded_file):
    file_type = uploaded_file.name.split('.')[-1].lower()
    text = ""
    try:
        if file_type == 'pdf':
            pdf_reader = pypdf.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        elif file_type == 'docx':
            doc = Document(uploaded_file)
            for para in doc.paragraphs:
                text += para.text + "\n"
        else:
            text = uploaded_file.read().decode("utf-8")
        return text.strip()
    except Exception as e:
        st.error(f"Error parsing {uploaded_file.name}: {e}")
        return None


def read_core_file(filename):
    try:
        with open(os.path.join("data", filename), "r", encoding="utf-8") as f:
            return f.read()
    except:
        return "Error reading file."

if "rag" not in st.session_state:
    st.session_state.rag = RAGPipeline(index_path="index/assignment")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_mode" not in st.session_state:
    st.session_state.current_mode = "Assignment Mode"

if "api_key" not in st.session_state:
    st.session_state.api_key = st.secrets.get("GROQ_API_KEY", "")

if "model_provider" not in st.session_state:
    st.session_state.model_provider = "Groq"


#sidebar
with st.sidebar:

    st.header(" System Configuration")

    # Data Source
    st.subheader(" Data Source")

    mode = st.radio(
        "Select Mode",
        [" Assignment Mode", "Custom File Mode"],
        label_visibility="collapsed"
    )

    # Mode logic
    if "Assignment" in mode:
        clean_mode = "Assignment Mode"
    else:
        clean_mode = "Custom File Mode"

    if clean_mode != st.session_state.current_mode:
        st.session_state.current_mode = clean_mode
        st.session_state.messages = []

        if clean_mode == "Assignment Mode":
            st.session_state.rag.load_index("index/assignment")
            st.toast("Loaded Core Documents")
        else:
            if os.path.exists("index/custom/vector_store.index"):
                st.session_state.rag.load_index("index/custom")
                st.toast("Loaded Custom Documents")
            else:
                st.warning("No custom index found.")

    st.divider()

    # Upload / Core Docs
    if clean_mode == "Custom File Mode":

        st.subheader(" Upload Documents")

        uploaded_files = st.file_uploader(
            "Upload files",
            type=["pdf", "docx", "md", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

        if uploaded_files and st.button("Index Files", type="primary"):
            with st.spinner("Processing files..."):
                docs = []
                for f in uploaded_files:
                    txt = extract_text_from_file(f)
                    if txt:
                        docs.append({"source": f.name, "text": txt})

                if docs:
                    if not os.path.exists("index/custom"):
                        os.makedirs("index/custom")

                    run_indexing_pipeline(docs, "index/custom")
                    st.session_state.rag.load_index("index/custom")
                    st.success(f"Indexed {len(docs)} files successfully.")

    else:

        st.subheader("📂 Core Documents")
        st.caption("Click to view content:")

        with st.expander("📄 doc1.md (Overview)"):
            st.text(read_core_file("doc1.md"))

        with st.expander("📄 doc2.md (Pricing)"):
            st.text(read_core_file("doc2.md"))

        with st.expander("📄 doc3.md (Policies)"):
            st.text(read_core_file("doc3.md"))

    st.divider()

    # Model Settings
    st.subheader("🤖 AI Model")

    model_provider = st.radio(
        "Choose LLM",
        ["Groq", "Ollama"],
        label_visibility="collapsed"
    )

    st.session_state.model_provider = model_provider

    if model_provider == "Groq":
        api_key = st.text_input("Groq API Key", type="password", value=st.session_state.api_key)
        if api_key:
            st.session_state.api_key = api_key
        st.caption("Get your key from [console.groq.com](https://console.groq.com)")
    else:
        st.caption("Make sure Ollama is running locally")

    col1, col2 = st.columns(2)

    with col1:
        if st.button(" Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    with col2:
        if st.button("➕ New Chat", use_container_width=True):
            st.rerun()

    st.markdown("""
        <a href="https://github.com/architzero/MiniRag-Construction-Assistant" 
           target="_blank" 
           class="github-btn">
            🔗 View GitHub Repository
        </a>
    """, unsafe_allow_html=True)

# Main Interface
st.title("Indecimal AI Assistant")
st.caption(f"📊 Data: {st.session_state.current_mode} | 🤖 Model: {st.session_state.model_provider}")
# Display Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg.get("sources"):
            with st.expander(" Verified Sources"):
                for src in msg["sources"]:
                    st.markdown(f"""
                    <div class="source-box">
                        <div class="source-header">
                            <span>{src['source']}</span>
                            <span>Score: {src['score']:.2f}</span>
                        </div>
                        <div class="source-text">
                           "{sanitize_text(src['text'])}"
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
#chat input
if prompt := st.chat_input("Ask about packages, pricing, or policies..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing documents..."):

            try:
                response = st.session_state.rag.run(
                    prompt,
                    chat_history=st.session_state.messages,
                    model_type=st.session_state.model_provider,
                    api_key=st.session_state.api_key if st.session_state.model_provider == "Groq" else None
                )

                answer = response["answer"]
                sources = response["sources"]

                st.markdown(answer)

                if sources:
                    with st.expander("🔍 Verified Sources"):
                        for src in sources:
                            st.markdown(f"""
                            <div class="source-box">
                                <div class="source-header">
                                    <span>{src['source']}</span>
                                    <span>Score: {src['score']:.2f}</span>
                                </div>
                                <div class="source-text">
                                   "{sanitize_text(src['text'])}"
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:
                st.error(f"Error: {e}")
