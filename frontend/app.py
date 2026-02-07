"""
Mini RAG Chatbot - Streamlit Interface
A beautiful, professional chatbot for construction marketplace queries
"""
import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embeddings import EmbeddingGenerator
from vector_store import VectorStore
from rag_pipeline import RAGPipeline

# Page configuration
st.set_page_config(
    page_title="Construction Assistant",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a clean, professional look - Construction themed
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #fafafa;
    }
    
    /* Header styling - Construction orange/slate theme */
    .header-container {
        background: linear-gradient(120deg, #2c3e50 0%, #34495e 50%, #e67e22 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        opacity: 0.95;
        font-weight: 300;
    }
    
    /* Chat messages */
    .user-message {
        background-color: #2c3e50;
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.5rem 0;
        max-width: 75%;
        margin-left: auto;
        box-shadow: 0 2px 4px rgba(44, 62, 80, 0.2);
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    .assistant-message {
        background-color: white;
        color: #2c3e50;
        padding: 1rem 1.5rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.5rem 0;
        max-width: 75%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        border-left: 4px solid #e67e22;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Context boxes */
    .context-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #e67e22;
        font-size: 0.85rem;
        line-height: 1.5;
    }
    
    .source-badge {
        background-color: #e67e22;
        color: white;
        padding: 0.25rem 0.7rem;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-right: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #e67e22;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s;
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
    }
    
    .stButton>button:hover {
        background-color: #d35400;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(230, 126, 34, 0.3);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Input box */
    .stTextInput>div>div>input {
        border-radius: 24px;
        border: 2px solid #dee2e6;
        padding: 0.8rem 1.2rem;
        font-size: 0.95rem;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #e67e22;
        box-shadow: 0 0 0 0.2rem rgba(230, 126, 34, 0.15);
    }
    
    /* Success/Info boxes */
    .stSuccess {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    
    .stInfo {
        background-color: #e8f4f8;
        border-left: 4px solid #2c3e50;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 6px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
    st.session_state.initialized = False

def initialize_rag():
    """Initialize RAG pipeline (cached)"""
    if not st.session_state.initialized:
        with st.spinner("🔧 Initializing AI Assistant... This may take a moment on first load."):
            try:
                # Load embedding model
                embedder = EmbeddingGenerator()
                
                # Load vector store
                vector_store = VectorStore(embedding_dim=embedder.embedding_dim)
                vector_store.load()
                
                # Initialize RAG pipeline
                rag = RAGPipeline(
                    vector_store=vector_store,
                    embedder=embedder,
                    llm_backend="ollama",
                    llm_model="llama3.2:3b",
                    top_k=3,
                    temperature=0.1
                )
                
                st.session_state.rag_pipeline = rag
                st.session_state.initialized = True
                return True
            except Exception as e:
                st.error(f"❌ Error initializing system: {str(e)}")
                st.error("Make sure you've run: `python src/vector_store.py` first!")
                return False
    return True

def display_message(role, content, context=None):
    """Display a chat message with styling"""
    if role == "user":
        st.markdown(f'<div class="user-message">👤 {content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-message">🤖 {content}</div>', unsafe_allow_html=True)
        
        # Display context if available
        if context and st.session_state.show_context:
            with st.expander("📚 View Retrieved Context", expanded=False):
                for i, ctx in enumerate(context, 1):
                    similarity_pct = ctx['similarity'] * 100
                    st.markdown(f"""
                    <div class="context-box">
                        <span class="source-badge">{ctx['source']}</span>
                        <span style="color: #e67e22; font-weight: 600;">Relevance: {similarity_pct:.1f}%</span>
                        <p style="margin-top: 0.5rem; color: #495057;">{ctx['text'][:300]}...</p>
                    </div>
                    """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="header-container">
        <div class="header-title">🏗️ Construction Assistant</div>
        <div class="header-subtitle">Ask me anything about Indecimal's construction services, quality checks, and policies</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        st.session_state.show_context = st.checkbox(
            "Show Retrieved Context",
            value=True,
            help="Display the source documents used to generate answers"
        )
        
        st.markdown("---")
        
        st.markdown("### 📊 System Info")
        if st.session_state.initialized:
            st.success("✅ AI Assistant Ready")
            st.info("🤖 Model: Llama 3.2 (3B)")
            st.info("🔍 Retrieval: Top-3 Chunks")
        else:
            st.warning("⏳ Not initialized")
        
        st.markdown("---")
        
        st.markdown("### 💡 Example Questions")
        st.markdown("""
        - What quality checks do you perform?
        - How does the payment system work?
        - What services does Indecimal provide?
        - Tell me about stage-based payments
        - What guarantees do you offer?
        """)
        
        st.markdown("---")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This AI assistant uses **Retrieval-Augmented Generation (RAG)** 
        to provide accurate, grounded answers based on Indecimal's 
        internal documents.
        
        **Tech Stack:**
        - 🔢 Embeddings: all-MiniLM-L6-v2
        - 🗄️ Vector Store: FAISS
        - 🤖 LLM: Llama 3.2 (3B)
        - 🎨 Frontend: Streamlit
        """)
    
    # Initialize RAG
    if not initialize_rag():
        st.stop()
    
    # Display chat history
    for message in st.session_state.messages:
        display_message(
            message["role"],
            message["content"],
            message.get("context")
        )
    
    # Chat input
    user_input = st.chat_input("Ask me anything about construction services...")
    
    if user_input:
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Display user message
        display_message("user", user_input)
        
        # Generate response
        with st.spinner("🤔 Thinking..."):
            try:
                response = st.session_state.rag_pipeline.query(user_input)
                
                # Add assistant message to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response['answer'],
                    "context": response.get('retrieved_context', [])
                })
                
                # Display assistant message
                display_message(
                    "assistant",
                    response['answer'],
                    response.get('retrieved_context', [])
                )
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
                display_message("assistant", error_msg)
        
        st.rerun()

if __name__ == "__main__":
    main()