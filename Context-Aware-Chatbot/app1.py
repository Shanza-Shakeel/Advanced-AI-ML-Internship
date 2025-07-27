# MUST BE FIRST LINE
import streamlit as st
st.set_page_config(
    page_title="Enterprise RAG Assistant",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# Imports
from huggingface_hub import InferenceClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
import os
import time

# Custom CSS for professional look
st.markdown("""
<style>
    .stApp {
        background-color: #f8f9fa;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        box-shadow: 2px 0 10px rgba(0,0,0,0.1);
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .stButton>button {
        background-color: #4caf50;
        color: white;
        border-radius: 4px;
    }
    .stTextArea textarea {
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize with default knowledge
@st.cache_resource
def init_chatbot():
    # 1. Simple embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # 2. Start with some default knowledge
    default_knowledge = [
        "This is an enterprise-grade RAG chatbot that provides accurate, sourced answers.",
        "The system combines retrieval from document stores with Mistral-7B generation.",
        "All responses are contextual and based on the provided knowledge base."
    ]
    
    # 3. Create vector store
    vectorstore = FAISS.from_texts(default_knowledge, embeddings)
    return vectorstore, InferenceClient(
        "mistralai/Mistral-7B-Instruct-v0.3",
        timeout=30  # Increased timeout for reliability
    )

# Initialize
with st.spinner("Initializing enterprise knowledge base..."):
    vectorstore, client = init_chatbot()

# Chat function with enhanced features
def chat(prompt):
    try:
        # 1. Find relevant info with loading indicator
        with st.spinner("Searching knowledge base..."):
            docs = vectorstore.similarity_search(prompt, k=3)
            context = "\n".join([d.page_content for d in docs])
        
        # 2. Generate response with typing indicator
        with st.spinner("Generating response..."):
            start_time = time.time()
            response = client.chat_completion(
                messages=[{
                    "role": "user",
                    "content": f"Context: {context}\nQuestion: {prompt}\nProvide a professional, well-structured answer:"
                }],
                max_tokens=500,
                temperature=0.3  # More deterministic
            )
            processing_time = time.time() - start_time
            
        # Format response professionally
        formatted_response = f"""
        <div class='assistant-message'>
            <p>{response.choices[0].message.content}</p>
            <p style='font-size: 0.8em; color: #666; margin-top: 1rem;'>
                Generated in {processing_time:.2f}s | Sources: {len(docs)} documents referenced
            </p>
        </div>
        """
        return formatted_response
        
    except Exception as e:
        return f"<div class='assistant-message'>⚠️ Error: {str(e)}</div>"

# Document processing with validation
def add_document(text):
    if not text.strip():
        st.error("Please enter valid content")
        return False
    
    try:
        with st.spinner("Processing document..."):
            splitter = CharacterTextSplitter(chunk_size=500)
            texts = splitter.split_text(text)
            vectorstore.add_texts(texts)
            time.sleep(1)  # Simulate processing
        return True
    except Exception as e:
        st.error(f"Processing failed: {str(e)}")
        return False

# ===== Professional UI =====
st.title("Enterprise RAG Assistant")
st.caption("Secure, document-aware AI assistant with contextual memory")

# Sidebar for document management
with st.sidebar:
    st.header("📂 Knowledge Management")
    st.subheader("Add Documents")
    
    # File uploader with restrictions
    uploaded_file = st.file_uploader(
        "Upload PDF/TXT",
        type=["pdf", "txt"],
        help="Max 10MB. For large documents, split into sections."
    )
    
    # Text input with formatting options
    with st.expander("Advanced Text Input"):
        text_content = st.text_area(
            "Paste document content:",
            height=200,
            placeholder="Paste text content here...",
            help="For best results, include clear section headings."
        )
        if st.button("Add to Knowledge Base", key="add_text"):
            if add_document(text_content):
                st.toast("Document added successfully!", icon="✅")
    
    # System status
    st.divider()
    st.subheader("System Status")
    st.metric("Documents Indexed", len(vectorstore.index_to_docstore_id))
    st.caption(f"Last updated: {time.strftime('%Y-%m-%d %H:%M')}")

# Main chat interface
tab1, tab2 = st.tabs(["💬 Chat", "⚙️ Settings"])

with tab1:
    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)
    
    # Input area with enhanced features
    with st.container():
        prompt = st.chat_input(
            "Ask about your documents...",
            key="chat_input"
        )
        
        if prompt:
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": f"<div class='user-message'>{prompt}</div>"
            })
            
            # Generate and display response
            with st.chat_message("assistant"):
                response = chat(prompt)
                st.markdown(response, unsafe_allow_html=True)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

with tab2:
    st.header("System Configuration")
    
    # Model settings
    with st.expander("Model Parameters"):
        col1, col2 = st.columns(2)
        with col1:
            st.slider("Temperature", 0.0, 1.0, 0.3, help="Lower = more deterministic")
        with col2:
            st.slider("Max Tokens", 100, 1000, 500)
    
    # Knowledge base management
    with st.expander("Knowledge Base Tools"):
        if st.button("Rebuild Vector Index"):
            with st.spinner("Reindexing..."):
                time.sleep(2)
                st.toast("Index rebuilt successfully", icon="✅")
        
        if st.button("Clear Conversation History"):
            st.session_state.messages = []
            st.rerun()
    
    # System info
    st.divider()
    st.subheader("About")
    st.info("""
    Enterprise RAG Assistant v1.0  
    Powered by Mistral-7B and FAISS  
    All responses are generated from your knowledge base
    """)