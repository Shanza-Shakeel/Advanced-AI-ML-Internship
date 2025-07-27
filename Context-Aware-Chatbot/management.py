import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from app import vectorstore

st.title("Document Management")

# File uploader
uploaded_file = st.file_uploader("Upload document", type=["pdf", "txt"])

if uploaded_file:
    # Save file temporarily
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    if st.button("Process Document"):
        with st.spinner("Processing..."):
            try:
                # Load document
                if uploaded_file.type == "application/pdf":
                    loader = PyPDFLoader(file_path)
                else:
                    loader = TextLoader(file_path)
                
                docs = loader.load()
                
                # Split text
                splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                texts = splitter.split_documents(docs)
                
                # Add to vector store
                vectorstore.add_documents(texts)
                st.success("Document processed!")
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                os.remove(file_path)