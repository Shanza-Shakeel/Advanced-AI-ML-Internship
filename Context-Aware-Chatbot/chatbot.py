import streamlit as st
from app1 import rag_query, process_feedback

st.title("RAG Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            col1, col2 = st.columns([1,15])
            with col1:
                if st.button("👍", key=f"thumbs_up_{i}"):
                    process_feedback(
                        st.session_state.messages[i-1]["content"],
                        message["content"],
                        True
                    )
            with col2:
                if st.button("👎", key=f"thumbs_down_{i}"):
                    process_feedback(
                        st.session_state.messages[i-1]["content"],
                        message["content"],
                        False
                    )

# Handle user input
if prompt := st.chat_input("Ask me anything"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    response = rag_query(prompt)
    
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    st.rerun()

# Sidebar controls
with st.sidebar:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()