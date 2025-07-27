\# Context-Aware RAG Chatbot



\## 📌 Objective

Build a conversational AI assistant that:

\- Retrieves information from custom knowledge bases

\- Maintains context across conversations

\- Generates accurate, document-backed responses using RAG architecture



\## 🛠️ Methodology \& Approach



\### Technical Stack

| Component           | Technology Used            |

|---------------------|---------------------------|

| Framework           | LangChain                  |

| Vector Store        | FAISS                      |

| Embeddings          | Sentence-Transformers      |

| LLM                 | Mistral-7B-Instruct        |

| Deployment          | Streamlit                  |



\### Key Features

1\. \*\*Document Processing Pipeline\*\*:

&nbsp;  ```mermaid

&nbsp;  graph TD

&nbsp;    A\[PDF/TXT Upload] --> B(Text Extraction)

&nbsp;    B --> C\[Chunking]

&nbsp;    C --> D\[Vector Embedding]

&nbsp;    D --> E\[FAISS Indexing]

