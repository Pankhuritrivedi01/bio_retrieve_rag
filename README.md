# 🧬 BioRetrieve RAG — Healthcare Information Assistant

> A document-grounded AI assistant that uses **Retrieval-Augmented Generation (RAG)** to answer healthcare questions using curated **WHO resources**.

## 🎯 What It Does

BioRetrieve lets users ask healthcare questions in natural language and retrieves relevant information from indexed healthcare documents before generating a response.

**Question → Embedding → Qdrant Search → Relevant Context → Llama 3.2 → Answer**

## 📊 Current Setup

| Component | Implementation |
|---|---|
| Healthcare documents | 3 WHO-based PDFs |
| Vector database | Qdrant |
| Embeddings | Nomic Embed Text |
| Embedding dimension | 768 |
| LLM | Llama 3.2 |
| LLM runtime | Ollama |
| RAG framework | Agno |
| Interface | Streamlit |
| PDF processing | PyPDF |

## 🏗️ Architecture

```text
User Question
      ↓
Streamlit Dashboard
      ↓
Nomic Embed Text
      ↓
Qdrant Vector Search
      ↓
Relevant Healthcare Context
      ↓
Llama 3.2 via Ollama
      ↓
Grounded Response
💡 Why It Is Useful
🔎 Quickly searches lengthy healthcare documents
📚 Makes trusted document collections easier to explore
🧠 Demonstrates practical RAG + LLM implementation
🏥 Can be extended with additional healthcare resources
💻 Provides an interactive web interface
📚 Knowledge Base

The current knowledge base contains WHO resources covering:

Cardiovascular disease
Diabetes
Noncommunicable diseases
🛠️ Tech Stack

Python · Streamlit · Agno · Qdrant · Ollama · Llama 3.2 · Nomic Embed Text · PyPDF

🚀 Run Locally
git clone https://github.com/Pankhutrivedi01/bio_retrieve_rag.git
cd bio_retrieve_rag

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt

Start Qdrant and Ollama, then ensure these models are available:

llama3.2
nomic-embed-text

Run the dashboard:

streamlit run ui.py

Open:

http://localhost:8501
📁 Project Structure
bio_retrieve_rag/
├── docs/                  # WHO healthcare documents
├── ui.py                  # Streamlit dashboard
├── bio_retrieve_rag.py    # RAG/AgentOS application
├── requirements.txt
├── .gitignore
└── README.md
🔮 Future Improvements
Source citations for retrieved information
Retrieval and response evaluation
Larger verified healthcare knowledge base
Document upload functionality
Cloud deployment
Conversation history
⚠️ Disclaimer

This is an educational AI/RAG project, not a medical diagnostic or treatment system. Responses should not be used for clinical decision-making.

👩‍💻 Author

Pankhuri Trivedi
B.Tech Bioinformatics | Data Science & AI
