🧬 BioRetrieve RAG – Healthcare Information Assistant

BioRetrieve RAG is a Retrieval-Augmented Generation (RAG) based healthcare information assistant built using local LLMs and a vector database. It retrieves trusted healthcare documents and generates evidence-based responses using semantic search.

This project demonstrates the integration of Large Language Models (LLMs), embeddings, and vector databases for domain-specific AI applications in healthcare.

🚀 Features

Retrieval-Augmented Generation (RAG) pipeline

Semantic search using vector embeddings

Healthcare document ingestion (WHO publications)

Evidence-based response generation

Local LLM support via Ollama

Agent-based architecture

Interactive UI using AgentOS

Built-in medical disclaimer for responsible AI usage

🏗 Architecture

The system consists of:

LLM Model – Powered by Ollama (e.g., llama3.2)

Embedding Model – Ollama embedder for semantic search

Vector Database – Qdrant for document indexing

Knowledge Base Layer – Stores and retrieves healthcare documents

Agent Layer – Generates responses with instructions

AgentOS UI – Web interface for interaction

Flow:

User Query → Embedding → Qdrant Search → Retrieved Context → LLM → Response

🛠 Tech Stack

Python

Agno Framework

Ollama (Local LLMs)

Qdrant (Vector Database)

AgentOS

📦 Installation
1️⃣ Install Dependencies
pip install agno qdrant-client ollama

2️⃣ Install and Run Ollama

Download Ollama and pull a model:

ollama pull llama3.2


Make sure Ollama is running locally.

3️⃣ Run Qdrant

Using Docker:

docker run -p 6333:6333 qdrant/qdrant

▶️ Running the Application

Run:

python bio_retrieve_rag.py


The AgentOS interface will launch locally.

📚 Knowledge Source

This project loads healthcare documentation from:

WHO healthcare publications

Documents are embedded and stored inside a Qdrant collection for semantic retrieval.

⚠️ Disclaimer

This assistant provides informational responses based on retrieved healthcare documents. It does not provide medical advice. Always consult qualified healthcare professionals for medical decisions.

🎯 Use Cases

Healthcare knowledge assistant

Medical literature exploration

Research support tool

RAG implementation demo for healthcare AI

📌 Future Improvements

Add multiple healthcare document sources

Support PDF ingestion

Improve retrieval ranking

Add citation formatting in responses

Deploy as cloud application
