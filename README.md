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

