# ===============================
# BioRetrieve RAG – Healthcare
# ===============================

from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.qdrant import Qdrant
from agno.knowledge.embedder.ollama import OllamaEmbedder
from agno.os import AgentOS

# -------------------------------
# Configuration
# -------------------------------
COLLECTION_NAME = "bioretrieve-healthcare-index"
QDRANT_URL = "http://localhost:6333"

# -------------------------------
# Vector Database
# -------------------------------
vector_db = Qdrant(
    collection=COLLECTION_NAME,
    url=QDRANT_URL,
    embedder=OllamaEmbedder()
)

# -------------------------------
# Knowledge Base (Healthcare Docs)
# -------------------------------
knowledge_base = Knowledge(vector_db=vector_db)

# Load medical content (run once)
knowledge_base.add_content(
    url="https://www.who.int/publications/i/item/WHO-UCN-NCD-20.1"
)

# -------------------------------
# Agent
# -------------------------------
agent = Agent(
    name="BioRetrieve RAG",
    model=Ollama(id="llama3.2"),  # use a model you pulled
    knowledge=knowledge_base,
    instructions="""
    You are a healthcare information assistant.
    - Provide evidence-based answers
    - Be clear and cautious
    - Add a disclaimer that this is not medical advice
    """
)


# -------------------------------
# AgentOS UI
# -------------------------------
agent_os = AgentOS(agents=[agent])
app = agent_os.get_app()

# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    agent_os.serve(app="bio_retrieve_rag:app", reload=True)
