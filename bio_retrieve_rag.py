from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.qdrant import Qdrant
from agno.knowledge.embedder.ollama import OllamaEmbedder
from agno.os import AgentOS


# --------------------------------------------------
# Configuration
# --------------------------------------------------

COLLECTION_NAME = "bioretrieve-healthcare-index"
QDRANT_URL = "http://localhost:6333"


# --------------------------------------------------
# Qdrant Vector Database
# --------------------------------------------------

vector_db = Qdrant(
    collection=COLLECTION_NAME,
    url=QDRANT_URL,
    embedder=OllamaEmbedder(
        id="nomic-embed-text",
        dimensions=768
    )
)


# --------------------------------------------------
# Knowledge Base
# --------------------------------------------------

knowledge_base = Knowledge(
    vector_db=vector_db
)


# --------------------------------------------------
# Add WHO Healthcare Source
# --------------------------------------------------
# This should be run once for ingestion.
# After successful ingestion, comment this out.

knowledge_base.add_content(
    url="https://www.who.int/publications/i/item/WHO-UCN-NCD-20.1"
)


# --------------------------------------------------
# Healthcare AI Agent
# --------------------------------------------------

agent = Agent(
    name="BioRetrieve RAG",
    model=Ollama(
        id="llama3.2"
    ),
    knowledge=knowledge_base,
    instructions="""
    You are a healthcare information assistant.

    - Provide evidence-based answers using the available knowledge.
    - Explain information clearly and simply.
    - Do not make unsupported medical claims.
    - If the available information is insufficient, say so.
    - Always include a disclaimer that this is not medical advice.
    """
)


# --------------------------------------------------
# AgentOS
# --------------------------------------------------

agent_os = AgentOS(
    agents=[agent]
)

app = agent_os.get_app()


# --------------------------------------------------
# Run Application
# --------------------------------------------------

if __name__ == "__main__":
    agent_os.serve(
        app="bio_retrieve_rag:app",
        reload=True
    )