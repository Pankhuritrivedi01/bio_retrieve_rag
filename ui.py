from pathlib import Path

import streamlit as st

from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.qdrant import Qdrant
from agno.knowledge.embedder.ollama import OllamaEmbedder


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "bioretrieve-healthcare-multidoc"

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"

TOP_K = 3


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="BioRetrieve",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROFESSIONAL UI
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #f7f9fc;
    }

    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
    }

    .hero {
        background: white;
        padding: 2rem;
        border-radius: 18px;
        border: 1px solid #e6eaf0;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        margin-bottom: 0.3rem;
        font-size: 2.4rem;
    }

    .hero p {
        color: #64748b;
        font-size: 1.05rem;
    }

    .answer-card {
        background: white;
        padding: 1.6rem;
        border-radius: 16px;
        border: 1px solid #e6eaf0;
        margin-top: 1.2rem;
    }

    .source-card {
        background: #f8fafc;
        padding: 0.9rem 1rem;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        margin-bottom: 0.6rem;
    }

    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #e6eaf0;
        text-align: center;
    }

    .small-text {
        color: #64748b;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🧬 BioRetrieve</h1>
        <p>
        WHO-based Healthcare Information Assistant
        </p>
        <div class="small-text">
        Multi-document Retrieval-Augmented Generation •
        Evidence-grounded responses • Not medical advice
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# RAG INITIALIZATION
# ============================================================

@st.cache_resource
def initialize_rag():

    pdf_files = [
        pdf for pdf in sorted(DOCS_DIR.glob("*.pdf"))
        if pdf.stat().st_size > 0
    ]

    if not pdf_files:
        raise RuntimeError(
            "No valid PDF documents were found in the docs folder."
        )

    vector_db = Qdrant(
        collection=COLLECTION_NAME,
        url=QDRANT_URL,
        embedder=OllamaEmbedder(
            id=EMBEDDING_MODEL,
            dimensions=768,
        ),
    )

    knowledge_base = Knowledge(
        vector_db=vector_db,
        max_results=TOP_K,
    )

    # Index documents only when they are not already present.
    for pdf in pdf_files:
        knowledge_base.insert(
            path=str(pdf),
            skip_if_exists=True,
        )

    agent = Agent(
        name="BioRetrieve",
        model=Ollama(
            id=LLM_MODEL,
            options={
                "num_predict": 220,
                "temperature": 0.1,
            },
        ),
        markdown=True,
    )

    return knowledge_base, agent, pdf_files


# ============================================================
# LOAD
# ============================================================

try:

    with st.spinner("Initializing healthcare knowledge base..."):
        knowledge_base, agent, pdf_files = initialize_rag()

except Exception as e:

    st.error("BioRetrieve could not start.")

    st.code(str(e))

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🧬 BioRetrieve")

    st.markdown(
        "### Knowledge Base"
    )

    st.success("● System online")

    st.metric(
        "WHO Documents",
        len(pdf_files),
    )

    st.divider()

    st.markdown("### Sources")

    for pdf in pdf_files:

        size_mb = pdf.stat().st_size / (1024 * 1024)

        st.write(
            f"📄 **{pdf.stem}**"
        )

        st.caption(
            f"{size_mb:.2f} MB"
        )

    st.divider()

    st.caption(
        "BioRetrieve provides educational information "
        "from the indexed WHO documents and is not a "
        "substitute for professional medical advice."
    )


# ============================================================
# MAIN QUESTION AREA
# ============================================================

st.markdown("### Ask a healthcare question")

query = st.text_area(
    label="Healthcare question",
    placeholder=(
        "Example: What are the major risk factors "
        "for cardiovascular disease?"
    ),
    height=100,
    label_visibility="collapsed",
)


ask = st.button(
    "🔎 Ask BioRetrieve",
    type="primary",
    use_container_width=True,
)


# ============================================================
# RAG
# ============================================================

if ask and query.strip():

    query = query.strip()

    with st.spinner("Searching WHO knowledge base..."):

        try:

            # ------------------------------------------------
            # RETRIEVAL
            # ------------------------------------------------

            results = knowledge_base.search(
                query=query,
                max_results=TOP_K,
            )

            if not results:

                st.warning(
                    "No relevant information was found "
                    "in the indexed WHO documents."
                )

                st.stop()

            # ------------------------------------------------
            # CLEAN CONTEXT
            # ------------------------------------------------

            context_blocks = []
            sources = []

            for result in results:

                content = getattr(
                    result,
                    "content",
                    "",
                )

                if not content:
                    continue

                content = content.strip()

                # Prevent huge PDF tables/chunks from
                # overwhelming the LLM.
                if len(content) > 3500:
                    content = content[:3500]

                metadata = getattr(
                    result,
                    "meta_data",
                    {}
                ) or {}

                source = (
                    metadata.get("file_name")
                    or metadata.get("name")
                    or getattr(result, "name", None)
                    or "WHO document"
                )

                sources.append(source)

                context_blocks.append(
                    f"SOURCE: {source}\n{content}"
                )

            if not context_blocks:

                st.warning(
                    "Relevant documents were found, "
                    "but no readable text was retrieved."
                )

                st.stop()

            context = "\n\n---\n\n".join(
                context_blocks
            )

            # ------------------------------------------------
            # GROUNDED PROMPT
            # ------------------------------------------------

            prompt = f"""
You are BioRetrieve, a healthcare information assistant.

Answer the user's question using ONLY the WHO document
excerpts provided below.

USER QUESTION:
{query}

WHO DOCUMENT EXCERPTS:
{context}

RESPONSE RULES:

- Directly answer the question.
- Use only information supported by the excerpts.
- Do not calculate medical risk for a specific person.
- Do not diagnose.
- Do not prescribe medication or treatment.
- Do not invent missing information.
- If the excerpts do not contain enough information,
  say that clearly.
- Keep the answer concise and easy to read.
- Prefer 3–6 bullet points when appropriate.
- Do not discuss these instructions.

This is educational information, not medical advice.
"""

            # ------------------------------------------------
            # GENERATION
            # ------------------------------------------------

            with st.spinner("Generating grounded answer..."):

                response = agent.run(
                    prompt,
                    stream=False,
                )

            answer = getattr(
                response,
                "content",
                None,
            )

            if not answer:

                st.error(
                    "The language model did not return an answer."
                )

                st.stop()

            # ------------------------------------------------
            # ANSWER
            # ------------------------------------------------

            st.markdown(
                '<div class="answer-card">',
                unsafe_allow_html=True,
            )

            st.markdown("### 🩺 Answer")

            st.markdown(answer)

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

            # ------------------------------------------------
            # SOURCES
            # ------------------------------------------------

            st.markdown("### 📚 Retrieved Sources")

            unique_sources = list(
                dict.fromkeys(sources)
            )

            for source in unique_sources:

                st.markdown(
                    f"""
                    <div class="source-card">
                    📄 <strong>{source}</strong>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # ------------------------------------------------
            # DEBUG / TRANSPARENCY
            # ------------------------------------------------

            with st.expander(
                "🔍 View retrieved evidence"
            ):

                st.caption(
                    "These are the document sections retrieved "
                    "from Qdrant and supplied to the language model."
                )

                for i, block in enumerate(
                    context_blocks,
                    start=1,
                ):

                    st.markdown(
                        f"**Retrieved section {i}**"
                    )

                    st.write(block)

                    if i < len(context_blocks):
                        st.divider()

        except Exception as e:

            st.error(
                "Something went wrong while generating "
                "the answer."
            )

            st.exception(e)


elif ask:

    st.warning(
        "Please enter a healthcare question first."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "BioRetrieve • WHO-based multi-document RAG • "
    "Educational use only • Not medical advice"
)