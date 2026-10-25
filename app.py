
import streamlit as st

from src.retriever import get_embeddings
from src.rag_chain import answer_question


st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
)

# Reuse the embedding model instead of loading it on every rerun.
@st.cache_resource
def load_embeddings():
    return get_embeddings()


st.markdown(
    """
    <style>
    .stApp {
        background: #0b1120;
    }

    .block-container {
        max-width: 1050px;
        padding-top: 2rem;
    }

    .hero {
        padding: 25px;
        border-radius: 18px;
        border: 1px solid #26354d;
        background: linear-gradient(135deg, #17243b, #101827);
        margin-bottom: 22px;
    }

    .hero h1 {
        color: #f8fafc;
    }

    .hero p {
        color: #bdc9dc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("📚 PDF RAG Assistant")
    st.caption("Ask questions from your own documents")
    st.divider()

    uploaded_pdf = st.file_uploader(
        "Upload your PDF",
        type=["pdf"],
    )

    if uploaded_pdf:
        st.success("PDF selected")
        st.caption(uploaded_pdf.name)

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("LLM: openai/gpt-oss-120b")
    st.caption("Embeddings: MiniLM (local)")


st.markdown(
    """
    <div class="hero">
        <h1>Ask your PDF anything ✨</h1>
        <p>
            Upload a document, ask a question, and explore
            answers grounded in its content.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if uploaded_pdf is None:
    st.info("Upload a PDF from the sidebar to get started.")
    st.subheader("How it works")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📄 Upload")
        st.write("Choose a PDF document.")

    with col2:
        st.markdown("### 🔎 Retrieve")
        st.write("Search relevant passages in the PDF.")

    with col3:
        st.markdown("### 💬 Ask")
        st.write("Get an answer with source pages.")

else:
    # Show previous conversation
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("View PDF sources"):
                    for source in message["sources"]:
                        st.markdown(
                            f"**Page {source['page']}**"
                        )
                        st.write(source["text"])

    question = st.chat_input(
        "Ask a question about your PDF..."
    )

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.spinner("Reading PDF and finding relevant content..."):
            try:
                embeddings = load_embeddings()

                result = answer_question(
                    pdf_bytes=uploaded_pdf.getvalue(),
                    question=question,
                    embeddings=embeddings,
                )

                answer = result["answer"]
                sources = result.get("sources", [])

            
            except Exception as exc:
                st.error("Question process karte waqt error aaya.")
                st.exception(exc)
                raise


        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )

        st.rerun()
