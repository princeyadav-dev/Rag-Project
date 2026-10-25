
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings() -> Embeddings:
    """Create embeddings locally using a sentence-transformer model."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def build_vectorstore(
    chunks: list[Document],
    embeddings: Embeddings,
) -> FAISS:
    if not chunks:
        raise ValueError("No PDF text chunks were provided.")

    return FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )


def retrieve_relevant_chunks(
    vectorstore: FAISS,
    question: str,
    k: int = 4,
) -> list[Document]:
    if not question.strip():
        return []

    return vectorstore.similarity_search(
        query=question,
        k=k,
    )
