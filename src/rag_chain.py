
from langchain_core.documents import Document

from src.pdf_loader import load_and_split_pdf
from src.retriever import build_vectorstore, retrieve_relevant_chunks
from src.llm import generate_answer


def answer_question(
    pdf_bytes: bytes,
    question: str,
    embeddings,
) -> dict:
    """Answer a question using evidence retrieved from the PDF."""

    if not question.strip():
        return {
            "answer": "Please enter a question.",
            "sources": [],
        }

    # 1. Read PDF and split it into chunks.
    chunks = load_and_split_pdf(pdf_bytes)

    # 2. Create a searchable vector index.
    vectorstore = build_vectorstore(chunks, embeddings)

    # 3. Retrieve candidate chunks.
    matches = retrieve_relevant_chunks(
        vectorstore,
        question,
        k=4,
    )

    if not matches:
        return {
            "answer": "I could not find the answer in the provided PDF.",
            "sources": [],
        }

    # 4. Prepare context and source page references.
    context_parts = []
    sources = []

    for doc in matches:
        page_number = doc.metadata.get("page", 0) + 1
        context_parts.append(
            f"[PDF page {page_number}]\n{doc.page_content}"
        )
        sources.append({
            "page": page_number,
            "text": doc.page_content,
        })

    context = "\n\n".join(context_parts)

    # 5. Generate an answer from the retrieved context.
    answer = generate_answer(
        question=question,
        context=context,
    )

    return {
        "answer": answer,
        "sources": sources,
    }
