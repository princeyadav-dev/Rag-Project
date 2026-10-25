
from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_pdf(pdf_bytes: bytes):
    """Load a PDF safely on Windows and split it into chunks."""

    # Create a temporary directory instead of keeping an open
    # temporary file handle while PyPDFLoader reads the PDF.
    with TemporaryDirectory(prefix="pdf_rag_") as temp_dir:
        pdf_path = Path(temp_dir) / "uploaded.pdf"
        pdf_path.write_bytes(pdf_bytes)

        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

    if not pages:
        raise ValueError("No readable pages found in the PDF.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=True,
    )

    chunks = splitter.split_documents(pages)

    if not chunks:
        raise ValueError(
            "No readable text found. The PDF may be scanned or image-only."
        )

    return chunks
