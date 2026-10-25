# PDF RAG Chatbot

Streamlit app jisme aap PDF upload karke uske content par sawal pooch sakte hain.

## Structure
- `app.py` — Streamlit UI
- `src/pdf_loader.py` — PDF read + chunking
- `src/retriever.py` — relevant chunks search
- `src/llm.py` — model API connection
- `src/rag_chain.py` — retrieval + generation pipeline
- `data/pdfs/` — uploaded PDFs
- `data/vectorstore/` — saved index

## Setup
```bash
uv sync
cp .env.example .env   # phir .env mein API key daalein
uv run streamlit run app.py
```

> `.env` ko kabhi GitHub par push na karein.
