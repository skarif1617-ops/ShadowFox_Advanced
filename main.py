import os
import uuid
import json
import time
import logging
import io

from dotenv import load_dotenv
import os as _os
load_dotenv(dotenv_path=_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".env"), encoding="utf-8-sig")

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pypdf import PdfReader

from schemas import UploadResponse, QueryRequest
from vector_store import build_index, doc_exists
from rag_graph import run_rag

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-backend")

if not os.getenv("HF_TOKEN"):
    raise RuntimeError("HF_TOKEN environment variable is not set.")
if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY environment variable is not set.")

app = FastAPI(title="Production RAG Assistant API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in real production
    allow_methods=["*"],
    allow_headers=["*"],
)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MAX_FILE_SIZE_MB = 10
ALLOWED_EXTENSIONS = (".pdf", ".txt", ".md")


def extract_text(filename: str, raw_bytes: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(raw_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    # .txt and .md are both plain UTF-8 text — Markdown syntax doesn't need
    # special stripping for retrieval purposes, headers/lists still carry
    # useful signal for embeddings.
    return raw_bytes.decode("utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks, start, text = [], 0, text.strip()
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(400, f"Only {', '.join(ALLOWED_EXTENSIONS)} files are supported.")

    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(400, f"File too large (max {MAX_FILE_SIZE_MB}MB).")

    try:
        text = extract_text(file.filename, raw_bytes)
    except Exception as e:
        logger.exception("Failed to extract text")
        raise HTTPException(400, f"Could not read file: {e}")

    if not text.strip():
        raise HTTPException(400, "No extractable text found in this file.")

    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    doc_id = str(uuid.uuid4())

    try:
        build_index(doc_id, chunks)
    except Exception as e:
        logger.exception("Failed to build vector index")
        raise HTTPException(502, f"Embedding service error: {e}")

    logger.info(f"Indexed doc_id={doc_id} filename={file.filename} chunks={len(chunks)}")
    return UploadResponse(doc_id=doc_id, filename=file.filename, chunk_count=len(chunks))


@app.post("/query")
def query_document(request: QueryRequest):
    """Streams the response as newline-delimited JSON (NDJSON).

    Design note: a groundedness check requires the COMPLETE answer before
    it can be verified, so true token-by-token streaming straight from the
    model would bypass that safety check entirely. Instead, this endpoint
    runs the full pipeline (rewrite -> retrieve -> rerank -> generate ->
    verify, with one retry if unverified) server-side first, THEN streams
    the already-validated answer back to the client word-by-word — giving
    a responsive, streaming UI without sacrificing groundedness
    verification. This tradeoff is deliberate and documented in the README.
    """
    if not doc_exists(request.doc_id):
        raise HTTPException(404, "Document not found. Upload it first via /upload.")

    def event_stream():
        try:
            result = run_rag(request.doc_id, request.question)
        except Exception as e:
            logger.exception("RAG pipeline failed")
            yield json.dumps({"type": "error", "detail": str(e)}) + "\n"
            return

        answer = result["answer"]
        for word in answer.split(" "):
            yield json.dumps({"type": "token", "content": word + " "}) + "\n"
            time.sleep(0.015)

        final_payload = {
            "type": "final",
            "rewritten_query": result["rewritten_query"],
            "sources": [{"text": c, "score": s} for c, s in result["reranked"]],
            "groundedness": result["groundedness"],
            "groundedness_note": result["groundedness_note"],
        }
        yield json.dumps(final_payload) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")