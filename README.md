# Production RAG Assistant (Advanced Level)

A production-style Retrieval-Augmented Generation (RAG) system with a
FastAPI backend, Streamlit frontend, Hugging Face embeddings + NumPy
vector search, Pydantic schema validation, a multi-step LangGraph
pipeline (query rewriting, retrieval, reranking, generation, and
groundedness verification), streaming answer delivery, and Docker
containerization.

## Architecture

```
[Streamlit frontend] --HTTP (streaming NDJSON)--> [FastAPI backend]
                                                          |
                                                          v
                                          [LangGraph pipeline, 5 nodes]
                                                          |
                        +---------------------+-----------------------+
                        v                     v                       v
              [Hugging Face API]        [in-memory vector store]  [Groq API]
              (BGE-small embeddings)     (NumPy cosine similarity)  (chat generation)
```

## The pipeline (LangGraph, 5 nodes)

```
rewrite_query -> retrieve -> rerank -> generate -> check_groundedness
                                            ^               |
                                            └── retry (max 1) ┘
```

1. **rewrite_query** — an LLM turns the raw user question into a shorter,
   keyword-focused search query. This helps when the user's phrasing
   doesn't closely match the document's own wording.
2. **retrieve** — vector search: the rewritten query is embedded
   (`BAAI/bge-small-en-v1.5`) and compared via cosine similarity against
   every chunk's embedding. Returns a broad candidate pool (top 6), not
   just the final answer set.
3. **rerank** — an LLM pass re-scores those 6 candidates against the
   *original* question and keeps only the genuinely relevant ones (top
   3). This catches cases where vector similarity surfaces a
   topically-similar but not actually useful passage — a known weakness
   of similarity search alone.
4. **generate** — produces a draft answer strictly from the reranked
   context, with an explicit instruction not to use outside knowledge.
5. **check_groundedness** — a second LLM pass verifies the draft answer
   is actually supported by the retrieved context. If it's flagged
   `NOT_GROUNDED`, the pipeline loops back to `generate` once more with a
   stricter instruction; otherwise the labeled result
   (`GROUNDED` / `UNCERTAIN` / `NOT_GROUNDED`) is returned to the client
   as a confidence-style indicator.

## Streaming design (and why it isn't naive token streaming)

A groundedness check requires the *complete* answer before it can be
verified — so streaming raw model output token-by-token straight to the
user would mean showing text before it's been checked at all. Instead:

1. The full pipeline (rewrite → retrieve → rerank → generate → verify,
   including the retry) runs server-side first.
2. Once the final, verified answer exists, the `/query` endpoint streams
   it back to the client as newline-delimited JSON (NDJSON) — word by
   word — followed by a final event carrying the sources, the rewritten
   query, and the groundedness label.
3. The Streamlit frontend consumes this with `requests.post(..., stream=True)`
   and progressively updates the answer with `st.empty()`.

This keeps the UI responsive and streaming-feeling without ever showing
an answer that hasn't passed the groundedness check — a deliberate
tradeoff between "stream everything raw" and "verify before showing
anything," explained here rather than left implicit.

## Why no FAISS / local ML runtime

FAISS, ONNX Runtime, and similar libraries ship compiled native binaries.
On Windows machines with strict Application Control / WDAC / AppLocker
policies (common on school or managed work devices), these can be
silently blocked, crashing the app with no clear Python-level error.
This system uses only HTTPS API calls for embeddings (Hugging Face) and
generation (Groq) — no local model inference anywhere — so it runs
cleanly regardless of local OS security restrictions. Tradeoff: retrieval
depends on network calls, and the in-memory NumPy store doesn't scale to
huge collections the way a real vector database would.

## Endpoints

- `GET /health` — liveness check.
- `POST /upload` — multipart file upload (PDF/txt/md, max 10MB). Returns
  `doc_id`, `filename`, `chunk_count`.
- `POST /query` — JSON body `{doc_id, question}`. Returns a streamed
  NDJSON response: a sequence of `{"type": "token", "content": "..."}`
  events, followed by one `{"type": "final", "sources": [...],
  "rewritten_query": "...", "groundedness": "...", "groundedness_note":
  "..."}` event.

## Run locally (without Docker)

```powershell
# From the project root, Terminal 1 - backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:HF_TOKEN="your_hugging_face_token"
$env:GROQ_API_KEY="your_groq_key"
uvicorn main:app --reload

# Terminal 2 - frontend (activate the same environment)
.\venv\Scripts\Activate.ps1
streamlit run app.py
```

## Run with Docker

```bash
cp .env.example .env
# edit .env and add your HF_TOKEN and GROQ_API_KEY
docker compose -f docker_compose.yml up --build
```

- Frontend: http://localhost:8501
- Backend docs: http://localhost:8000/docs

## Deploy from GitHub

GitHub stores the source code; it does not host this Streamlit and FastAPI
application. To deploy both services, push this repository to GitHub, then
create a Blueprint on Render using the repository's `render.yaml`. The
Blueprint creates the API and frontend services and connects the frontend to
the API's HTTPS URL. Enter `HF_TOKEN` and `GROQ_API_KEY` when prompted; do not
commit `.env` or put API keys in the Blueprint. The API has no authentication,
so do not upload sensitive documents.

The in-memory document index is temporary and is lost when the backend
restarts or sleeps. Free Render services may sleep when idle.

## Error handling & reliability

- Every backend endpoint wraps risky operations (file parsing, embedding
  calls, generation calls) in try/except, returning proper HTTP status
  codes (400 bad input, 404 missing document, 502 upstream API failure)
  instead of raw stack traces.
- Generation calls cascade across multiple candidate Groq models on
  rate-limit/timeout/404, before failing.
- Query rewriting and reranking both fall back gracefully (to the raw
  question, and to plain vector-similarity order, respectively) if the
  LLM call for that step fails or returns unparseable output — a broken
  rewrite/rerank step degrades the pipeline instead of crashing it.
- The frontend surfaces the backend's error `detail` directly instead of
  a generic failure message.

## What makes this "production-style" rather than a basic demo

- **Modular, multi-step pipeline** with a single responsibility per node,
  not one function doing retrieval-and-generation together.
- **Retrieval quality handled explicitly**: broad vector recall (6
  candidates) followed by an LLM relevance filter (rerank to 3) — not
  "take the top-3 by cosine similarity and hope."
- **Hallucination reduction handled explicitly, twice**: once via strict
  grounding instructions in the generation prompt, and again via an
  independent groundedness-verification pass with an automatic retry —
  not just "trust the prompt."
- **Typed request/response contracts** via Pydantic.
- **Observability**: the rewritten query, retrieved+reranked sources with
  similarity scores, and a groundedness label are all surfaced to the
  user, not hidden inside the model call.
- **Streaming delivery** engineered around the groundedness check rather
  than in spite of it.
- **Multi-format ingestion**: PDF, TXT, and Markdown.
- **Containerized**, two-service architecture.

## Limitations (worth mentioning in the video)

- The vector store is in-memory and per-process — restarting the backend
  loses all indexed documents (a real deployment would persist to disk or
  a managed vector DB such as Qdrant or pgvector).
- No authentication on the API.
- Reranking and groundedness checking each add an extra LLM call per
  question — a deliberate quality/latency tradeoff, mitigated by Groq's
  fast inference.
- Embedding one chunk at a time via the HF Inference API is simple but
  not the fastest approach at scale; batching would be the next
  optimization.