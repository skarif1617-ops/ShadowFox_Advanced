"""In-memory vector store using Hugging Face embeddings + NumPy cosine
similarity search. One entry per uploaded document, keyed by doc_id.

Deliberately avoids FAISS/ONNX/local ML runtimes — both embeddings and
generation are pure HTTPS API calls, so this runs cleanly even on machines
with strict Application Control / WDAC / AppLocker policies that block
compiled native DLLs.
"""

import os
import numpy as np
from huggingface_hub import InferenceClient

HF_TOKEN = os.getenv("HF_TOKEN")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

_hf_client = InferenceClient(token=HF_TOKEN)

# doc_id -> {"chunks": list[str], "embeddings": np.ndarray}
_STORE: dict[str, dict] = {}


def embed_documents(chunks: list[str]) -> np.ndarray:
    """Embed document chunks as-is (no instruction prefix)."""
    vectors = []
    for chunk in chunks:
        vec = _hf_client.feature_extraction(chunk, model=EMBED_MODEL)
        vectors.append(np.array(vec, dtype="float32").squeeze())
    return np.array(vectors, dtype="float32")


def embed_query(query: str) -> np.ndarray:
    """BGE models are trained asymmetrically — queries get an instruction
    prefix so they align correctly with passage embeddings in vector space."""
    instructed = f"Represent this sentence for searching relevant passages: {query}"
    vec = _hf_client.feature_extraction(instructed, model=EMBED_MODEL)
    return np.array(vec, dtype="float32").squeeze()


def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    doc_norms = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10)
    return doc_norms @ query_norm


def build_index(doc_id: str, chunks: list[str]) -> None:
    embeddings = embed_documents(chunks)
    _STORE[doc_id] = {"chunks": chunks, "embeddings": embeddings}


def doc_exists(doc_id: str) -> bool:
    return doc_id in _STORE


def search(doc_id: str, query: str, k: int = 6) -> list[tuple[str, float]]:
    """Vector search: returns top-k candidates by cosine similarity.
    Called with a larger k than the final answer needs, so the rerank
    step downstream has real candidates to choose from."""
    entry = _STORE[doc_id]
    query_vec = embed_query(query)
    scores = cosine_similarity(query_vec, entry["embeddings"])
    top_idx = np.argsort(scores)[::-1][:k]
    return [(entry["chunks"][i], float(scores[i])) for i in top_idx]