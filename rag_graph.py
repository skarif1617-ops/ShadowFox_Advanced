"""Multi-step LangGraph RAG pipeline:

    rewrite_query -> retrieve -> rerank -> generate -> check_groundedness
                                                 ^                |
                                                 └── retry (max 1) ┘

Each stage is a separate, inspectable node with a single responsibility —
this is what separates a "production-style" pipeline from a single
retrieve-then-generate function:

- rewrite_query: turns a possibly vague user question into a sharper
  search query (helps retrieval when the user's phrasing doesn't match
  the document's wording).
- retrieve: vector search over the document's embeddings (broad recall —
  pulls more candidates than we'll ultimately use).
- rerank: an LLM pass that filters/reorders those candidates by actual
  relevance to the ORIGINAL question (vector similarity alone can surface
  topically-similar but not-actually-useful passages).
- generate: produces a draft answer strictly from the reranked context.
- check_groundedness: an LLM pass that verifies the draft answer is
  actually supported by the context. If not, the pipeline retries
  generation once with a stricter instruction; otherwise the answer is
  returned with a confidence-style label the UI can display.
"""

import os
import re
import json
from typing import TypedDict, List, Tuple
from langgraph.graph import StateGraph, END
from openai import OpenAI

from vector_store import search

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
_client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

MODEL_CANDIDATES = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
]

RETRIEVE_K = 6   # broad candidate pool for reranking
FINAL_K = 3      # chunks actually used for generation


class RAGState(TypedDict):
    doc_id: str
    question: str
    rewritten_query: str
    candidates: List[Tuple[str, float]]
    reranked: List[Tuple[str, float]]
    answer: str
    groundedness: str        # "GROUNDED" | "NOT_GROUNDED" | "UNCERTAIN"
    groundedness_note: str
    retry_count: int


def _call_with_failover(messages: list[dict], temperature: float = 0.2, max_tokens: int = 500) -> str:
    last_error = None
    for model in MODEL_CANDIDATES:
        try:
            response = _client.chat.completions.create(
                model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            err_msg = str(e).lower()
            if any(k in err_msg for k in ["model_not_found", "404", "timed out", "timeout", "503"]):
                continue
            raise e
    raise last_error or RuntimeError("All model candidates failed.")


# ---------------------------------------------------------------------------
# Node 1: query rewriting
# ---------------------------------------------------------------------------
def rewrite_query_node(state: RAGState) -> RAGState:
    prompt = (
        "Rewrite the following user question into a short, keyword-focused "
        "search query optimized for retrieving relevant passages from a "
        "document. Keep it factual and concise — output ONLY the rewritten "
        "query, nothing else.\n\n"
        f"Question: {state['question']}"
    )
    try:
        rewritten = _call_with_failover(
            [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=60
        )
        state["rewritten_query"] = rewritten.strip().strip('"')
    except Exception:
        # if rewriting fails for any reason, fall back to the raw question
        state["rewritten_query"] = state["question"]
    return state


# ---------------------------------------------------------------------------
# Node 2: retrieval (vector search)
# ---------------------------------------------------------------------------
def retrieve_node(state: RAGState) -> RAGState:
    results = search(state["doc_id"], state["rewritten_query"], k=RETRIEVE_K)
    state["candidates"] = results
    return state


# ---------------------------------------------------------------------------
# Node 3: reranking (LLM relevance filter over the vector-search candidates)
# ---------------------------------------------------------------------------
def rerank_node(state: RAGState) -> RAGState:
    candidates = state["candidates"]

    if not candidates:
        state["reranked"] = []
        return state

    numbered = "\n\n".join(f"[{i+1}] {chunk[:400]}" for i, (chunk, _) in enumerate(candidates))
    prompt = (
        "You are ranking passages by relevance to a question. Given the "
        "question and numbered passages below, return a JSON array of the "
        f"passage numbers, most relevant first, keeping at most {FINAL_K}. "
        "Only include passages that are genuinely relevant. Output ONLY "
        "the JSON array, e.g. [2, 1, 5] — nothing else.\n\n"
        f"Question: {state['question']}\n\n"
        f"Passages:\n{numbered}"
    )

    try:
        raw = _call_with_failover(
            [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=100
        )
        match = re.search(r"\[[\d,\s]*\]", raw)
        indices = json.loads(match.group(0)) if match else []
        indices = [i for i in indices if 1 <= i <= len(candidates)]
        if not indices:
            raise ValueError("no valid indices returned")
        state["reranked"] = [candidates[i - 1] for i in indices[:FINAL_K]]
    except Exception:
        # fall back to plain vector-similarity order if reranking fails/parses badly
        state["reranked"] = candidates[:FINAL_K]

    return state


# ---------------------------------------------------------------------------
# Node 4: grounded generation
# ---------------------------------------------------------------------------
def generate_node(state: RAGState) -> RAGState:
    context = "\n\n---\n\n".join(chunk for chunk, _ in state["reranked"])

    strict_note = ""
    if state.get("retry_count", 0) > 0:
        strict_note = (
            "\n\nIMPORTANT: Your previous answer was flagged as not fully "
            "supported by the context. Re-answer using ONLY facts that "
            "literally appear in the context below. If the context does "
            "not contain the answer, say so explicitly instead of guessing."
        )

    prompt = (
        "You are a helpful assistant answering questions based only on the "
        "provided document context. If the answer is not in the context, "
        "say 'I do not have enough information in the provided document to "
        f"answer that.' — do not make anything up.{strict_note}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {state['question']}\n\n"
        "Answer clearly and concisely, citing relevant details from the context."
    )

    state["answer"] = _call_with_failover(
        [
            {"role": "system", "content": "You are a helpful, strictly document-grounded assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=500,
    )
    return state


# ---------------------------------------------------------------------------
# Node 5: groundedness check
# ---------------------------------------------------------------------------
def check_groundedness_node(state: RAGState) -> RAGState:
    context = "\n\n---\n\n".join(chunk for chunk, _ in state["reranked"])
    prompt = (
        "You are a fact-checking assistant. Given the CONTEXT and the "
        "ANSWER below, determine whether the answer is fully supported by "
        "the context (no invented facts, no outside knowledge). Respond "
        "with exactly one word first — GROUNDED, NOT_GROUNDED, or "
        "UNCERTAIN — followed by a colon and a one-sentence reason.\n\n"
        f"Context:\n{context}\n\n"
        f"Answer:\n{state['answer']}"
    )

    try:
        verdict = _call_with_failover(
            [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=80
        ).strip()
        label = verdict.split(":")[0].strip().upper()
        if label not in ("GROUNDED", "NOT_GROUNDED", "UNCERTAIN"):
            label = "UNCERTAIN"
        note = verdict.split(":", 1)[1].strip() if ":" in verdict else ""
    except Exception:
        label, note = "UNCERTAIN", "Groundedness check could not be completed."

    state["groundedness"] = label
    state["groundedness_note"] = note
    state["retry_count"] = state.get("retry_count", 0) + 1
    return state


def _route_after_groundedness(state: RAGState) -> str:
    if state["groundedness"] == "NOT_GROUNDED" and state.get("retry_count", 0) <= 1:
        return "retry"
    return "end"


def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate", generate_node)
    graph.add_node("check_groundedness", check_groundedness_node)

    graph.set_entry_point("rewrite_query")
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", "check_groundedness")
    graph.add_conditional_edges(
        "check_groundedness", _route_after_groundedness, {"retry": "generate", "end": END}
    )
    return graph.compile()


rag_app = build_graph()


def run_rag(doc_id: str, question: str) -> RAGState:
    initial_state: RAGState = {
        "doc_id": doc_id,
        "question": question,
        "rewritten_query": "",
        "candidates": [],
        "reranked": [],
        "answer": "",
        "groundedness": "",
        "groundedness_note": "",
        "retry_count": 0,
    }
    return rag_app.invoke(initial_state)