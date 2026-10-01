import streamlit as st
import requests
import json
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
if "://" not in BACKEND_URL:
    BACKEND_URL = f"http://{BACKEND_URL}"

st.set_page_config(page_title="Production RAG Assistant", page_icon="🚀")

if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

st.title("🚀 Production RAG Assistant")
st.caption(
    f"Backend: {BACKEND_URL} | Embeddings: BGE-small (HF) | Generation: Groq | "
    "Pipeline: rewrite → retrieve → rerank → generate → verify"
)

# ---- Backend health check ----
try:
    health = requests.get(f"{BACKEND_URL}/health", timeout=90)
    if health.status_code != 200:
        st.error("Backend is not healthy. Please check the API service.")
except requests.exceptions.RequestException:
    st.error("⚠️ Cannot reach the backend API. Is it running?")
    st.stop()

# ---- Upload ----
uploaded_file = st.file_uploader("Upload a PDF, .txt, or .md file", type=["pdf", "txt", "md"])

if uploaded_file is not None and uploaded_file.name != st.session_state.doc_name:
    with st.spinner("Uploading, chunking, and embedding document..."):
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            resp = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            st.session_state.doc_id = data["doc_id"]
            st.session_state.doc_name = uploaded_file.name
            st.success(f"Indexed '{data['filename']}' into {data['chunk_count']} chunks.")
        except requests.exceptions.HTTPError as e:
            st.error(f"Upload failed: {resp.json().get('detail', str(e))}")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach backend: {e}")

# ---- Ask questions ----
GROUNDEDNESS_STYLE = {
    "GROUNDED": ("🟢", "Grounded — answer is supported by the document"),
    "UNCERTAIN": ("🟡", "Uncertain — please verify against the source"),
    "NOT_GROUNDED": ("🔴", "Low confidence — answer may not be fully supported"),
}

if st.session_state.doc_id:
    st.divider()
    question = st.text_input("Ask a question about the document:")
    ask_clicked = st.button("Ask", type="primary")

    if ask_clicked:
        if not question or not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/query",
                    json={"doc_id": st.session_state.doc_id, "question": question},
                    stream=True,
                    timeout=120,
                )
                resp.raise_for_status()

                st.subheader("💡 Answer")
                answer_placeholder = st.empty()
                accumulated = ""
                final_payload = None

                for line in resp.iter_lines():
                    if not line:
                        continue
                    event = json.loads(line)

                    if event["type"] == "token":
                        accumulated += event["content"]
                        answer_placeholder.markdown(accumulated + "▌")
                    elif event["type"] == "final":
                        final_payload = event
                    elif event["type"] == "error":
                        st.error(f"Something went wrong: {event['detail']}")
                        accumulated = ""
                        break

                if accumulated:
                    answer_placeholder.markdown(accumulated)

                if final_payload:
                    icon, label = GROUNDEDNESS_STYLE.get(
                        final_payload["groundedness"], ("⚪", "Unknown")
                    )
                    st.caption(f"{icon} **{label}**")
                    if final_payload.get("groundedness_note"):
                        st.caption(f"_{final_payload['groundedness_note']}_")

                    with st.expander("🔍 Query rewriting (observability)"):
                        st.write(f"**Original question:** {question}")
                        st.write(f"**Rewritten search query:** {final_payload['rewritten_query']}")

                    with st.expander("📚 Retrieved & reranked sources"):
                        for i, src in enumerate(final_payload["sources"], 1):
                            st.markdown(f"**Source {i}** — cosine similarity: `{src['score']:.3f}`")
                            st.caption(src["text"][:500])

            except requests.exceptions.HTTPError:
                st.error(f"Query failed: {resp.json().get('detail', 'Unknown error')}")
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach backend: {e}")
else:
    st.info("Upload a document above to get started.")
