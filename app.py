"""
Chat With Your Documents — a beginner-friendly RAG app.

How it works:
  1. Read text from your PDFs, page by page.
  2. Split the text into small overlapping chunks.
  3. Turn each chunk into an embedding (a list of numbers that captures meaning).
  4. When you ask a question, embed it too and find the most similar chunks.
  5. Send those chunks + your question to an LLM and ask it to answer ONLY from them.

Run with:  streamlit run app.py
"""

import pymupdf  # reads PDFs
import numpy as np
import ollama
import streamlit as st

# ---------------- Settings (try changing these later!) ----------------
EMBED_MODEL = "nomic-embed-text"  # Ollama embedding model (run: ollama pull nomic-embed-text)
LLM_MODEL = "qwen3.5:9b"          # must match a NAME from `ollama list`
# Starting values for the sidebar sliders
DEFAULT_CHUNK_SIZE = 200          # words per chunk
DEFAULT_TOP_K = 4                 # how many chunks to give the LLM
OVERLAP_RATIO = 0.2               # neighbouring chunks share 20% of their words


# ---------------- Step 1: Read PDFs ----------------
def extract_pages(file_bytes):
    """Return a list of (page_number, text) for every page in the PDF."""
    doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    return [(i + 1, page.get_text()) for i, page in enumerate(doc)]


# ---------------- Step 2: Chunk the text ----------------
def chunk_pages(pages, filename, chunk_size):
    """Split each page into overlapping word chunks, remembering where each came from."""
    chunks = []
    overlap = int(chunk_size * OVERLAP_RATIO)
    step = chunk_size - overlap
    for page_num, text in pages:
        words = text.split()
        for start in range(0, len(words), step):
            piece = " ".join(words[start:start + chunk_size])
            if piece.strip():
                chunks.append({"text": piece, "page": page_num, "source": filename})
            if start + chunk_size >= len(words):
                break
    return chunks


# ---------------- Step 3: Embeddings ----------------
def embed(texts, batch_size=32):
    """Turn a list of texts into normalized vectors using Ollama."""
    vectors = []
    for i in range(0, len(texts), batch_size):
        response = ollama.embed(model=EMBED_MODEL, input=texts[i:i + batch_size])
        vectors.extend(response["embeddings"])
    vectors = np.array(vectors, dtype=np.float32)
    # Scale every vector to length 1 so a dot product equals cosine similarity
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


# ---------------- Step 4: Search ----------------
def search(question, chunks, vectors, k):
    """Find the k chunks most similar in meaning to the question."""
    q = embed([question])[0]
    scores = vectors @ q                      # cosine similarity with every chunk
    best = np.argsort(scores)[::-1][:k]       # indices of the highest scores
    return [(chunks[i], float(scores[i])) for i in best]


# ---------------- Step 5: Ask the LLM ----------------
def build_prompt(question, hits):
    context = "\n\n".join(
        f"[{c['source']}, page {c['page']}]\n{c['text']}" for c, _ in hits
    )
    return f"""You are a helpful assistant that answers questions about the user's documents.
Use ONLY the context below. If the answer is not in the context, say
"I couldn't find that in your documents." Cite sources like (file.pdf, page 3).

Context:
{context}

Question: {question}
Answer:"""


def ask_llm(prompt):
    # think=False skips Qwen's hidden "thinking" step so answers come back faster
    response = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}],
                           think=False)
    return response["message"]["content"]


# ---------------- The web interface ----------------
st.set_page_config(page_title="Chat With Your Documents", page_icon="📄")
st.title("📄 Chat With Your Documents")

# session_state keeps data alive between button clicks
for key, default in {"chunks": [], "vectors": None, "messages": [],
                     "processed_chunk_size": None}.items():
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    st.header("1. Upload PDFs")
    files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    # Slider: st.slider(label, min, max, starting value, step)
    chunk_size = st.slider("Chunk size (words)", 50, 500, DEFAULT_CHUNK_SIZE, 50,
                           help="Smaller = more precise search. Bigger = more context per chunk.")

    if st.button("Process documents", disabled=not files):
        with st.spinner("Reading and embedding your documents..."):
            all_chunks = []
            for f in files:
                all_chunks += chunk_pages(extract_pages(f.read()), f.name, chunk_size)
            if all_chunks:
                try:
                    st.session_state.vectors = embed([c["text"] for c in all_chunks])
                    st.session_state.chunks = all_chunks
                    st.session_state.messages = []
                    st.session_state.processed_chunk_size = chunk_size
                    st.success(f"Ready! {len(all_chunks)} chunks from {len(files)} file(s).")
                except Exception as e:
                    st.error(f"Embedding failed. Is Ollama running, and did you run "
                             f"`ollama pull {EMBED_MODEL}`? Error: {e}")
            else:
                st.error("No text found. Scanned PDFs need OCR first.")

    # Chunk size only takes effect when documents are processed, so remind the user
    processed = st.session_state.processed_chunk_size
    if processed is not None and processed != chunk_size:
        st.warning(f"Your documents were processed with chunk size {processed}. "
                   "Click **Process documents** again to use the new size.")

    st.header("Settings")
    # TOP_K takes effect on the very next question, no re-processing needed
    top_k = st.slider("Chunks to retrieve (TOP_K)", 1, 10, DEFAULT_TOP_K,
                      help="How many chunks the AI reads before answering.")
    show_sources = st.checkbox("Show retrieved chunks", value=True)

st.header("2. Ask a question")

# Replay the conversation so far
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask something about your documents...")
if question:
    if st.session_state.vectors is None:
        st.warning("Upload and process a PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        hits = search(question, st.session_state.chunks, st.session_state.vectors, top_k)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = ask_llm(build_prompt(question, hits))
                except Exception as e:
                    answer = f"Could not reach Ollama. Is it running? Error: {e}"
            st.markdown(answer)
            if show_sources:
                with st.expander("Retrieved chunks"):
                    for c, score in hits:
                        st.markdown(f"**{c['source']}, page {c['page']}** (score {score:.2f})")
                        st.caption(c["text"][:500] + "...")
        st.session_state.messages.append({"role": "assistant", "content": answer})