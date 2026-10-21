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
CHUNK_SIZE = 200                  # words per chunk
OVERLAP = 40                      # words shared between neighbouring chunks
TOP_K = 4                         # how many chunks to give the LLM


# ---------------- Step 1: Read PDFs ----------------
def extract_pages(file_bytes):
    """Return a list of (page_number, text) for every page in the PDF."""
    doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    return [(i + 1, page.get_text()) for i, page in enumerate(doc)]


# ---------------- Step 2: Chunk the text ----------------
def chunk_pages(pages, filename):
    """Split each page into overlapping word chunks, remembering where each came from."""
    chunks = []
    step = CHUNK_SIZE - OVERLAP
    for page_num, text in pages:
        words = text.split()
        for start in range(0, len(words), step):
            piece = " ".join(words[start:start + CHUNK_SIZE])
            if piece.strip():
                chunks.append({"text": piece, "page": page_num, "source": filename})
            if start + CHUNK_SIZE >= len(words):
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
def search(question, chunks, vectors, k=TOP_K):
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
    response = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}], think=False)
    return response["message"]["content"]


# ---------------- The web interface ----------------
st.set_page_config(page_title="Chat With Your Documents", page_icon="📄")
st.title("📄 Chat With Your Documents")

# session_state keeps data alive between button clicks
for key, default in {"chunks": [], "vectors": None, "messages": []}.items():
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    st.header("1. Upload PDFs")
    files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    if st.button("Process documents", disabled=not files):
        with st.spinner("Reading and embedding your documents..."):
            all_chunks = []
            for f in files:
                all_chunks += chunk_pages(extract_pages(f.read()), f.name)
            if all_chunks:
                try:
                    st.session_state.vectors = embed([c["text"] for c in all_chunks])
                    st.session_state.chunks = all_chunks
                    st.session_state.messages = []
                    st.success(f"Ready! {len(all_chunks)} chunks from {len(files)} file(s).")
                except Exception as e:
                    st.error(f"Embedding failed. Is Ollama running, and did you run "
                             f"`ollama pull {EMBED_MODEL}`? Error: {e}")
            else:
                st.error("No text found. Scanned PDFs need OCR first.")
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

        hits = search(question, st.session_state.chunks, st.session_state.vectors)
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