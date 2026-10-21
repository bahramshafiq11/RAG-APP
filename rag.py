"""
rag.py: the AI logic of "Chat With Your Documents".

Both app.py (the web interface) and evaluate.py (the accuracy test) import from here,
so the logic lives in one place.

How it works:
  1. Read text from your PDFs, page by page.
  2. Split the text into small overlapping chunks.
  3. Turn each chunk into an embedding (a list of numbers that captures meaning).
  4. When you ask a question, embed it too and find the most similar chunks.
  5. Send those chunks + your question to an LLM and ask it to answer ONLY from them.
"""

import json
from pathlib import Path

import numpy as np
import ollama
import pymupdf  # reads PDFs

# ---------------- Settings (try changing these later!) ----------------
EMBED_MODEL = "nomic-embed-text"  # Ollama embedding model (run: ollama pull nomic-embed-text)
LLM_MODEL = "qwen3.5:9b"          # must match a NAME from `ollama list`
# Starting values for the sidebar sliders
DEFAULT_CHUNK_SIZE = 200          # words per chunk
DEFAULT_TOP_K = 4                 # how many chunks to give the LLM
OVERLAP_RATIO = 0.2               # neighbouring chunks share 20% of their words

# Processed documents are saved here so they survive an app restart
SAVE_DIR = Path(__file__).parent / "saved_index"
# Your slider settings (like TOP_K) are saved here
SETTINGS_FILE = Path(__file__).parent / "settings.json"


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


# ---------------- Saving and loading processed documents ----------------
def save_index(chunks, vectors, chunk_size, filenames):
    """Write the chunks, their vectors and some details to the saved_index folder."""
    SAVE_DIR.mkdir(exist_ok=True)
    np.save(SAVE_DIR / "vectors.npy", vectors)  # the numbers, in NumPy's own format
    with open(SAVE_DIR / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)  # the text, page and file of each chunk
    info = {"chunk_size": chunk_size, "embed_model": EMBED_MODEL, "files": filenames}
    with open(SAVE_DIR / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def load_index():
    """Return (chunks, vectors, info) from disk, or None if nothing usable is saved."""
    try:
        with open(SAVE_DIR / "info.json", encoding="utf-8") as f:
            info = json.load(f)
        # Vectors from a different embedding model can't be compared with new questions
        if info["embed_model"] != EMBED_MODEL:
            return None
        with open(SAVE_DIR / "chunks.json", encoding="utf-8") as f:
            chunks = json.load(f)
        vectors = np.load(SAVE_DIR / "vectors.npy")
        return chunks, vectors, info
    except (FileNotFoundError, KeyError, json.JSONDecodeError, ValueError):
        return None


def load_settings():
    """Return the saved settings as a dictionary, or an empty one if there are none."""
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_settings(settings):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)


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


def stream_llm(prompt):
    """Yield the answer a few words at a time, as the model writes it."""
    # stream=True makes Ollama send pieces of the answer as soon as they're ready
    # think=False skips Qwen's hidden "thinking" step so answers come back faster
    stream = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}],
                         stream=True, think=False)
    for part in stream:
        yield part["message"]["content"]


def answer_question(question, chunks, vectors, k):
    """Search, then return (full answer text, retrieved chunks). Used by evaluate.py."""
    hits = search(question, chunks, vectors, k)
    answer = "".join(stream_llm(build_prompt(question, hits)))
    return answer, hits