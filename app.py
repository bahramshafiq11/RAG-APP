"""
Chat With Your Documents: the web interface.

The AI logic is in rag.py. This file only builds the page.

Run with:  streamlit run app.py
"""

import shutil

import streamlit as st

from rag import (DEFAULT_CHUNK_SIZE, DEFAULT_TOP_K, EMBED_MODEL, SAVE_DIR, build_prompt,
                 chunk_pages, embed, extract_pages, load_index, load_settings, save_index,
                 save_settings, search, stream_llm)

# ---------------- The web interface ----------------
st.set_page_config(page_title="Chat With Your Documents", page_icon="📄")
st.title("📄 Chat With Your Documents")

# session_state keeps data alive between button clicks
for key, default in {"chunks": [], "vectors": None, "messages": [],
                     "processed_chunk_size": None, "loaded_files": [],
                     "tried_loading": False}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# On the first run of a new session, load previously processed documents from disk
if not st.session_state.tried_loading:
    st.session_state.tried_loading = True
    # Put the TOP_K slider back where it was last time
    st.session_state.top_k = load_settings().get("top_k", DEFAULT_TOP_K)
    saved = load_index()
    if saved is not None:
        chunks, vectors, info = saved
        st.session_state.chunks = chunks
        st.session_state.vectors = vectors
        st.session_state.processed_chunk_size = info["chunk_size"]
        st.session_state.loaded_files = info["files"]

with st.sidebar:
    st.header("1. Upload PDFs")
    files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    # Slider: st.slider(label, min, max, starting value, step)
    # Start at the size the loaded documents used, so the slider matches them
    start_size = st.session_state.processed_chunk_size or DEFAULT_CHUNK_SIZE
    chunk_size = st.slider("Chunk size (words)", 50, 500, start_size, 50,
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
                    st.session_state.loaded_files = [f.name for f in files]
                    save_index(all_chunks, st.session_state.vectors, chunk_size,
                               st.session_state.loaded_files)
                    st.success(f"Ready! {len(all_chunks)} chunks from {len(files)} file(s). "
                               "Saved, so they'll still be here next time.")
                except Exception as e:
                    st.error(f"Embedding failed. Is Ollama running, and did you run "
                             f"`ollama pull {EMBED_MODEL}`? Error: {e}")
            else:
                st.error("No text found. Scanned PDFs need OCR first.")

    # Show which documents are loaded, with a button to forget them
    if st.session_state.loaded_files:
        st.info("Loaded: " + ", ".join(st.session_state.loaded_files)
                + f" ({len(st.session_state.chunks)} chunks)")
        if st.button("Clear saved documents"):
            shutil.rmtree(SAVE_DIR, ignore_errors=True)
            st.session_state.chunks = []
            st.session_state.vectors = None
            st.session_state.messages = []
            st.session_state.processed_chunk_size = None
            st.session_state.loaded_files = []
            st.rerun()

    # Chunk size only takes effect when documents are processed, so remind the user
    processed = st.session_state.processed_chunk_size
    if processed is not None and processed != chunk_size:
        st.warning(f"Your documents were processed with chunk size {processed}. "
                   "Click **Process documents** again to use the new size.")

    st.header("Settings")
    # TOP_K takes effect on the very next question, no re-processing needed
    # key="top_k" links the slider to st.session_state.top_k, which was loaded above
    top_k = st.slider("Chunks to retrieve (TOP_K)", 1, 10, key="top_k",
                      help="How many chunks the AI reads before answering.")
    # Save the new position whenever you move the slider
    settings = load_settings()
    if settings.get("top_k") != top_k:
        settings["top_k"] = top_k
        save_settings(settings)
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
            # Show "Thinking..." until the first words arrive, then remove it
            status = st.empty()
            status.markdown("_Thinking..._")

            def words_then_clear_status():
                first = True
                for piece in stream_llm(build_prompt(question, hits)):
                    if first:
                        status.empty()
                        first = False
                    yield piece

            try:
                # st.write_stream prints each piece as it arrives and returns the full text
                answer = st.write_stream(words_then_clear_status())
            except Exception as e:
                status.empty()
                answer = f"Could not reach Ollama. Is it running? Error: {e}"
                st.markdown(answer)
            if show_sources:
                with st.expander("Retrieved chunks"):
                    for c, score in hits:
                        st.markdown(f"**{c['source']}, page {c['page']}** (score {score:.2f})")
                        st.caption(c["text"][:500] + "...")
        st.session_state.messages.append({"role": "assistant", "content": answer})