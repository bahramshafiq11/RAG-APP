# 📄 Chat With Your Documents

A local AI assistant that answers questions about your PDFs, with page-level citations. Everything runs on your own computer, so your documents never leave your machine.

Built with **RAG (Retrieval-Augmented Generation)**: the app first finds the most relevant parts of your documents, then asks a language model to answer using only those parts.

<!-- Add a screenshot: save it as screenshot.png in this folder, then remove the arrows around the next line -->
<!-- ![App screenshot](screenshot.png) -->

## Features

- Upload one or more PDFs and ask questions in plain language
- Answers cite the source file and page number
- Says so when the answer isn't in your documents, instead of making something up
- Shows the retrieved passages so you can check where each answer came from
- Fully offline: uses local models through [Ollama](https://ollama.com)

## How it works

1. **Extract:** pull the text from each PDF page with PyMuPDF
2. **Chunk:** split the text into overlapping 200-word pieces
3. **Embed:** turn each chunk into a vector with `nomic-embed-text`
4. **Retrieve:** embed the question and find the most similar chunks with cosine similarity
5. **Generate:** send those chunks and the question to `qwen3.5:9b` with instructions to answer only from that context

## Tech stack

Python · Streamlit · Ollama · Qwen 3.5 · nomic-embed-text · PyMuPDF · NumPy

## Getting started

**Requirements:** Python 3.10+, [Ollama](https://ollama.com), and about 8 GB of free RAM.

```bash
# 1. Download the models
ollama pull qwen3.5:9b
ollama pull nomic-embed-text

# 2. Clone the project and install the packages
git clone https://github.com/<your-username>/rag-app.git
cd rag-app
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt

# 3. Run it
streamlit run app.py
```

Then open http://localhost:8501, upload a PDF, click **Process documents**, and ask a question.

## Configuration

You can change these settings at the top of `app.py`:

| Setting | Default | What it does |
|---|---|---|
| `LLM_MODEL` | `qwen3.5:9b` | Model that writes the answers |
| `EMBED_MODEL` | `nomic-embed-text` | Model that creates embeddings |
| `CHUNK_SIZE` | `200` | Words per chunk |
| `OVERLAP` | `40` | Words shared between neighbouring chunks |
| `TOP_K` | `4` | Chunks sent to the model per question |

## Roadmap

- [ ] Sidebar controls for `TOP_K` and chunk size
- [ ] Save embeddings to disk so documents don't need re-processing
- [ ] Vector database (ChromaDB)
- [ ] Evaluation set to measure answer accuracy
- [ ] OCR support for scanned PDFs
