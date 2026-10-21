# 📄 Chat With Your Documents

A local AI assistant that answers questions about your PDFs, with page-level citations. Everything runs on your own computer, so your documents never leave your machine.

Built with **RAG (Retrieval-Augmented Generation)**: the app first finds the most relevant parts of your documents, then asks a language model to answer using only those parts.

<!-- Add a screenshot: save it as screenshot.png in this folder, then remove the arrows around the next line -->
<!-- ![App screenshot](screenshot.png) -->

## Results

Tested with [`evaluate.py`](evaluate.py) on 18 questions about a 6-page sample handbook ([`sample_docs/`](sample_docs)): 15 questions answered in the document, and 3 that are not, where the correct behaviour is to say so instead of guessing.

| TOP_K | Search found the right page | Answer correct | Avg time per answer |
|---|---|---|---|
| 1 | 15/15 (100%) | 18/18 (100%) | 9.0 s |
| 4 | 15/15 (100%) | 18/18 (100%) | 11.1 s |

Setup: `qwen3.5:9b` with thinking off, `nomic-embed-text` embeddings, chunk size 50 words (23 chunks).

**Takeaways**
- The app refused all 3 questions that weren't in the document, so it doesn't make up answers.
- For a short, focused document, retrieving more chunks (TOP_K 4) made answers 23% slower without improving accuracy, so TOP_K 1 is the better setting here.
- The sample questions are direct, so a perfect score is expected. Longer documents and questions worded differently from the text are the next test.

## Features

- Upload one or more PDFs and ask questions in plain language
- Answers stream in word by word and cite the source file and page
- Says so when the answer isn't in your documents
- Shows the retrieved passages so you can check where each answer came from
- Sidebar sliders for chunk size and TOP_K
- Processed documents and settings are saved, so they're still there after a restart
- Built-in accuracy test
- Fully offline: uses local models through [Ollama](https://ollama.com)

## How it works

1. **Extract:** pull the text from each PDF page with PyMuPDF
2. **Chunk:** split the text into overlapping pieces (size set in the sidebar, 20% overlap)
3. **Embed:** turn each chunk into a vector with `nomic-embed-text`
4. **Retrieve:** embed the question and find the most similar chunks with cosine similarity
5. **Generate:** send those chunks and the question to `qwen3.5:9b` with instructions to answer only from that context

## Project structure

| File | What it does |
|---|---|
| `rag.py` | The AI logic: reading, chunking, embedding, search, saving, answering |
| `app.py` | The Streamlit web interface |
| `evaluate.py` | Accuracy test: scores search and answers against known questions |
| `eval_questions.json` | Test questions with expected pages and keywords |
| `sample_docs/` | A sample PDF (a fictional company handbook) for testing |

## Tech stack

Python · Streamlit · Ollama · Qwen 3.5 · nomic-embed-text · PyMuPDF · NumPy

## Getting started

**Requirements:** Python 3.10+, [Ollama](https://ollama.com), and about 8 GB of free RAM.

```bash
# 1. Download the models
ollama pull qwen3.5:9b
ollama pull nomic-embed-text

# 2. Clone the project and install the packages
git clone https://github.com/bahramshafiq11/rag-app.git
cd rag-app
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt

# 3. Run it
streamlit run app.py
```

Then open http://localhost:8501, upload a PDF (try `sample_docs/nimbus_employee_handbook.pdf`), click **Process documents**, and ask a question.

## Running the accuracy test

Process a PDF in the app first, then:

```bash
python evaluate.py --retrieval-only   # test only the search (fast)
python evaluate.py                    # full test with the TOP_K from the app
python evaluate.py --top-k 1 4 8      # compare several TOP_K values
```

Every answer is saved to `eval_results.csv`. To test your own PDF, replace the questions in `eval_questions.json`.

## Configuration

Model settings are at the top of `rag.py`:

| Setting | Default | What it does |
|---|---|---|
| `LLM_MODEL` | `qwen3.5:9b` | Model that writes the answers |
| `EMBED_MODEL` | `nomic-embed-text` | Model that creates embeddings |
| `DEFAULT_CHUNK_SIZE` | `200` | Starting value of the chunk size slider |
| `DEFAULT_TOP_K` | `4` | Starting value of the TOP_K slider |
| `OVERLAP_RATIO` | `0.2` | Share of words neighbouring chunks have in common |

## Roadmap

- [x] Sidebar controls for TOP_K and chunk size
- [x] Save processed documents between restarts
- [x] Streaming answers
- [x] Evaluation set to measure accuracy
- [ ] Harder evaluation: longer documents and reworded questions
- [ ] Vector database (ChromaDB) for large document collections
- [ ] OCR support for scanned PDFs
- [ ] Online deployment