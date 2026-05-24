# RAG Document Q&A

A minimal Retrieval-Augmented Generation (RAG) pipeline built from scratch.
Upload any PDF or TXT → ask questions → get answers grounded in the document.

## Architecture

```text
Document (.pdf/.txt)
    │
    ▼
[Ingest + Chunk]  →  ~300-word overlapping windows
    │
    ▼
[Embed]           →  sentence-transformers/all-MiniLM-L6-v2 (384-dim)
    │
    ▼
[FAISS Index]     →  IndexFlatIP (cosine similarity)
    │
    ▼
User Query → [Embed Query] → [Top-K Retrieval] → [Prompt] → [Gemma / Flan-T5]
                                                                      │
                                                                      ▼
                                                                   Answer
```

## Tech Stack

| Component    | Library                             |
|--------------|-------------------------------------|
| Ingestion    | PyMuPDF                             |
| Embeddings   | sentence-transformers (MiniLM-L6)   |
| Vector Store | FAISS (IndexFlatIP)                 |
| Generation   | OpenRouter Gemma 4 31B or local Flan-T5 |
| UI           | Gradio                              |

## Evaluation Results

| Metric                   | Score |
|--------------------------|-------|
| Retrieval Accuracy       | —     |
| Avg Keyword Match Score  | —     |

> Fill in from `results/eval_results.json` after running evaluate.py

## How to Run

```bash
pip install -r requirements.txt
# Add a document to data/documents/
export OPENROUTER_API_KEY="your_openrouter_key"
python src/pipeline.py    # terminal Q&A
python src/app.py         # Gradio UI at localhost:7860
```

The default generator is OpenRouter's free `google/gemma-4-31b-it:free` model.
This project does not require an OpenAI API key. Do not commit real API keys;
put `OPENROUTER_API_KEY` in your shell environment or in a local `.env` file.
Free hosted models can be rate-limited during busy periods; by default the app
falls back to `google/flan-t5-small` locally if OpenRouter returns a rate-limit
error.

For a key-free local fallback, set this in `config.yaml`:

```yaml
generation_provider: "local"
generation_model: "google/flan-t5-small"
```

The first local run downloads the embedding and generator models from
Hugging Face, then reuses the local cache.

## Optional Hosted Generation

To use the Hugging Face Inference API instead of OpenRouter:

1. Set `generation_provider: "hf_inference_api"` in `config.yaml`.
2. Export a token before running the app:

```bash
export HF_API_TOKEN="your_huggingface_token"
python src/app.py
```

Do not commit real API keys. `.env` is ignored; `.env.example` is included only
as a template.

## Project Structure

```text
rag-document-qa/
├── config.yaml
├── requirements.txt
├── src/
│   ├── ingest.py       # PDF/TXT loading + chunking
│   ├── embeddings.py   # sentence-transformers + FAISS index build
│   ├── retriever.py    # cosine similarity retrieval
│   ├── generator.py    # OpenRouter / Flan-T5 local / HF Inference API
│   ├── pipeline.py     # end-to-end wiring
│   ├── evaluate.py     # retrieval accuracy + keyword match eval
│   └── app.py          # Gradio web UI
├── data/
│   ├── documents/      # put your PDFs/TXTs here
│   ├── index/          # FAISS index + chunks (auto-generated)
│   └── eval_qa.json    # your hand-crafted Q&A pairs
└── results/
    └── eval_results.json
```
