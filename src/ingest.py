"""
ingest.py
Load and chunk PDF, TXT, JSON CVE, or URL-based documents into overlapping text windows.
This is the first stage of the CTI-RAG pipeline.
"""

import re
import json
import pickle
import datetime
import tempfile
import urllib.request
from pathlib import Path
from typing import List, Dict

try:
    from config_utils import PROJECT_ROOT, load_config
except ImportError:
    from .config_utils import PROJECT_ROOT, load_config


def load_pdf(file_path: str) -> str:
    """Extract raw text from a PDF using PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF not installed. Run: pip install PyMuPDF")

    doc = fitz.open(file_path)
    full_text = ""
    for page_num, page in enumerate(doc):
        page_text = page.get_text("text")
        full_text += f"\n[Page {page_num + 1}]\n{page_text}"
    page_count = len(doc)
    doc.close()
    print(f"Loaded PDF: {file_path} ({page_count} pages, {len(full_text)} characters)")
    return full_text


def load_txt(file_path: str) -> str:
    """Load raw text from a .txt file."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    print(f"Loaded TXT: {file_path} ({len(text)} characters)")
    return text


def load_from_url(url: str) -> str:
    """Download a PDF or HTML document from a URL and extract its text."""
    with urllib.request.urlopen(url, timeout=15) as response:
        content_type = response.headers.get("Content-Type", "")
        data = response.read()

    if "pdf" in content_type or url.lower().endswith(".pdf"):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        return load_pdf(tmp_path)
    else:
        # Treat as plain text / HTML — strip tags if needed
        text = data.decode("utf-8", errors="replace")
        text = re.sub(r"<[^>]+>", " ", text)  # strip HTML tags
        return text


def load_json_cve(file_path: str) -> str:
    """Load and flatten a NVD CVE JSON feed into plain text for chunking."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lines = []
    # NVD CVE 2.0 feed structure
    for item in data.get("vulnerabilities", []):
        cve = item.get("cve", {})
        cve_id = cve.get("id", "")
        descriptions = cve.get("descriptions", [])
        desc = next((d["value"] for d in descriptions if d["lang"] == "en"), "")
        published = cve.get("published", "")
        severity = ""
        try:
            severity = cve["metrics"]["cvssMetricV31"][0]["cvssData"]["baseSeverity"]
        except (KeyError, IndexError):
            pass
        lines.append(f"CVE ID: {cve_id} | Published: {published} | Severity: {severity}\n{desc}")

    return "\n\n".join(lines)


def load_document(file_path: str) -> str:
    """Auto-detect file type and load accordingly. Now also handles URLs."""
    if file_path.startswith("http://") or file_path.startswith("https://"):
        return load_from_url(file_path)
    path = Path(file_path)
    if path.suffix.lower() == ".pdf":
        return load_pdf(file_path)
    elif path.suffix.lower() == ".txt":
        return load_txt(file_path)
    elif path.suffix.lower() == ".json":
        return load_json_cve(file_path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}. Use .pdf, .txt, .json, or a URL.")


def clean_text(text: str) -> str:
    """
    Basic text cleaning:
    - Collapse multiple whitespace/newlines
    - Remove page headers/footers artifacts
    - Normalize unicode spaces
    """
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\u00a0', ' ', text)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    return text.strip()


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[Dict]:
    """
    Split text into overlapping word-based chunks.
    Each chunk is a dict: {chunk_id, text, word_start, word_end, word_count}
    """
    words = text.split()
    total_words = len(words)

    if total_words == 0:
        return []

    chunks = []
    chunk_id = 0
    start = 0

    while start < total_words:
        end = min(start + chunk_size, total_words)
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words)

        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text_str,
            "word_start": start,
            "word_end": end,
            "word_count": len(chunk_words),
        })

        chunk_id += 1
        next_start = start + chunk_size - chunk_overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start

        if end == total_words:
            break

    print(f"Chunking complete: {len(chunks)} chunks from {total_words} words")
    print(f"  Chunk size: {chunk_size} words | Overlap: {chunk_overlap} words")
    return chunks


def ingest_document(file_path: str, cfg: dict) -> List[Dict]:
    """Full ingestion pipeline: load → clean → chunk."""
    raw_text = load_document(file_path)
    clean = clean_text(raw_text)
    chunks = chunk_text(clean, cfg["chunk_size"], cfg["chunk_overlap"])

    # Inject source metadata into each chunk
    source_name = file_path if file_path.startswith("http") else Path(file_path).name
    for chunk in chunks:
        chunk["source"] = source_name
        chunk["ingested_at"] = datetime.datetime.utcnow().isoformat()

    Path(cfg["chunks_save_path"]).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg["chunks_save_path"], "wb") as f:
        pickle.dump(chunks, f)
    print(f"Chunks saved to: {cfg['chunks_save_path']}")

    return chunks


def load_chunks(chunks_path: str) -> List[Dict]:
    """Load previously saved chunks from disk."""
    with open(chunks_path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    import sys
    cfg = load_config()

    test_files = list((PROJECT_ROOT / "data" / "documents").glob("*.*"))
    if not test_files:
        print("No documents found in data/documents/. Add a .pdf or .txt file first.")
        sys.exit(0)

    chunks = ingest_document(str(test_files[0]), cfg)
    if chunks:
        print(f"\nSample chunk 0:\n{chunks[0]['text'][:300]}...")
        print(f"\nSample chunk -1:\n{chunks[-1]['text'][:300]}...")
