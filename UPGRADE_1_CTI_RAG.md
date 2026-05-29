# Upgrade 1: Threat Intelligence Analyst RAG (CTI-RAG)

## Goal

Transform the existing RAG Document Q&A pipeline into a **Cyber Threat Intelligence (CTI) Q&A system**. Analysts should be able to feed it real-world threat reports and query them with security-specific natural language questions.

---

## Project Rename

- Rename the project to `CTI-RAG` or `ThreatDoc-QA`
- Update `README.md` title, description, and architecture section to reflect the security use case
- Update the Gradio UI title in `src/app.py` to `🛡️ CTI Threat Intelligence Q&A`

---

## Change 1 — Extend `src/ingest.py`: URL-based document fetching

Add a new function `load_from_url(url: str) -> str` that downloads a PDF or HTML page from a URL and extracts its text. This allows the corpus to be built from live sources like CISA advisories or NVD.

```python
# Add to ingest.py

import urllib.request
import tempfile

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
        import re
        text = re.sub(r"<[^>]+>", " ", text)  # strip HTML tags
        return text


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
```

Also add a JSON loader for CVE/NVD feeds:

```python
def load_json_cve(file_path: str) -> str:
    """Load and flatten a NVD CVE JSON feed into plain text for chunking."""
    import json
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
```

---

## Change 2 — Add metadata to chunks in `src/ingest.py`

Extend the `chunk_text()` output and `ingest_document()` call to include source metadata in every chunk dict. This powers source attribution in the UI.

In `ingest_document()`, pass the file path into `chunk_text()` and store it:

```python
def ingest_document(file_path: str, cfg: dict) -> List[Dict]:
    """Full ingestion pipeline: load → clean → chunk."""
    raw_text = load_document(file_path)
    clean = clean_text(raw_text)
    chunks = chunk_text(clean, cfg["chunk_size"], cfg["chunk_overlap"])

    # Inject source metadata into each chunk
    source_name = file_path if file_path.startswith("http") else Path(file_path).name
    import datetime
    for chunk in chunks:
        chunk["source"] = source_name
        chunk["ingested_at"] = datetime.datetime.utcnow().isoformat()

    Path(cfg["chunks_save_path"]).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg["chunks_save_path"], "wb") as f:
        pickle.dump(chunks, f)
    print(f"Chunks saved to: {cfg['chunks_save_path']}")
    return chunks
```

---

## Change 3 — Update `PROMPT_TEMPLATE` in `src/generator.py`

Replace the generic prompt with a security analyst–oriented one that instructs the model to reference MITRE ATT&CK IDs and threat actor names when found in context.

```python
PROMPT_TEMPLATE = """You are a Cyber Threat Intelligence (CTI) analyst assistant.
Answer the analyst's question using ONLY the provided threat intelligence context below.
If the answer is not found in the context, say "Insufficient intelligence in the loaded documents to answer this."

When relevant, reference:
- MITRE ATT&CK tactic and technique IDs (e.g., T1059 - Command and Scripting Interpreter)
- Threat actor names or groups mentioned in the context
- CVE identifiers if present
- IOCs (IPs, domains, file hashes) if mentioned

Context:
{context}

Analyst Question: {question}

Intelligence Assessment:"""
```

---

## Change 4 — Add `src/ioc_extractor.py` (new file)

Create a new module that post-processes the LLM answer and retrieves structured IOCs from it using regex. Call this from `pipeline.py`'s `ask()` method.

```python
"""
ioc_extractor.py
Extract Indicators of Compromise (IOCs) from generated answers and retrieved chunks.
Looks for: CVE IDs, IP addresses, SHA256/MD5 hashes, domains, MITRE ATT&CK technique IDs.
"""

import re
from typing import Dict, List


PATTERNS = {
    "cve_ids":      r"CVE-\d{4}-\d{4,7}",
    "ipv4":         r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    "md5":          r"\b[a-fA-F0-9]{32}\b",
    "sha256":       r"\b[a-fA-F0-9]{64}\b",
    "mitre_attack": r"T\d{4}(?:\.\d{3})?",
    "domains":      r"\b(?:[a-zA-Z0-9\-]+\.)+(?:com|net|org|io|gov|mil|ru|cn|info|xyz)\b",
}


def extract_iocs(text: str) -> Dict[str, List[str]]:
    """Return a dict of IOC type → list of unique matches found in text."""
    results = {}
    for ioc_type, pattern in PATTERNS.items():
        matches = list(set(re.findall(pattern, text)))
        if matches:
            results[ioc_type] = matches
    return results


def extract_from_result(result: dict) -> Dict[str, List[str]]:
    """
    Run IOC extraction over the answer + all retrieved chunk texts.
    Merges and deduplicates across all sources.
    """
    combined_text = result.get("answer", "")
    for chunk in result.get("retrieved_chunks", []):
        combined_text += "\n" + chunk.get("text", "")

    return extract_iocs(combined_text)
```

---

## Change 5 — Update `src/pipeline.py`: call IOC extractor in `ask()`

In the `ask()` method, import and call `extract_from_result` and include the IOCs in the returned dict.

```python
# Add at top of pipeline.py
try:
    from ioc_extractor import extract_from_result
except ImportError:
    from .ioc_extractor import extract_from_result

# In RAGPipeline.ask(), update the return dict:
    iocs = extract_from_result({
        "answer": answer,
        "retrieved_chunks": retrieved,
    })

    return {
        "question": question,
        "answer": answer,
        "retrieved_chunks": retrieved,
        "context_used": context,
        "iocs": iocs,          # <-- new
    }
```

---

## Change 6 — Update `src/app.py`: surface IOCs and source attribution in UI

Update the Gradio interface to:
1. Show extracted IOCs as a separate output panel
2. Show `source` and `ingested_at` in the retrieved chunks section
3. Accept a URL input in addition to file upload

```python
# Replace the answer_question function and UI blocks in app.py

def answer_question(question: str, show_sources: bool) -> tuple[str, str, str]:
    if not question.strip():
        return "Please enter a question.", "", ""
    if not pipeline._is_ready:
        return "Please upload and index a document first.", "", ""

    try:
        result = pipeline.ask(question, verbose=False)
        answer = result["answer"]

        # Format IOCs
        ioc_text = ""
        iocs = result.get("iocs", {})
        if iocs:
            parts = []
            for ioc_type, values in iocs.items():
                parts.append(f"**{ioc_type.upper()}**: {', '.join(values)}")
            ioc_text = "\n\n".join(parts)
        else:
            ioc_text = "_No IOCs detected in this response._"

        # Format sources
        sources = ""
        if show_sources:
            sources_parts = []
            for chunk in result["retrieved_chunks"]:
                src = chunk.get("source", "unknown")
                score = chunk.get("score", 0)
                sources_parts.append(
                    f"**Source: {src}** (relevance: {score:.3f})\n"
                    f"{chunk['text'][:300]}..."
                )
            sources = "\n\n---\n\n".join(sources_parts)

        return answer, ioc_text, sources
    except Exception as e:
        return f"Error: {str(e)}", "", ""


# Also add a URL indexing function alongside process_document:
def index_from_url(url: str) -> str:
    if not url.strip():
        return "No URL provided."
    try:
        pipeline.index_document(url.strip())
        chunk_count = len(pipeline.retriever.chunks)
        return f"✅ Indexed from URL! {chunk_count} chunks created."
    except Exception as e:
        return f"❌ Error: {str(e)}"
```

Update the Gradio `Blocks` layout to add:
- A `gr.Textbox` for URL input + an "Index from URL" button
- A third output `gr.Markdown` panel labelled "Extracted IOCs"
- Update `ask_btn.click` outputs to include the IOC panel

---

## Change 7 — Update `data/documents/` with real CTI sources

Replace placeholder documents with real public threat intelligence. Suggested seed documents (all freely available as PDFs):

| Source | What to download |
|--------|-----------------|
| CISA | Advisories from https://www.cisa.gov/news-events/cybersecurity-advisories |
| Mandiant / Google TAG | Public APT reports (search "Mandiant APT PDF") |
| MITRE ATT&CK | Technique pages exported as PDF |
| NVD | CVE JSON feed from https://nvd.nist.gov/vuln/data-feeds |

Add a script `scripts/seed_corpus.py` that auto-downloads a few CISA advisories to `data/documents/` on first run.

---

## Change 8 — Update `README.md`

Replace the generic README with:
- New project name: **CTI-RAG: Threat Intelligence Q&A**
- Description: "A RAG-powered assistant for querying cybersecurity threat reports, CISA advisories, and CVE feeds using natural language."
- Updated architecture diagram showing CTI-specific data flow
- Example queries section:
  - `"What TTPs does APT29 use for lateral movement?"`
  - `"Which CVEs in this advisory affect Windows Server?"`
  - `"What IOCs are associated with the Lazarus Group?"`
- Fill in evaluation metrics after running `evaluate.py` on a security-specific Q&A set
- Add a `data/eval_qa.json` with at least 5 CTI-specific Q&A pairs for the eval harness

---

## New file structure after upgrade

```text
CTI-RAG/
├── config.yaml
├── requirements.txt
├── scripts/
│   └── seed_corpus.py          # auto-downloads CISA advisories
├── src/
│   ├── ingest.py               # + URL loader, JSON CVE loader, source metadata
│   ├── ioc_extractor.py        # NEW — regex IOC extraction
│   ├── embeddings.py           # unchanged
│   ├── retriever.py            # unchanged
│   ├── generator.py            # updated PROMPT_TEMPLATE
│   ├── pipeline.py             # ask() now returns iocs dict
│   ├── evaluate.py             # unchanged (update eval_qa.json)
│   └── app.py                  # updated UI with IOC panel + URL input
├── data/
│   ├── documents/              # CISA advisories, APT reports, CVE feeds
│   ├── index/
│   └── eval_qa.json            # CTI-specific Q&A pairs
└── results/
    └── eval_results.json
```

---

## Summary of all changes

| File | Action |
|------|--------|
| `src/ingest.py` | Add `load_from_url()`, `load_json_cve()`, source metadata injection |
| `src/generator.py` | Replace `PROMPT_TEMPLATE` with CTI analyst framing |
| `src/ioc_extractor.py` | **Create new** — regex IOC extraction module |
| `src/pipeline.py` | Call `extract_from_result()` in `ask()`, add `iocs` to return dict |
| `src/app.py` | Add URL input, IOC output panel, source attribution in chunks display |
| `README.md` | Full rewrite for CTI use case with example queries |
| `data/documents/` | Replace with real CISA/Mandiant/NVD documents |
| `scripts/seed_corpus.py` | **Create new** — auto-download seed CTI documents |
| `data/eval_qa.json` | Replace template with 5+ CTI-specific Q&A pairs |
