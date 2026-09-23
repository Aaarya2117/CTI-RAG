"""
seed_corpus.py
Auto-download a few public CISA cybersecurity advisories (PDF) to data/documents/.
Run once to bootstrap the CTI-RAG corpus with real threat intelligence.

Usage:
    python scripts/seed_corpus.py
"""

import os
import sys
import urllib.request

# Resolve project root relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DOCS_DIR = os.path.join(PROJECT_ROOT, "data", "documents")

# Publicly available cyber threat intelligence documents (PDF)
SEED_URLS = [
    {
        "url": "https://attack.mitre.org/docs/ATTACK_Design_and_Philosophy_March_2020.pdf",
        "filename": "mitre_attack_design_philosophy.pdf",
        "description": "MITRE ATT&CK: Design and Philosophy (March 2020)",
    },
    {
        "url": "https://nvlpubs.nist.gov/nistpubs/specialpublications/nist.sp.800-61r3.pdf",
        "filename": "nist_sp800-61r3_incident_response.pdf",
        "description": "NIST SP 800-61r3: Incident Response Recommendations and Considerations",
    },
    {
        "url": "https://www.ic3.gov/AnnualReport/Reports/2023_IC3Report.pdf",
        "filename": "fbi_ic3_2023_internet_crime_report.pdf",
        "description": "FBI IC3 2023 Internet Crime Report",
    },
]


def download_file(url: str, dest: str, description: str) -> bool:
    """Download a file from a URL to a local path. Returns True on success."""
    if os.path.exists(dest):
        print(f"  ✓ Already exists: {os.path.basename(dest)}")
        return True

    print(f"  ↓ Downloading: {description}")
    print(f"    URL: {url}")
    try:
        urllib.request.urlretrieve(url, dest)
        size_kb = os.path.getsize(dest) / 1024
        print(f"    ✅ Saved ({size_kb:.0f} KB): {os.path.basename(dest)}")
        return True
    except Exception as e:
        print(f"    ❌ Failed: {e}")
        # Remove partial downloads
        if os.path.exists(dest):
            os.remove(dest)
        return False


def main():
    os.makedirs(DOCS_DIR, exist_ok=True)

    print("=" * 60)
    print("CTI-RAG Seed Corpus Downloader")
    print("=" * 60)
    print(f"Target directory: {DOCS_DIR}\n")

    success = 0
    for entry in SEED_URLS:
        dest_path = os.path.join(DOCS_DIR, entry["filename"])
        if download_file(entry["url"], dest_path, entry["description"]):
            success += 1

    print(f"\n{'=' * 60}")
    print(f"Downloaded {success}/{len(SEED_URLS)} documents.")
    if success > 0:
        print("Run `python src/pipeline.py` to index them.")
    else:
        print("No documents were downloaded. Check your internet connection.")
        print("You can also manually place PDF/TXT files in data/documents/.")
    print("=" * 60)


if __name__ == "__main__":
    main()
