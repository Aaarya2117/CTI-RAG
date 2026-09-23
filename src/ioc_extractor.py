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
