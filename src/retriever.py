"""
retriever.py
Given a user query, retrieve the top-k most relevant chunks from the FAISS index.
This is the R in RAG — written from scratch, no LangChain.
"""

import os
os.environ.setdefault("USE_TF", "0")
import numpy as np

import faiss
from typing import List, Dict, Optional, Any
from sentence_transformers import SentenceTransformer

try:
    from laya_layer import route_query
except ImportError:
    try:
        from .laya_layer import route_query
    except ImportError:
        route_query = None


INTENT_TO_CATEGORY = {
    "cve_lookup": "vuln_detail",
    "ttp_mapping": "ttp",
    "ioc_lookup": "ioc_list",
    "mitigation": "mitigation",
    "actor_profile": "actor_profile",
}


def _parse_intent(intent_val: Any) -> str:
    """Extract string intent name from Laya's choice output."""
    if isinstance(intent_val, str):
        return intent_val.lower().strip()
    if isinstance(intent_val, dict):
        for k in ("choice", "answer", "intent", "value"):
            if k in intent_val and isinstance(intent_val[k], str):
                return intent_val[k].lower().strip()
    return "general"


def _is_in_scope(in_scope_val: Any) -> bool:
    """Evaluate whether the query was classified as in-scope CTI."""
    if in_scope_val is None:
        return True
    if isinstance(in_scope_val, bool):
        return in_scope_val
    if isinstance(in_scope_val, (int, float)):
        return in_scope_val >= 0.5
    if isinstance(in_scope_val, str):
        return in_scope_val.lower().strip() not in ("no", "false", "0", "out_of_scope")
    if isinstance(in_scope_val, dict):
        if "choice" in in_scope_val:
            val = in_scope_val["choice"]
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.lower().strip() in ("yes", "true", "1")
        if "prob" in in_scope_val:
            return float(in_scope_val["prob"]) >= 0.5
        if "score" in in_scope_val:
            return float(in_scope_val["score"]) >= 0.5
        if "answer" in in_scope_val:
            ans = in_scope_val["answer"]
            if isinstance(ans, bool):
                return ans
            if isinstance(ans, str):
                return ans.lower().strip() in ("yes", "true", "1")
    return True


class Retriever:
    """
    Retrieves relevant document chunks for a query using FAISS.
    Uses L2-normalized embeddings so cosine similarity = inner product.
    Supports Laya-guided intent boosting and query scope routing.
    """

    def __init__(
        self,
        index: faiss.IndexFlatIP,
        chunks: List[Dict],
        model: SentenceTransformer,
        top_k: int = 4,
    ):
        self.index = index
        self.chunks = chunks
        self.model = model
        self.top_k = top_k

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        routing: Optional[Dict[str, Any]] = None,
        boost_weight: float = 0.15,
    ) -> List[Dict]:
        k = top_k or self.top_k
        query_emb = self.embed_query(query)

        # If routing is provided with a specific intent, search more candidates to allow reranking/boosting
        intent = _parse_intent(routing.get("intent")) if routing else None
        target_category = INTENT_TO_CATEGORY.get(intent) if intent else None

        fetch_k = min(len(self.chunks), max(k * 3, k)) if (target_category and len(self.chunks) > 0) else k
        fetch_k = max(fetch_k, 1)

        scores, indices = self.index.search(query_emb, fetch_k)

        candidates = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1 or idx >= len(self.chunks):
                continue
            chunk_data = dict(self.chunks[idx])
            base_score = float(score)
            boost = 0.0

            if target_category:
                laya_tags = chunk_data.get("laya_tags") or {}
                chunk_cat = laya_tags.get("category")
                if isinstance(chunk_cat, dict):
                    chunk_cat = chunk_cat.get("choice") or chunk_cat.get("answer")
                if chunk_cat and str(chunk_cat).lower() == target_category.lower():
                    boost += boost_weight

                # Extra boost if looking for CVE and chunk references an active CVE
                if intent == "cve_lookup":
                    ref_cve = laya_tags.get("references_active_cve")
                    if ref_cve is True or (isinstance(ref_cve, dict) and ref_cve.get("choice") is True):
                        boost += 0.10

            final_score = base_score + boost
            chunk_data["score"] = base_score
            chunk_data["boosted_score"] = float(final_score)
            chunk_data["boost"] = float(boost)
            candidates.append(chunk_data)

        # Sort by boosted_score descending if boosting was applied, else keep base score order
        if target_category:
            candidates.sort(key=lambda x: x["boosted_score"], reverse=True)

        selected = candidates[:k]
        for rank, c in enumerate(selected):
            c["rank"] = rank + 1

        return selected

    def route_and_retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        boost_weight: float = 0.15,
    ) -> Dict[str, Any]:
        """
        Route query via Laya, check scope guardrail, and retrieve boosted chunks.

        Returns:
            dict containing:
              - "query": input query
              - "routing": Laya routing output (intent, in_scope)
              - "in_scope": bool
              - "refusal": None if in_scope, or refusal string if out_of_scope
              - "chunks": list of retrieved chunks
              - "retrieved_chunks": alias for chunks
        """
        routing = None
        if route_query is not None:
            try:
                routing = route_query(query)
            except Exception as exc:
                print(f"Warning: Laya route_query failed: {exc}")
                routing = {"intent": "general", "in_scope": True}
        else:
            routing = {"intent": "general", "in_scope": True}

        in_scope = _is_in_scope(routing.get("in_scope"))
        if not in_scope:
            return {
                "query": query,
                "routing": routing,
                "in_scope": False,
                "refusal": (
                    "Query refused: input was classified as out-of-scope or an adversarial "
                    "instruction override attempt."
                ),
                "chunks": [],
                "retrieved_chunks": [],
            }

        chunks = self.retrieve(query, top_k=top_k, routing=routing, boost_weight=boost_weight)
        return {
            "query": query,
            "routing": routing,
            "in_scope": True,
            "refusal": None,
            "chunks": chunks,
            "retrieved_chunks": chunks,
        }

    def format_context(self, retrieved_chunks: List[Dict]) -> str:
        parts = []
        for chunk in retrieved_chunks:
            parts.append(chunk["text"])
        return "\n\n---\n\n".join(parts)


if __name__ == "__main__":
    from ingest import load_config, load_chunks
    from embeddings import load_embedding_model, load_index

    cfg = load_config()
    chunks = load_chunks(cfg["chunks_save_path"])
    model = load_embedding_model(cfg)
    index = load_index(cfg["index_save_path"])

    retriever = Retriever(index, chunks, model, top_k=cfg["top_k"])

    test_queries = [
        "What is the main argument of the document?",
        "What methods are described?",
        "What are the conclusions?",
    ]

    for query in test_queries:
        routed_result = retriever.route_and_retrieve(query)
        print(f"\nQuery: {query}")
        print(f"  Routing: {routed_result.get('routing')}")
        print(f"  In scope: {routed_result.get('in_scope')}")
        for r in routed_result["chunks"]:
            boost_info = f" (boosted={r.get('boosted_score', r['score']):.4f})" if 'boosted_score' in r else ""
            print(f"  [{r['rank']}] score={r['score']:.4f}{boost_info} — {r['text'][:100]}...")

