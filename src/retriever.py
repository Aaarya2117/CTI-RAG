"""
retriever.py
Given a user query, retrieve the top-k most relevant chunks from the FAISS index.
This is the R in RAG — written from scratch, no LangChain.
"""

import numpy as np
import faiss
from typing import List, Dict
from sentence_transformers import SentenceTransformer


class Retriever:
    """
    Retrieves relevant document chunks for a query using FAISS.
    Uses L2-normalized embeddings so cosine similarity = inner product.
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

    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        k = top_k or self.top_k
        query_emb = self.embed_query(query)

        scores, indices = self.index.search(query_emb, k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:
                continue
            results.append({
                **self.chunks[idx],
                "score": float(score),
                "rank": rank + 1,
            })

        return results

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
        results = retriever.retrieve(query)
        print(f"\nQuery: {query}")
        for r in results:
            print(f"  [{r['rank']}] score={r['score']:.4f} — {r['text'][:100]}...")
