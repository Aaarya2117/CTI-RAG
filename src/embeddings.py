"""
embeddings.py
Embed text chunks using sentence-transformers and store in a FAISS index.
"""

import pickle
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer

try:
    from config_utils import load_config
except ImportError:
    from .config_utils import load_config


def load_embedding_model(cfg: dict):
    """Load the embedding model (either API-based or local)."""
    model_name = cfg["embedding_model"]
    provider = cfg.get("embedding_provider", "local").lower()
    print(f"Loading embedding model: {model_name} via {provider}")
    
    if provider == "gemini_api":
        import os
        token = cfg.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY")
        if not token:
            raise ValueError("GEMINI_API_KEY must be set in .env to use the Gemini API for embeddings.")
        return GeminiAPIEmbedder(model_name, token)
        
    # Local fallback
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


class GeminiAPIEmbedder:
    """Wraps Google Gemini API for embeddings."""
    def __init__(self, model_name: str, gemini_key: str):
        from google import genai
        self.client = genai.Client(api_key=gemini_key)
        # Strip "models/" if present, as the new SDK expects 'text-embedding-004'
        if model_name.startswith("models/"):
            model_name = model_name[7:]
        self.model_name = model_name
        
    def encode(self, texts, batch_size=None, show_progress_bar=None, normalize_embeddings=True, convert_to_numpy=True):
        import numpy as np
        
        embeddings = []
        # Gemini embed_content treats a list of strings as ONE concatenated
        # content and returns a single embedding.  We must call it once per
        # text to get one embedding per chunk.
        for i, text in enumerate(texts):
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=text,
            )
            embeddings.append(result.embeddings[0].values)
            
        embeddings = np.array(embeddings)
        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.where(norms == 0, 1, norms)
            
        return embeddings


def embed_chunks(
    chunks: List[Dict],
    model: SentenceTransformer,
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings.astype("float32")


def build_faiss_index(embeddings: np.ndarray, embedding_dim: int) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings)
    print(f"FAISS index built: {index.ntotal} vectors, dim={embedding_dim}")
    return index


def save_index(index: faiss.IndexFlatIP, index_path: str):
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)
    print(f"FAISS index saved: {index_path}")


def load_index(index_path: str) -> faiss.IndexFlatIP:
    index = faiss.read_index(index_path)
    print(f"FAISS index loaded: {index.ntotal} vectors")
    return index


def build_and_save_index(chunks: List[Dict], cfg: dict):
    model = load_embedding_model(cfg)
    embeddings = embed_chunks(chunks, model)
    index = build_faiss_index(embeddings, cfg["embedding_dim"])
    save_index(index, cfg["index_save_path"])
    return model, index


if __name__ == "__main__":
    from ingest import load_config, load_chunks

    cfg = load_config()
    chunks = load_chunks(cfg["chunks_save_path"])
    model, index = build_and_save_index(chunks, cfg)

    test_query = "What is the main topic of this document?"
    q_emb = model.encode([test_query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    scores, indices = index.search(q_emb, 3)
    print(f"\nTest retrieval for: '{test_query}'")
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        print(f"  Rank {rank+1} (score={score:.4f}): {chunks[idx]['text'][:120]}...")
