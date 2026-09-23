"""
embeddings.py
Embed text chunks using sentence-transformers or HuggingFace embeddings and store in a FAISS index.
Preserves GeminiAPIEmbedder as an optional fallback.
"""

import pickle
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Union, Any, Optional

try:
    from config_utils import load_config
except ImportError:
    from .config_utils import load_config


class LocalEmbedder:
    """
    Wraps local sentence-transformers / HuggingFace embeddings for both
    SentenceTransformer (.encode) and LangChain (.embed_documents / .embed_query) compatibility.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", prefer_langchain: bool = False):
        self.model_name = model_name
        self._st_model = None
        self._hf_embeddings = None
        self._dim = None

        if prefer_langchain:
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
                self._hf_embeddings = HuggingFaceEmbeddings(model_name=model_name)
                if hasattr(self._hf_embeddings, "client") and hasattr(self._hf_embeddings.client, "get_sentence_embedding_dimension"):
                    self._dim = self._hf_embeddings.client.get_sentence_embedding_dimension()
            except ImportError:
                pass

        if self._hf_embeddings is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._st_model = SentenceTransformer(model_name)
                if hasattr(self._st_model, "get_sentence_embedding_dimension"):
                    self._dim = self._st_model.get_sentence_embedding_dimension()
            except ImportError:
                try:
                    from langchain_huggingface import HuggingFaceEmbeddings
                    self._hf_embeddings = HuggingFaceEmbeddings(model_name=model_name)
                    if hasattr(self._hf_embeddings, "client") and hasattr(self._hf_embeddings.client, "get_sentence_embedding_dimension"):
                        self._dim = self._hf_embeddings.client.get_sentence_embedding_dimension()
                except ImportError:
                    raise ImportError(
                        "Neither 'sentence_transformers' nor 'langchain_huggingface' is installed. "
                        "Please install them via: pip install sentence-transformers langchain-huggingface"
                    )

    def get_sentence_embedding_dimension(self) -> int:
        if self._dim is not None:
            return self._dim
        if self._st_model is not None and hasattr(self._st_model, "get_sentence_embedding_dimension"):
            self._dim = self._st_model.get_sentence_embedding_dimension()
            return self._dim
        sample = self.encode(["dim_probe"])
        self._dim = sample.shape[1]
        return self._dim

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 64,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        **kwargs,
    ):
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            dim = self.get_sentence_embedding_dimension()
            arr = np.empty((0, dim), dtype="float32")
            return arr if convert_to_numpy else []

        if self._st_model is not None:
            return self._st_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=normalize_embeddings,
                convert_to_numpy=convert_to_numpy,
                **kwargs,
            )
        else:
            vecs = self._hf_embeddings.embed_documents(list(texts))
            arr = np.array(vecs, dtype=np.float32)
            if normalize_embeddings:
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                arr = arr / np.where(norms == 0, 1.0, norms)
            return arr if convert_to_numpy else arr.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """LangChain compatible interface."""
        arr = self.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return arr.tolist()

    def embed_query(self, text: str) -> List[float]:
        """LangChain compatible interface."""
        arr = self.encode([text], normalize_embeddings=True, convert_to_numpy=True)
        return arr[0].tolist()

    def __getattr__(self, name: str):
        if self._st_model is not None and hasattr(self._st_model, name):
            return getattr(self._st_model, name)
        if self._hf_embeddings is not None and hasattr(self._hf_embeddings, name):
            return getattr(self._hf_embeddings, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


class GeminiAPIEmbedder:
    """Wraps Google Gemini API for embeddings."""

    def __init__(self, model_name: str, gemini_key: str):
        from google import genai
        self.client = genai.Client(api_key=gemini_key)
        # Strip "models/" if present, as the new SDK expects 'text-embedding-004'
        if model_name.startswith("models/"):
            model_name = model_name[7:]
        self.model_name = model_name
        self._dim = None

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        show_progress_bar: Optional[bool] = None,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        **kwargs,
    ):
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            dim = self.get_sentence_embedding_dimension()
            arr = np.empty((0, dim), dtype="float32")
            return arr if convert_to_numpy else []

        embeddings = []
        # Gemini embed_content treats a list of strings as ONE concatenated
        # content and returns a single embedding. We must call it once per
        # text to get one embedding per chunk.
        for text in texts:
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=text,
            )
            embeddings.append(result.embeddings[0].values)

        embeddings = np.array(embeddings, dtype=np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.where(norms == 0, 1.0, norms)

        return embeddings if convert_to_numpy else embeddings.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """LangChain compatible interface."""
        arr = self.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return arr.tolist()

    def embed_query(self, text: str) -> List[float]:
        """LangChain compatible interface."""
        arr = self.encode([text], normalize_embeddings=True, convert_to_numpy=True)
        return arr[0].tolist()

    def get_sentence_embedding_dimension(self) -> int:
        if self._dim is None:
            probe = self.encode(["probe"])
            self._dim = probe.shape[1]
        return self._dim


def load_embedding_model(cfg: dict):
    """Load the embedding model (either local or API-based)."""
    model_name = cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    provider = cfg.get("embedding_provider", "local").lower()
    print(f"Loading embedding model: {model_name} via {provider}")

    if provider == "gemini_api":
        import os
        token = cfg.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY")
        if not token:
            raise ValueError("GEMINI_API_KEY must be set in .env to use the Gemini API for embeddings.")
        return GeminiAPIEmbedder(model_name, token)

    prefer_langchain = provider in ("langchain", "huggingface", "langchain_huggingface")
    model = LocalEmbedder(model_name, prefer_langchain=prefer_langchain)
    print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


def embed_chunks(
    chunks: List[Dict],
    model: Any,
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

    if not isinstance(embeddings, np.ndarray):
        embeddings = np.asarray(embeddings, dtype="float32")

    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings.astype("float32")


def build_faiss_index(embeddings: np.ndarray, embedding_dim: int) -> faiss.IndexFlatIP:
    if embeddings.shape[1] != embedding_dim:
        raise ValueError(
            f"Embedding dimension mismatch: computed embeddings have dimension {embeddings.shape[1]}, "
            f"but config specifies embedding_dim={embedding_dim}. "
            f"If you changed the embedding model, delete data/index/* and rebuild the index."
        )
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
    print(f"FAISS index loaded: {index.ntotal} vectors, dim={index.d}")
    return index


def build_and_save_index(chunks: List[Dict], cfg: dict):
    model = load_embedding_model(cfg)
    embeddings = embed_chunks(chunks, model)
    index = build_faiss_index(embeddings, cfg["embedding_dim"])
    save_index(index, cfg["index_save_path"])
    return model, index


if __name__ == "__main__":
    from ingest import load_chunks

    cfg = load_config()
    chunks = load_chunks(cfg["chunks_save_path"])
    model, index = build_and_save_index(chunks, cfg)

    test_query = "What is the main topic of this document?"
    q_emb = model.encode([test_query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    scores, indices = index.search(q_emb, min(3, len(chunks)))
    print(f"\nTest retrieval for: '{test_query}'")
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx == -1:
            continue
        print(f"  Rank {rank+1} (score={score:.4f}): {chunks[idx]['text'][:120]}...")
