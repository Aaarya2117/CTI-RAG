"""
pipeline.py
Wires together ingestion → embedding → retrieval → generation.
This is the main entry point for the RAG Q&A system.
"""

from pathlib import Path
from typing import Optional

try:
    from config_utils import PROJECT_ROOT
    from ingest import ingest_document, load_chunks, load_config
    from embeddings import load_embedding_model, load_index, build_and_save_index
    from retriever import Retriever
    from generator import build_generator
except ImportError:
    from .config_utils import PROJECT_ROOT
    from .ingest import ingest_document, load_chunks, load_config
    from .embeddings import load_embedding_model, load_index, build_and_save_index
    from .retriever import Retriever
    from .generator import build_generator


class RAGPipeline:
    """
    End-to-end RAG pipeline.
    Once initialized with a document, call .ask(question) to get answers.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.retriever: Optional[Retriever] = None
        self.generator = None
        self._is_ready = False

    def index_document(self, file_path: str):
        print(f"\n{'='*50}")
        print(f"Indexing: {file_path}")
        print(f"{'='*50}")

        chunks = ingest_document(file_path, self.cfg)
        embed_model, index = build_and_save_index(chunks, self.cfg)
        self.retriever = Retriever(index, chunks, embed_model, top_k=self.cfg["top_k"])

        print(f"\nDocument indexed successfully. {len(chunks)} chunks ready.")
        self._is_ready = True

    def load_existing_index(self):
        chunks = load_chunks(self.cfg["chunks_save_path"])
        embed_model = load_embedding_model(self.cfg)
        index = load_index(self.cfg["index_save_path"])
        self.retriever = Retriever(index, chunks, embed_model, top_k=self.cfg["top_k"])
        self._is_ready = True
        print(f"Loaded existing index: {len(chunks)} chunks")

    def _load_generator_if_needed(self):
        if self.generator is None:
            print("Loading generator model (first query may be slow)...")
            self.generator = build_generator(self.cfg)

    def ask(self, question: str, verbose: bool = False) -> dict:
        if not self._is_ready:
            raise RuntimeError("No document indexed. Call index_document() or load_existing_index() first.")

        self._load_generator_if_needed()
        retrieved = self.retriever.retrieve(question)
        context = self.retriever.format_context(retrieved)

        if verbose:
            print(f"\nRetrieved {len(retrieved)} chunks:")
            for r in retrieved:
                print(f"  [{r['rank']}] score={r['score']:.4f} — {r['text'][:80]}...")

        answer = self.generator.generate(context, question)

        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": retrieved,
            "context_used": context,
        }


if __name__ == "__main__":
    import sys

    cfg = load_config()
    pipeline = RAGPipeline(cfg)

    index_exists = (
        Path(cfg["index_save_path"]).exists() and
        Path(cfg["chunks_save_path"]).exists()
    )

    if index_exists:
        print("Found existing index. Loading...")
        pipeline.load_existing_index()
    else:
        docs = list((PROJECT_ROOT / "data" / "documents").glob("*.*"))
        if not docs:
            print("No documents found in data/documents/. Add a .pdf or .txt file.")
            sys.exit(1)
        pipeline.index_document(str(docs[0]))

    print("\n" + "="*50)
    print("RAG Document Q&A — type 'quit' to exit")
    print("="*50 + "\n")

    while True:
        query = input("Your question: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue
        try:
            result = pipeline.ask(query, verbose=True)
            print(f"\nAnswer: {result['answer']}\n")
        except Exception as exc:
            print(f"\nError: {exc}\n")
