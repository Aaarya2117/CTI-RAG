"""
pipeline.py
LangChain Expression Language (LCEL) pipeline orchestrating:
Laya Query Routing -> FAISS Retrieval -> Prompt Assembly -> OpenRouter Generation -> Laya Answer Gating.
"""

import os
os.environ.setdefault("USE_TF", "0")


from pathlib import Path
from typing import Optional, Dict, Any, List

try:
    from langchain_core.runnables import RunnableLambda
except ImportError:
    RunnableLambda = None

try:
    from config_utils import PROJECT_ROOT, load_config
    from ingest import ingest_document, load_chunks
    from embeddings import load_embedding_model, load_index, build_and_save_index
    from retriever import Retriever
    from generator import build_generator, PROMPT_TEMPLATE
    from laya_layer import route_query, gate_answer
    from ioc_extractor import extract_from_result
except ImportError:
    from .config_utils import PROJECT_ROOT, load_config
    from .ingest import ingest_document, load_chunks
    from .embeddings import load_embedding_model, load_index, build_and_save_index
    from .retriever import Retriever
    from .generator import build_generator, PROMPT_TEMPLATE
    from .laya_layer import route_query, gate_answer
    from .ioc_extractor import extract_from_result


class RAGPipeline:
    """
    End-to-end RAG pipeline using LangChain LCEL.
    Orchestrates Laya query routing -> FAISS retrieval with boosting -> prompt formatting ->
    OpenRouter generation -> Laya answer gating.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.retriever: Optional[Retriever] = None
        self.generator = None
        self._is_ready = False
        self._chain = None

    def index_document(self, file_path: str):
        print(f"\n{'='*50}")
        print(f"Indexing: {file_path}")
        print(f"{'='*50}")

        chunks = ingest_document(file_path, self.cfg)
        embed_model, index = build_and_save_index(chunks, self.cfg)
        self.retriever = Retriever(index, chunks, embed_model, top_k=self.cfg["top_k"])

        print(f"\nDocument indexed successfully. {len(chunks)} chunks ready.")
        self._is_ready = True
        self._chain = None

    def load_existing_index(self):
        chunks = load_chunks(self.cfg["chunks_save_path"])
        embed_model = load_embedding_model(self.cfg)
        index = load_index(self.cfg["index_save_path"])
        self.retriever = Retriever(index, chunks, embed_model, top_k=self.cfg["top_k"])
        self._is_ready = True
        self._chain = None
        print(f"Loaded existing index: {len(chunks)} chunks")

    def _load_generator_if_needed(self):
        if self.generator is None:
            print("Loading generator model (first query may be slow)...")
            self.generator = build_generator(self.cfg)

    def get_or_build_chain(self):
        """Construct the LCEL pipeline chain."""
        if self._chain is not None:
            return self._chain

        self._load_generator_if_needed()

        def route_step(input_data: Dict[str, Any]) -> Dict[str, Any]:
            q = input_data["question"]
            try:
                routing = route_query(q)
            except Exception as e:
                routing = {"intent": "general", "in_scope": True}
            return {**input_data, "routing": routing}

        def retrieve_step(state: Dict[str, Any]) -> Dict[str, Any]:
            q = state["question"]
            routing = state.get("routing", {})

            # Check scope guardrail
            in_scope = routing.get("in_scope", True)
            if in_scope is False or (isinstance(in_scope, dict) and in_scope.get("choice") is False):
                return {
                    **state,
                    "retrieved_chunks": [],
                    "context": "",
                    "refusal": (
                        "Query refused: input was classified as out-of-scope or an adversarial "
                        "instruction override attempt."
                    ),
                }

            retrieved = self.retriever.retrieve(q, routing=routing)
            context = self.retriever.format_context(retrieved)
            return {
                **state,
                "retrieved_chunks": retrieved,
                "context": context,
                "refusal": None,
            }

        def prompt_step(state: Dict[str, Any]) -> Dict[str, Any]:
            if state.get("refusal"):
                return state
            prompt = PROMPT_TEMPLATE.format(context=state["context"], question=state["question"])
            return {**state, "prompt": prompt}

        def generate_step(state: Dict[str, Any]) -> Dict[str, Any]:
            if state.get("refusal"):
                return {**state, "answer": state["refusal"]}

            try:
                answer = self.generator.generate(state["context"], state["question"])
            except ValueError as e:
                # Handle missing API key gracefully during testing
                answer = f"Generation unavailable: {e}"
            except Exception as e:
                answer = f"Generation error: {e}"

            return {**state, "answer": answer}

        def gate_step(state: Dict[str, Any]) -> Dict[str, Any]:
            answer = state["answer"]
            retrieved = state.get("retrieved_chunks", [])
            retrieved_ids = [c.get("chunk_id", i) for i, c in enumerate(retrieved)]

            gate_res = {"grounded": True, "cites_id": False}
            if not state.get("refusal") and not answer.startswith("Generation unavailable"):
                try:
                    gate_res = gate_answer(answer, retrieved_ids)
                except Exception as e:
                    gate_res = {"grounded": True, "cites_id": False, "error": str(e)}

            iocs = extract_from_result({
                "answer": answer,
                "retrieved_chunks": retrieved,
            })

            return {
                "question": state["question"],
                "answer": answer,
                "retrieved_chunks": retrieved,
                "context_used": state.get("context", ""),
                "routing": state.get("routing", {}),
                "gate": gate_res,
                "iocs": iocs,
            }

        if RunnableLambda is not None:
            self._chain = (
                RunnableLambda(route_step)
                | RunnableLambda(retrieve_step)
                | RunnableLambda(prompt_step)
                | RunnableLambda(generate_step)
                | RunnableLambda(gate_step)
            )
        else:
            class SimpleChain:
                def invoke(self, inp):
                    s = route_step(inp)
                    s = retrieve_step(s)
                    s = prompt_step(s)
                    s = generate_step(s)
                    return gate_step(s)
            self._chain = SimpleChain()

        return self._chain

    def ask(self, question: str, verbose: bool = False) -> dict:
        if not self._is_ready:
            raise RuntimeError("No document indexed. Call index_document() or load_existing_index() first.")

        chain = self.get_or_build_chain()
        result = chain.invoke({"question": question})

        if verbose:
            retrieved = result.get("retrieved_chunks", [])
            print(f"\nRetrieved {len(retrieved)} chunks:")
            for r in retrieved:
                boost_info = f" (boosted={r.get('boosted_score', r['score']):.4f})" if "boosted_score" in r else ""
                print(f"  [{r['rank']}] score={r['score']:.4f}{boost_info} — {r['text'][:80]}...")
            print(f"\nRouting: {result.get('routing')}")
            print(f"Gate: {result.get('gate')}")

        return result


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
    print("RAG Document Q&A (LCEL + Laya + OpenRouter) — type 'quit' to exit")
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
