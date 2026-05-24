"""
evaluate.py
Evaluate the RAG pipeline on a hand-crafted Q&A set.
Metrics: retrieval accuracy (did the right chunk get retrieved?) + answer relevance (manual).
"""

import json
import yaml
from pathlib import Path
from pipeline import RAGPipeline, load_config


EVAL_QA_TEMPLATE = [
    {
        "question": "REPLACE WITH YOUR QUESTION 1",
        "expected_answer_keywords": ["keyword1", "keyword2"],
        "expected_chunk_contains": "phrase that should appear in the top retrieved chunk"
    },
    {
        "question": "REPLACE WITH YOUR QUESTION 2",
        "expected_answer_keywords": ["keyword3"],
        "expected_chunk_contains": "another phrase"
    }
]


def keyword_match_score(answer: str, keywords: list) -> float:
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return hits / len(keywords) if keywords else 0.0


def retrieval_accuracy(retrieved_chunks: list, expected_phrase: str) -> bool:
    phrase_lower = expected_phrase.lower()
    return any(phrase_lower in chunk["text"].lower() for chunk in retrieved_chunks)


def run_evaluation(pipeline: RAGPipeline, qa_pairs: list, cfg: dict) -> dict:
    print(f"\nRunning evaluation on {len(qa_pairs)} Q&A pairs...")

    results = []
    retrieval_hits = 0
    total_keyword_score = 0.0

    for i, qa in enumerate(qa_pairs):
        print(f"\n[{i+1}/{len(qa_pairs)}] Q: {qa['question']}")

        result = pipeline.ask(qa["question"])
        answer = result["answer"]
        retrieved = result["retrieved_chunks"]

        kw_score = keyword_match_score(answer, qa.get("expected_answer_keywords", []))
        ret_hit = retrieval_accuracy(retrieved, qa.get("expected_chunk_contains", ""))

        if ret_hit:
            retrieval_hits += 1
        total_keyword_score += kw_score

        print(f"  Answer: {answer[:120]}...")
        print(f"  Keyword match: {kw_score:.2f} | Retrieval hit: {ret_hit}")

        results.append({
            "question": qa["question"],
            "answer": answer,
            "expected_keywords": qa.get("expected_answer_keywords", []),
            "keyword_match_score": kw_score,
            "retrieval_hit": ret_hit,
            "top_chunk_text": retrieved[0]["text"][:200] if retrieved else "",
            "top_chunk_score": retrieved[0]["score"] if retrieved else 0.0,
        })

    summary = {
        "total_questions": len(qa_pairs),
        "retrieval_accuracy": retrieval_hits / len(qa_pairs) if qa_pairs else 0.0,
        "avg_keyword_match_score": total_keyword_score / len(qa_pairs) if qa_pairs else 0.0,
        "per_question_results": results,
    }

    print(f"\n{'='*50}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"Retrieval Accuracy:      {summary['retrieval_accuracy']:.2%}")
    print(f"Avg Keyword Match Score: {summary['avg_keyword_match_score']:.2f}")
    print(f"{'='*50}")

    save_path = cfg["results_path"]
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nEvaluation results saved: {save_path}")

    return summary


if __name__ == "__main__":
    cfg = load_config()

    qa_path = cfg["eval_qa_path"]
    if not Path(qa_path).exists():
        print(f"No eval Q&A file found at {qa_path}")
        print("Creating a template file — fill it in with your questions, then re-run.")
        Path(qa_path).parent.mkdir(parents=True, exist_ok=True)
        with open(qa_path, "w") as f:
            json.dump(EVAL_QA_TEMPLATE, f, indent=2)
        print(f"Template saved to: {qa_path}")
        exit(0)

    with open(qa_path) as f:
        qa_pairs = json.load(f)

    pipeline = RAGPipeline(cfg)
    pipeline.load_existing_index()
    run_evaluation(pipeline, qa_pairs, cfg)
