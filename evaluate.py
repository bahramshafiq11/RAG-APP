"""
evaluate.py: measure how accurate your RAG app is.

It asks every question in eval_questions.json and checks two things:
  1. Retrieval: did the search find a chunk from the page where the answer is?
  2. Answer:    does the AI's answer contain the words you expect?
                (For questions that are NOT in your documents, the AI should say
                 it couldn't find the answer instead of making something up.)

Before running: open the app, upload your PDF and click "Process documents".

Examples:
  python evaluate.py                     # use the TOP_K from the app's slider
  python evaluate.py --top-k 1 4 8       # compare several TOP_K values
  python evaluate.py --retrieval-only    # only test the search (fast, no LLM)
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from rag import DEFAULT_TOP_K, LLM_MODEL, answer_question, load_index, load_settings, search

QUESTIONS_FILE = Path(__file__).parent / "eval_questions.json"
RESULTS_FILE = Path(__file__).parent / "eval_results.csv"

# Phrases that count as "the AI admitted it doesn't know"
NOT_FOUND_PHRASES = ["couldn't find", "could not find", "not in your documents",
                     "not mentioned", "no information"]


def load_questions(path):
    with open(path, encoding="utf-8") as f:
        questions = json.load(f)["questions"]
    for q in questions:
        if "REPLACE" in json.dumps(q):
            sys.exit(f"Edit {path.name} first: replace the REPLACE examples with "
                     "questions about your own PDF.")
    return questions


def check_retrieval(q, hits):
    """True if any retrieved chunk is from one of the expected pages (None if not checked)."""
    if q.get("not_in_document") or not q.get("expected_pages"):
        return None
    return any(chunk["page"] in q["expected_pages"] for chunk, _ in hits)


def check_answer(q, answer):
    """True if the answer passes: has all expected keywords, or admits it doesn't know."""
    text = answer.lower().replace("’", "'")  # treat curly apostrophes like normal ones
    if q.get("not_in_document"):
        return any(phrase in text for phrase in NOT_FOUND_PHRASES)
    return all(word.lower() in text for word in q.get("expected_keywords", []))


def percent(passed, total):
    return f"{passed}/{total} ({100 * passed / total:.0f}%)" if total else "n/a"


def main():
    parser = argparse.ArgumentParser(description="Measure the accuracy of the RAG app.")
    parser.add_argument("--top-k", type=int, nargs="+",
                        help="one or more TOP_K values to test (default: the app's slider)")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="only test the search, skip asking the LLM")
    args = parser.parse_args()

    saved = load_index()
    if saved is None:
        sys.exit("No processed documents found. Open the app, upload your PDF and "
                 "click 'Process documents' first.")
    chunks, vectors, info = saved
    questions = load_questions(QUESTIONS_FILE)
    top_ks = args.top_k or [load_settings().get("top_k", DEFAULT_TOP_K)]

    print(f"Documents: {', '.join(info['files'])} | {len(chunks)} chunks "
          f"(chunk size {info['chunk_size']}) | model: {LLM_MODEL}")
    print(f"Questions: {len(questions)} | TOP_K values: {top_ks}\n")

    rows, summary = [], []
    for k in top_ks:
        print(f"=== TOP_K = {k} ===")
        retrieval_pass = retrieval_total = answer_pass = answer_total = 0
        seconds_total = 0.0

        for i, q in enumerate(questions, 1):
            start = time.time()
            if args.retrieval_only:
                hits, answer, answer_ok = search(q["question"], chunks, vectors, k), "", None
            else:
                answer, hits = answer_question(q["question"], chunks, vectors, k)
                answer_ok = check_answer(q, answer)
                answer_total += 1
                answer_pass += answer_ok
            seconds = time.time() - start
            seconds_total += seconds

            retrieval_ok = check_retrieval(q, hits)
            if retrieval_ok is not None:
                retrieval_total += 1
                retrieval_pass += retrieval_ok

            pages = [chunk["page"] for chunk, _ in hits]
            marks = []
            if retrieval_ok is not None:
                marks.append("search " + ("PASS" if retrieval_ok else "FAIL"))
            if answer_ok is not None:
                marks.append("answer " + ("PASS" if answer_ok else "FAIL"))
            if not marks:
                marks.append("skipped: not in document")
            print(f"  {i:>2}. [{' | '.join(marks)}] {q['question']}  (pages {pages}, {seconds:.1f}s)")

            rows.append({"top_k": k, "question": q["question"],
                         "search_found_page": retrieval_ok, "answer_correct": answer_ok,
                         "retrieved_pages": pages, "seconds": round(seconds, 1),
                         "answer": answer.strip()})

        summary.append((k, percent(retrieval_pass, retrieval_total),
                        percent(answer_pass, answer_total),
                        seconds_total / len(questions)))
        print()

    print("=== SUMMARY ===")
    print(f"{'TOP_K':>5}  {'Search found the page':<24}{'Answer correct':<18}Avg time")
    for k, retrieval, answer, avg in summary:
        print(f"{k:>5}  {retrieval:<24}{answer:<18}{avg:.1f}s")

    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nFull results (including every answer) saved to {RESULTS_FILE.name}")


if __name__ == "__main__":
    main()