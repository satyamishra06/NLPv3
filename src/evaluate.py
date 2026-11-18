import os
import json
import time
import ollama
from rag_engine import retrieve, build_prompt, OLLAMA_MODEL, TOP_K, PROJECT_DIR

QUERIES_PATH = os.path.join(PROJECT_DIR, "testing", "test_queries.json")
BASELINE_PROMPT_PATH = os.path.join(PROJECT_DIR, "prompts", "eval_system_prompt.txt")
OUT_DIR = os.path.join(PROJECT_DIR, "evaluation")
RESULTS_PATH = os.path.join(OUT_DIR, "results.jsonl")
TEMPERATURE = 0.3


def load_queries() -> list:
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_baseline_prompt() -> str:
    with open(BASELINE_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read().strip()


def run_baseline(query: str) -> str:
    messages = [
        {"role": "system", "content": load_baseline_prompt()},
        {"role": "user", "content": query},
    ]
    response = ollama.chat(model=OLLAMA_MODEL, messages=messages, options={"temperature": TEMPERATURE})
    return response["message"]["content"]


def run_rag(query: str, k: int = TOP_K) -> tuple:
    results = retrieve(query, k=k)
    context_chunks = [r["chunk"] for r in results]
    messages = build_prompt(query, context_chunks)
    response = ollama.chat(model=OLLAMA_MODEL, messages=messages, options={"temperature": TEMPERATURE})
    answer = response["message"]["content"]
    sources = [
        {"doc_name": r["chunk"]["doc_name"], "chunk_id": r["chunk"]["chunk_id"], "score": r["score"]}
        for r in results
    ]
    return answer, sources


def load_done_ids() -> set:
    done = set()
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["id"])
    return done


def main():
    queries = load_queries()
    os.makedirs(OUT_DIR, exist_ok=True)
    done_ids = load_done_ids()

    pending = [q for q in queries if q["id"] not in done_ids]
    print(f"Queries: {len(queries)} total, {len(done_ids)} already done, {len(pending)} to run")

    with open(RESULTS_PATH, "a", encoding="utf-8") as f:
        for q in pending:
            print(f"[{q['id']}/{len(queries)}] {q['query']}")

            t0 = time.time()
            baseline_answer = run_baseline(q["query"])
            baseline_time = round(time.time() - t0, 2)

            t0 = time.time()
            rag_answer, rag_sources = run_rag(q["query"])
            rag_time = round(time.time() - t0, 2)

            record = {
                "id": q["id"],
                "query": q["query"],
                "category": q["category"],
                "answer_in_kb": q["answer_in_kb"],
                "expected_doc": q.get("expected_doc"),
                "baseline_answer": baseline_answer,
                "rag_answer": rag_answer,
                "rag_sources": rag_sources,
                "baseline_time_s": baseline_time,
                "rag_time_s": rag_time,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            print(f"      baseline={baseline_time}s rag={rag_time}s")

    print(f"\nDone. Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
