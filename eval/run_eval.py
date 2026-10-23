"""Run the gold evaluation set through the RAG pipeline and record raw outputs.

Scoring happens in a separate step. This script only captures what the system
produced for each question, so runs can be compared after retrieval changes.

--retriever selects the engine: dense (rag.py) or hybrid (hybrid.py).
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import rag  # noqa: E402

DEFAULT_GOLD = REPO_ROOT / "eval" / "gold_set.json"
DEFAULT_OUT_DIR = REPO_ROOT / "eval" / "results"
DEFAULT_SLEEP = 2.0
MAX_RETRIES = 3
BACKOFF_SECONDS = [5, 15, 45]


def load_engine(name: str):
    if name == "hybrid":
        import hybrid
        engine = hybrid
    else:
        engine = rag
    assert_answer_takes_k(engine, name)
    return engine


def assert_answer_takes_k(engine, name: str) -> None:
    """Generation must honour --k, or retrieval and generation see different contexts."""
    params = inspect.signature(engine.answer).parameters
    if len(params) < 2:
        raise SystemExit(
            f"{name}.answer() takes no k argument, so --k would apply to retrieval "
            f"but not to generation, and the run would be invalid.\n"
            f"Fix: change it to answer(query, k=TOP_K) and pass k through to retrieve()."
        )


def load_gold(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise ValueError(f"{path} must contain a JSON list")
    required = {"id", "question", "reference_answer", "type", "answerable"}
    for item in items:
        missing = required - item.keys()
        if missing:
            raise ValueError(f"{item.get('id', '?')} missing fields: {sorted(missing)}")
    return items


def normalise_chunk(chunk) -> dict:
    """Chunks are dicts with text/source/page, but tolerate missing keys."""
    if isinstance(chunk, dict):
        return {
            "text": chunk.get("text", ""),
            "source": chunk.get("source"),
            "page": chunk.get("page"),
        }
    return {"text": str(chunk), "source": None, "page": None}


def call_with_retry(fn, *args, retries: int = MAX_RETRIES):
    last_error = None
    for attempt in range(retries):
        try:
            return fn(*args), None
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries - 1:
                wait = BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)]
                print(f"    retry {attempt + 1}/{retries - 1} in {wait}s: {exc}")
                time.sleep(wait)
    return None, last_error


def run_item(item: dict, engine, k: int) -> dict:
    record = {
        "id": item["id"],
        "question": item["question"],
        "type": item["type"],
        "answerable": item["answerable"],
        "reference_answer": item["reference_answer"],
        "source_hint": item.get("source"),
        "retrieved_chunks": [],
        "predicted_answer": None,
        "predicted_sources": [],
        "latency_seconds": None,
        "error": None,
    }

    start = time.perf_counter()

    chunks, err = call_with_retry(engine.retrieve, item["question"], k)
    if err is not None:
        record["error"] = f"retrieve failed: {err}"
        record["latency_seconds"] = round(time.perf_counter() - start, 3)
        return record
    record["retrieved_chunks"] = [normalise_chunk(c) for c in chunks]

    result, err = call_with_retry(engine.answer, item["question"], k)
    if err is not None:
        record["error"] = f"answer failed: {err}"
        record["latency_seconds"] = round(time.perf_counter() - start, 3)
        return record

    text, sources = result
    record["predicted_answer"] = text
    record["predicted_sources"] = sources
    record["latency_seconds"] = round(time.perf_counter() - start, 3)
    return record


def summarise(records: list[dict]) -> None:
    ok = [r for r in records if r["error"] is None]
    failed = [r for r in records if r["error"] is not None]
    latencies = [r["latency_seconds"] for r in ok if r["latency_seconds"] is not None]

    print("\n" + "-" * 62)
    print(f"{'questions run':<28}{len(records)}")
    print(f"{'completed':<28}{len(ok)}")
    print(f"{'failed':<28}{len(failed)}")
    if latencies:
        print(f"{'mean latency (s)':<28}{sum(latencies) / len(latencies):.2f}")
        print(f"{'max latency (s)':<28}{max(latencies):.2f}")

    sources_per_q = [
        len({c["source"] for c in r["retrieved_chunks"] if c["source"]})
        for r in ok if r["retrieved_chunks"]
    ]
    if sources_per_q:
        print(f"{'mean distinct sources':<28}{sum(sources_per_q) / len(sources_per_q):.2f}")

    by_type: dict[str, int] = {}
    for r in records:
        by_type[r["type"]] = by_type.get(r["type"], 0) + 1
    print("\nby type")
    for t in sorted(by_type):
        print(f"  {t:<26}{by_type[t]}")

    empty = [r["id"] for r in ok if not (r["predicted_answer"] or "").strip()]
    if empty:
        print(f"\nempty answers: {', '.join(empty)}")
    if failed:
        print(f"errors: {', '.join(r['id'] for r in failed)}")
    print("-" * 62)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the gold set through the RAG pipeline.")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--retriever", choices=["dense", "hybrid"], default="dense")
    parser.add_argument("--k", type=int, default=rag.TOP_K)
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP,
                        help="seconds between questions, to stay inside Groq free-tier limits")
    parser.add_argument("--limit", type=int, default=None, help="run only the first N questions")
    parser.add_argument("--tag", type=str, default=None,
                        help="label for this run; defaults to the retriever name")
    args = parser.parse_args()

    engine = load_engine(args.retriever)
    tag = args.tag or args.retriever

    items = load_gold(args.gold)
    if args.limit:
        items = items[: args.limit]

    print(f"gold set       {args.gold}")
    print(f"questions      {len(items)}")
    print(f"retriever      {args.retriever}")
    print(f"embed model    {rag.EMBED_MODEL}")
    print(f"llm model      {rag.LLM_MODEL}")
    print(f"top_k          {args.k}")
    print(f"tag            {tag}\n")

    records = []
    for n, item in enumerate(items, 1):
        print(f"[{n}/{len(items)}] {item['id']}  {item['question'][:60]}")
        try:
            record = run_item(item, engine, args.k)
        except Exception:  # noqa: BLE001
            record = {
                "id": item["id"],
                "question": item["question"],
                "type": item["type"],
                "answerable": item["answerable"],
                "reference_answer": item["reference_answer"],
                "error": traceback.format_exc(limit=3),
            }
        records.append(record)
        if record.get("error"):
            print(f"    ERROR {record['error'].splitlines()[-1][:80]}")
        if n < len(items):
            time.sleep(args.sleep)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"run_{tag}_{timestamp}.json"

    payload = {
        "run": {
            "tag": tag,
            "retriever": args.retriever,
            "timestamp_utc": timestamp,
            "gold_set": str(args.gold.relative_to(REPO_ROOT)),
            "embed_model": rag.EMBED_MODEL,
            "llm_model": rag.LLM_MODEL,
            "top_k": args.k,
            "question_count": len(records),
        },
        "results": records,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    summarise(records)
    print(f"\nwrote {out_path.relative_to(REPO_ROOT)}")
    return 1 if any(r.get("error") for r in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())