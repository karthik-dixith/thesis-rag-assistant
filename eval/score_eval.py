"""Score a run file produced by run_eval.py using an LLM judge on Groq.

Metrics:
  faithfulness      claims in the answer supported by the retrieved context
  answer_relevance  how fully the answer addresses the question
  context_precision fraction of retrieved chunks relevant to the question
  refusal_correct   refused an unanswerable question, answered an answerable one

Relevance and faithfulness are averaged over answerable questions only, since a
correct refusal scores zero on both by construction.

Partial results are written whenever scoring stops early, and --resume continues
from one. Per-minute rate limits are retried; per-day limits abort immediately.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

load_dotenv(REPO_ROOT / ".env")

RESULTS_DIR = REPO_ROOT / "eval" / "results"
DEFAULT_JUDGE_MODEL = "openai/gpt-oss-120b"
JUDGE_TEMPERATURE = 0.0
DEFAULT_SLEEP = 1.5
DEFAULT_MAX_CHUNK_CHARS = 1200
MAX_RETRIES = 3
BACKOFF_SECONDS = [5, 15, 45]

_client = Groq(api_key=os.environ["GROQ_API_KEY"])


class DailyLimitReached(RuntimeError):
    """Per-day token quota exhausted. Retrying will not help."""


FAITHFULNESS_PROMPT = """You are grading whether an answer is grounded in the context it was given.

Break the ANSWER into atomic factual claims. For each claim, decide whether the CONTEXT
directly supports it. A claim that is plausible but not stated in the CONTEXT is unsupported.
Ignore hedging, citations and formatting; grade only factual content.

If the ANSWER makes no factual claims, return an empty list. A refusal such as
"I don't have that information" makes no factual claims and must return an empty list.

Return JSON only, in this exact shape:
{{"claims": [{{"claim": "...", "verdict": 1, "reason": "..."}}]}}
verdict is 1 if supported by the CONTEXT, 0 if not.

CONTEXT:
{context}

ANSWER:
{answer}"""


RELEVANCE_PROMPT = """You are grading whether an answer addresses the question asked.

Score using this rubric:
  1   fully addresses the question
  0.5 addresses the question partially, or answers only some parts of it
  0   does not address the question, or declines to answer

Also report whether the answer declined to provide information (a refusal such as
"I don't have that information"), regardless of whether declining was appropriate.

Judge only whether the question was addressed. Do not judge factual correctness.

Return JSON only, in this exact shape:
{{"refused": false, "score": 1, "reason": "..."}}

QUESTION:
{question}

ANSWER:
{answer}"""


PRECISION_PROMPT = """You are grading whether retrieved passages are useful for answering a question.

For each numbered CHUNK, decide whether it contains information that helps produce the
REFERENCE ANSWER. A chunk on the same broad topic but carrying none of the needed
information is not relevant.

Return JSON only, in this exact shape:
{{"verdicts": [{{"index": 1, "verdict": 1, "reason": "..."}}]}}
verdict is 1 if the chunk is relevant, 0 if not. Include every chunk exactly once.

QUESTION:
{question}

REFERENCE ANSWER:
{reference}

CHUNKS:
{chunks}"""


def is_daily_limit(exc: Exception) -> bool:
    text = str(exc).lower()
    return "tpd" in text or "tokens per day" in text


def judge(prompt: str, model: str) -> dict:
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = _client.chat.completions.create(
                model=model,
                temperature=JUDGE_TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
            )
            return parse_json(response.choices[0].message.content)
        except Exception as exc:  # noqa: BLE001
            if is_daily_limit(exc):
                raise DailyLimitReached(str(exc)) from exc
            last_error = exc
            if attempt < MAX_RETRIES - 1:
                wait = BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)]
                print(f"    retry {attempt + 1} in {wait}s: {str(exc)[:90]}")
                time.sleep(wait)
    raise RuntimeError(f"judge failed after {MAX_RETRIES} attempts: {last_error}")


def parse_json(text: str) -> dict:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))
    braced = re.search(r"\{.*\}", text, re.DOTALL)
    if braced:
        return json.loads(braced.group(0))
    raise ValueError(f"no JSON found in judge response: {text[:200]}")


def clip(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + " [...]"


def format_context(chunks: list[dict], limit: int) -> str:
    parts = []
    for n, chunk in enumerate(chunks, 1):
        src = chunk.get("source") or "unknown"
        page = chunk.get("page")
        parts.append(f"[chunk {n}] ({src}, page {page})\n{clip(chunk.get('text', ''), limit)}")
    return "\n\n".join(parts) if parts else "(no context retrieved)"


def score_record(record: dict, model: str, limit: int) -> dict:
    answer = (record.get("predicted_answer") or "").strip()
    chunks = record.get("retrieved_chunks") or []

    scores = {
        "id": record["id"],
        "type": record["type"],
        "answerable": record["answerable"],
        "faithfulness": None,
        "answer_relevance": None,
        "context_precision": None,
        "refused": None,
        "refusal_correct": None,
        "claims": [],
        "chunk_verdicts": [],
        "notes": [],
    }

    if record.get("error") or not answer:
        scores["notes"].append("no answer produced; skipped")
        return scores

    faith = judge(
        FAITHFULNESS_PROMPT.format(context=format_context(chunks, limit), answer=answer),
        model,
    )
    claims = faith.get("claims", [])
    scores["claims"] = claims
    if claims:
        supported = sum(1 for c in claims if int(c.get("verdict", 0)) == 1)
        scores["faithfulness"] = round(supported / len(claims), 3)
    else:
        scores["notes"].append("no factual claims to grade")

    rel = judge(
        RELEVANCE_PROMPT.format(question=record["question"], answer=answer),
        model,
    )
    scores["answer_relevance"] = float(rel.get("score", 0))
    scores["refused"] = bool(rel.get("refused", False))
    scores["refusal_correct"] = scores["refused"] != bool(record["answerable"])

    if chunks:
        numbered = "\n\n".join(
            f"[{n}] {clip(c.get('text', ''), limit)}" for n, c in enumerate(chunks, 1)
        )
        prec = judge(
            PRECISION_PROMPT.format(
                question=record["question"],
                reference=record["reference_answer"],
                chunks=numbered,
            ),
            model,
        )
        verdicts = prec.get("verdicts", [])
        scores["chunk_verdicts"] = verdicts
        if verdicts:
            relevant = sum(1 for v in verdicts if int(v.get("verdict", 0)) == 1)
            scores["context_precision"] = round(relevant / len(verdicts), 3)

    return scores


def mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None


def fmt(value) -> str:
    return "-" if value is None else f"{value:.2f}"


def summarise(scored: list[dict], complete: bool) -> dict:
    answerable = [s for s in scored if s["answerable"]]
    negatives = [s for s in scored if not s["answerable"]]

    aggregate = {
        "faithfulness": mean([s["faithfulness"] for s in answerable]),
        "answer_relevance": mean([s["answer_relevance"] for s in answerable]),
        "context_precision": mean([s["context_precision"] for s in answerable]),
        "refusal_accuracy": mean(
            [1.0 if s["refusal_correct"] else 0.0 for s in scored
             if s["refusal_correct"] is not None]
        ),
    }

    print("\n" + "=" * 64)
    if not complete:
        print("PARTIAL RESULTS - scoring stopped early, numbers are not comparable")
        print("-" * 64)
    print(f"{'metric':<24}{'score':>8}   basis")
    print("-" * 64)
    print(f"{'faithfulness':<24}{fmt(aggregate['faithfulness']):>8}   {len(answerable)} answerable")
    print(f"{'answer relevance':<24}{fmt(aggregate['answer_relevance']):>8}   {len(answerable)} answerable")
    print(f"{'context precision':<24}{fmt(aggregate['context_precision']):>8}   {len(answerable)} answerable")
    print(f"{'refusal accuracy':<24}{fmt(aggregate['refusal_accuracy']):>8}   all {len(scored)}")
    print("=" * 64)

    print(f"\n{'id':<6}{'type':<12}{'faith':>7}{'relev':>7}{'prec':>7}  refusal")
    print("-" * 64)
    for s in scored:
        if s["refusal_correct"] is None:
            flag = "-"
        else:
            flag = "ok" if s["refusal_correct"] else "WRONG"
        print(f"{s['id']:<6}{s['type']:<12}{fmt(s['faithfulness']):>7}"
              f"{fmt(s['answer_relevance']):>7}{fmt(s['context_precision']):>7}  {flag}")

    bad = [s["id"] for s in scored if s["refusal_correct"] is False]
    if bad:
        print(f"\nrefusal failures: {', '.join(bad)}")
    weak = [s["id"] for s in answerable
            if s["context_precision"] is not None and s["context_precision"] < 0.5]
    if weak:
        print(f"context precision below 0.5: {', '.join(weak)}")
    if negatives:
        print(f"\nnegatives ({len(negatives)}) excluded from faithfulness, relevance and precision")

    return aggregate


def latest_run() -> Path:
    runs = sorted(RESULTS_DIR.glob("run_*.json"))
    if not runs:
        raise SystemExit(f"no run files in {RESULTS_DIR}; run run_eval.py first")
    return runs[-1]


def resolve_in_results(path: Path) -> Path:
    """Accept a full path, a path relative to cwd, or a bare filename in results/."""
    if path.exists():
        return path
    candidate = RESULTS_DIR / path.name
    if candidate.exists():
        return candidate
    raise SystemExit(f"file not found: {path} (also looked in {RESULTS_DIR})")


def load_resume(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    return {s["id"]: s for s in payload.get("per_question", [])}


def write_output(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Score a run file with an LLM judge.")
    parser.add_argument("--run", type=Path, default=None, help="defaults to the newest run file")
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-chunk-chars", type=int, default=DEFAULT_MAX_CHUNK_CHARS,
                        help="truncate chunk text in judge prompts to control token spend")
    parser.add_argument("--resume", type=Path, default=None,
                        help="a previous scored file; already-scored questions are reused")
    args = parser.parse_args()

    judge_model = args.judge_model
    run_path = resolve_in_results(args.run) if args.run else latest_run()

    with run_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    records = payload["results"]
    if args.limit:
        records = records[: args.limit]

    done = load_resume(resolve_in_results(args.resume)) if args.resume else {}

    print(f"run file       {run_path.name}")
    print(f"generator      {payload['run']['llm_model']}")
    print(f"judge          {judge_model}")
    print(f"questions      {len(records)}")
    print(f"chunk clip     {args.max_chunk_chars} chars")
    if done:
        print(f"resumed        {len(done)} already scored")
    print()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"scored_{payload['run']['tag']}_{timestamp}.json"

    scored: list[dict] = []
    complete = True
    stop_reason = None

    for n, record in enumerate(records, 1):
        if record["id"] in done:
            print(f"[{n}/{len(records)}] {record['id']}  (reused)")
            scored.append(done[record["id"]])
            continue
        print(f"[{n}/{len(records)}] {record['id']}")
        try:
            scored.append(score_record(record, judge_model, args.max_chunk_chars))
        except DailyLimitReached as exc:
            complete = False
            stop_reason = f"daily token limit reached: {str(exc)[:160]}"
            print(f"\n  STOPPED  {stop_reason}")
            break
        except Exception as exc:  # noqa: BLE001
            complete = False
            stop_reason = f"{type(exc).__name__}: {str(exc)[:160]}"
            print(f"\n  STOPPED  {stop_reason}")
            break
        if n < len(records):
            time.sleep(args.sleep)

    aggregate = summarise(scored, complete)

    write_output(out_path, {
        "scoring": {
            "run_file": run_path.name,
            "tag": payload["run"]["tag"],
            "judge_model": judge_model,
            "generator_model": payload["run"]["llm_model"],
            "top_k": payload["run"]["top_k"],
            "max_chunk_chars": args.max_chunk_chars,
            "timestamp_utc": timestamp,
            "complete": complete,
            "stop_reason": stop_reason,
            "scored_count": len(scored),
            "question_count": len(records),
        },
        "aggregate": aggregate,
        "per_question": scored,
    })

    print(f"\nwrote {out_path.relative_to(REPO_ROOT)}")
    if not complete:
        print(f"resume with: python eval/score_eval.py --run {run_path.name} "
              f"--resume {out_path.name}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())