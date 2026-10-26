"""Hybrid retrieval: dense FAISS + BM25, fused with Reciprocal Rank Fusion.

Exposes the same retrieve/answer interface as rag.py so the two can be swapped
in the eval runner without touching the deployed app.

Generation parameters below must stay identical to rag.answer, otherwise a run
comparison measures the prompt as well as the retriever.
"""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

import rag

FUSION_POOL = 20
RRF_K = 60
TEMPERATURE = 0.2

_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


# Chunks are already loaded by rag at import; reuse rather than loading twice.
_corpus = [tokenize(c.get("text", "")) for c in rag._chunks]
_bm25 = BM25Okapi(_corpus)


def _dense_ranked(query: str, n: int) -> list[int]:
    vec = rag._model.encode([query], normalize_embeddings=True)
    _, indices = rag._index.search(np.asarray(vec, dtype="float32"), n)
    return [int(i) for i in indices[0] if i != -1]


def _bm25_ranked(query: str, n: int) -> list[int]:
    scores = _bm25.get_scores(tokenize(query))
    order = np.argsort(scores)[::-1][:n]
    return [int(i) for i in order if scores[i] > 0]


def _rrf(rankings: list[list[int]], k_rrf: int = RRF_K) -> list[tuple[int, float]]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, 1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k_rrf + rank)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


def retrieve(query: str, k: int = rag.TOP_K, pool: int = FUSION_POOL) -> list[dict]:
    dense = _dense_ranked(query, pool)
    lexical = _bm25_ranked(query, pool)
    fused = _rrf([dense, lexical])
    return [rag._chunks[idx] for idx, _ in fused[:k]]


def retrieve_debug(query: str, k: int = rag.TOP_K, pool: int = FUSION_POOL) -> dict:
    """Per-ranker positions for the fused results. Used to inspect what BM25 added."""
    dense = _dense_ranked(query, pool)
    lexical = _bm25_ranked(query, pool)
    fused = _rrf([dense, lexical])

    rows = []
    for idx, score in fused[:k]:
        rows.append({
            "chunk_index": idx,
            "rrf_score": round(score, 5),
            "dense_rank": dense.index(idx) + 1 if idx in dense else None,
            "bm25_rank": lexical.index(idx) + 1 if idx in lexical else None,
            "source": rag._chunks[idx].get("source"),
            "page": rag._chunks[idx].get("page"),
        })
    return {"query": query, "results": rows}


def answer(query: str, k: int = rag.TOP_K) -> tuple[str, list[dict]]:
    chunks = retrieve(query, k)
    prompt = rag.build_prompt(query, chunks)
    response = rag._client.chat.completions.create(
        model=rag.LLM_MODEL,
        temperature=TEMPERATURE,
        messages=[{"role": "user", "content": prompt}],
    )
        text = rag.clean_citations(response.choices[0].message.content)
    sources = [{"source": c.get("source"), "page": c.get("page")} for c in chunks]
    return text, sources


if __name__ == "__main__":
    import json
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "How did the VIO benchmark measure positioning accuracy?"
    print(json.dumps(retrieve_debug(q), indent=2))
