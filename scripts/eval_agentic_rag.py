#!/usr/bin/env python3
"""Evaluate baseline vs improved RAG on LLM-RL papers.

Usage:
  python scripts/eval_agentic_rag.py --base-url http://localhost:3100

What it does:
1) Ingest up to 10 LLM-RL papers (starting with DeepSeekMath/GRPO).
2) Build a topic and add these papers.
3) Run one differentiating question through baseline and improved topic RAG.
4) Print side-by-side outputs and simple quality heuristics.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import requests


DEFAULT_PAPERS = [
    "2402.03300",  # DeepSeekMath (introduces GRPO)
    "2203.02155",  # InstructGPT (RLHF)
    "2305.18290",  # RRHF
    "2307.09288",  # DPO
    "2403.12036",  # IPO / preference optimization variants
    "2405.14734",  # SimPO
    "2404.10719",  # KTO / preference optimization family
    "2406.08414",  # RLHF scaling / alignment variants
    "2312.00849",  # RLAIF-related
    "2402.01306",  # reward modeling / alignment-related
]

EVAL_QUESTION = (
    "Compare GRPO, PPO-based RLHF, and DPO-style methods for LLM alignment: "
    "which one is more stable under sparse/length-biased rewards, what are the main trade-offs, "
    "and in what scenarios should each be preferred? Cite specific evidence."
)


def post(base: str, path: str, payload: dict[str, Any], timeout: int = 180) -> dict[str, Any]:
    r = requests.post(f"{base}{path}", json=payload, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"{path} failed [{r.status_code}]: {r.text[:500]}")
    return r.json()


def get(base: str, path: str, timeout: int = 60) -> dict[str, Any]:
    r = requests.get(f"{base}{path}", timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"{path} failed [{r.status_code}]: {r.text[:500]}")
    return r.json()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:3100")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--question", default=EVAL_QUESTION)
    args = ap.parse_args()

    base = args.base_url.rstrip("/")

    print("[1/5] Health check")
    print(get(base, "/health"))

    print("[2/5] Configure endpoints for deepseek + embed")
    post(
        base,
        "/config",
        {
            "settings": {
                "llm_base_url": "https://restrainable-victor-nonosmotically.ngrok-free.dev/deepseek/v1",
                "embedding_base_url": "https://restrainable-victor-nonosmotically.ngrok-free.dev/embed/v1",
            }
        },
    )

    print("[3/5] Ingest papers")
    ingested_ids: list[str] = []
    for arxiv_id in DEFAULT_PAPERS[: args.limit]:
        try:
            # download metadata + pdf
            resolved = post(base, "/arxiv/resolve", {"arxiv_id": arxiv_id})
            dl = post(base, "/arxiv/download", {"arxiv_id": arxiv_id}, timeout=300)
            # save paper first
            save = post(
                base,
                "/papers/save",
                {
                    "arxiv_id": resolved["arxiv_id"],
                    "title": resolved["title"],
                    "authors": resolved["authors"],
                    "abstract": resolved.get("summary"),
                    "pdf_url": resolved.get("pdf_url"),
                    "pdf_path": dl["pdf_path"],
                },
                timeout=300,
            )
            # ensure full-text embedding index
            post(
                base,
                "/embed/fulltext",
                {"arxiv_id": resolved["arxiv_id"], "pdf_path": dl["pdf_path"]},
                timeout=600,
            )
            ingested_ids.append(resolved["arxiv_id"])
            print(f"  + {resolved['arxiv_id']}: {resolved['title'][:80]}")
        except Exception as e:
            print(f"  - skip {arxiv_id}: {e}")

    if not ingested_ids:
        raise RuntimeError("No papers ingested; cannot run evaluation")

    print("[4/5] Create topic and add ingested papers")
    topic_name = f"agentic-rag-llm-rl-{int(time.time())}"
    topic = post(base, "/topic/create", {"name": topic_name, "topic_query": "LLM RL algorithms"})
    topic_id = topic["topic_id"]

    all_papers = get(base, "/tree")
    # fallback lookup from DB not exposed; use topic search to discover candidates
    searched = post(base, "/topic/search", {"topic": "LLM RL algorithms", "limit": 200, "offset": 0})
    id_to_pid = {p["arxiv_id"]: p["paper_id"] for p in searched.get("papers", []) if p.get("arxiv_id")}
    paper_ids = [id_to_pid[a] for a in ingested_ids if a in id_to_pid]
    if not paper_ids:
        raise RuntimeError("Could not map ingested arxiv IDs to paper IDs for topic pool")
    post(base, f"/topic/{topic_id}/papers", {"paper_ids": paper_ids})

    print("[5/5] Compare baseline vs improved topic query")
    compare = post(base, f"/topic/{topic_id}/query/compare", {"question": args.question}, timeout=900)

    baseline_answer = compare["baseline"]["answer"]
    improved_answer = compare["improved"]["answer"]

    out = {
        "question": args.question,
        "papers_used": len(paper_ids),
        "baseline": baseline_answer,
        "improved": improved_answer,
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
