#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import pathlib
import re
from dataclasses import dataclass
from typing import Any

import arxiv
from openai import AsyncOpenAI

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src/backend"))
import rag  # noqa: E402

LLM_BASE = "https://restrainable-victor-nonosmotically.ngrok-free.dev/deepseek/v1"
EMBED_BASE = "https://restrainable-victor-nonosmotically.ngrok-free.dev/embed/v1"
API_KEY = "none"
LLM_MODEL = "deepseek-ai/DeepSeek-V3.2"
EMBED_MODEL = "Qwen/Qwen3-VL-Embedding-8B"

QUESTION = (
    "Across these papers, compare GRPO, PPO-style RLHF, and DPO-style methods for LLM alignment: "
    "which appears more stable under sparse or length-biased rewards, what are the key trade-offs, "
    "and when should each be preferred? Ground your answer in evidence."
)

KEY_TERMS = ["grpo", "ppo", "rlhf", "dpo", "stability", "sparse", "length", "reward", "kl", "variance"]


@dataclass
class PaperRef:
    arxiv_id: str
    title: str
    pdf_path: str


def pick_candidate_ids() -> list[str]:
    ids = ["2402.03300"]  # DeepSeekMath / GRPO first
    client = arxiv.Client()
    searches = [
        arxiv.Search(query='all:"reinforcement learning from human feedback"', max_results=30, sort_by=arxiv.SortCriterion.Relevance),
        arxiv.Search(query='all:"direct preference optimization"', max_results=30, sort_by=arxiv.SortCriterion.Relevance),
        arxiv.Search(query='all:"group relative policy optimization"', max_results=20, sort_by=arxiv.SortCriterion.Relevance),
    ]
    seen = set(ids)
    for s in searches:
        for r in client.results(s):
            aid = r.get_short_id().split("v")[0]
            if aid not in seen:
                seen.add(aid)
                ids.append(aid)
            if len(ids) >= 30:
                return ids
    return ids


def download_papers(limit: int = 10) -> list[PaperRef]:
    out_dir = pathlib.Path("/workspace/repos/paper-curator-agentic-rag/storage/eval_llm_rl")
    out_dir.mkdir(parents=True, exist_ok=True)
    refs: list[PaperRef] = []

    client = arxiv.Client()
    for aid in pick_candidate_ids():
        if len(refs) >= limit:
            break
        try:
            search = arxiv.Search(id_list=[aid])
            res = list(client.results(search))
            if not res:
                continue
            p = res[0]
            pdf_path = p.download_pdf(dirpath=str(out_dir))
            refs.append(PaperRef(arxiv_id=p.get_short_id().split("v")[0], title=p.title, pdf_path=pdf_path))
            print(f"+ downloaded {refs[-1].arxiv_id}: {refs[-1].title[:90]}")
        except Exception as e:
            print(f"- skip {aid}: {e}")
    return refs


def score_answer(text: str) -> dict[str, Any]:
    low = text.lower()
    term_hits = [t for t in KEY_TERMS if t in low]
    citation_hits = len(re.findall(r"\[source\]|source|chunk#|\[paper", low))
    return {
        "term_hits": len(term_hits),
        "terms": term_hits,
        "citation_hits": citation_hits,
        "length": len(text),
    }


async def aggregate_topic_answer(client: AsyncOpenAI, question: str, per_paper: list[dict[str, str]], mode: str) -> str:
    block = "\n\n".join([f"Paper: {x['title']}\nAnswer: {x['answer']}" for x in per_paper])
    prompt = (
        f"Question: {question}\n\n"
        f"You are synthesizing {mode} per-paper RAG answers. Produce one final topic answer with:\n"
        "1) explicit method comparison; 2) concrete trade-offs; 3) when-to-use guidance; "
        "4) highlight uncertainty where evidence is weak.\n\n"
        f"Per-paper evidence:\n{block}\n"
    )
    resp = await client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1400,
    )
    return (resp.choices[0].message.content or "").strip()


async def run_eval() -> dict[str, Any]:
    papers = download_papers(limit=10)
    if len(papers) < 4:
        raise RuntimeError("Too few papers downloaded for meaningful evaluation")

    llm_client = AsyncOpenAI(base_url=LLM_BASE, api_key=API_KEY)
    embed_client = AsyncOpenAI(base_url=EMBED_BASE, api_key=API_KEY)

    baseline_rows = []
    improved_rows = []

    for p in papers:
        base = await rag.rag_answer_async(
            question=QUESTION,
            pdf_path=p.pdf_path,
            arxiv_id=None,
            llm_client=llm_client,
            llm_model=LLM_MODEL,
            embed_client=embed_client,
            embed_model=EMBED_MODEL,
            return_details=True,
            rag_mode=rag.RAG_MODE_BASELINE,
        )
        imp = await rag.rag_answer_async(
            question=QUESTION,
            pdf_path=p.pdf_path,
            arxiv_id=None,
            llm_client=llm_client,
            llm_model=LLM_MODEL,
            embed_client=embed_client,
            embed_model=EMBED_MODEL,
            return_details=True,
            rag_mode=rag.RAG_MODE_IMPROVED,
        )
        baseline_rows.append({"title": p.title, "arxiv_id": p.arxiv_id, "answer": base["answer"], "details": base})
        improved_rows.append({"title": p.title, "arxiv_id": p.arxiv_id, "answer": imp["answer"], "details": imp})
        print(f"qa done: {p.arxiv_id}")

    baseline_topic = await aggregate_topic_answer(llm_client, QUESTION, baseline_rows, "baseline")
    improved_topic = await aggregate_topic_answer(llm_client, QUESTION, improved_rows, "improved")

    baseline_score = score_answer(baseline_topic)
    improved_score = score_answer(improved_topic)

    result = {
        "question": QUESTION,
        "papers": [{"arxiv_id": p.arxiv_id, "title": p.title, "pdf_path": p.pdf_path} for p in papers],
        "baseline_topic_answer": baseline_topic,
        "improved_topic_answer": improved_topic,
        "baseline_score": baseline_score,
        "improved_score": improved_score,
        "delta": {
            "term_hits": improved_score["term_hits"] - baseline_score["term_hits"],
            "citation_hits": improved_score["citation_hits"] - baseline_score["citation_hits"],
            "length": improved_score["length"] - baseline_score["length"],
        },
    }

    out_path = pathlib.Path("/workspace/repos/paper-curator-agentic-rag/storage/eval_llm_rl/result.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    res = asyncio.run(run_eval())
    print(json.dumps({
        "papers": len(res["papers"]),
        "baseline_score": res["baseline_score"],
        "improved_score": res["improved_score"],
        "delta": res["delta"],
    }, indent=2, ensure_ascii=False))
