"""Custom RAG (Retrieval-Augmented Generation) implementation.

This module now supports two retrieval modes:
- baseline: legacy top-k cosine retrieval
- improved: hybrid rerank (semantic + keyword), MMR diversification,
  temporal/source-aware weighting, metadata-rich citations, and two-step retrieval.
"""
from __future__ import annotations

import math
import pathlib
import re
from datetime import datetime
from typing import Any, Optional

import numpy as np
from fastapi import HTTPException
from openai import AsyncOpenAI

import db


# ---------------------------------------------------------------------------
# Retrieval mode constants
# ---------------------------------------------------------------------------

RAG_MODE_BASELINE = "baseline"
RAG_MODE_IMPROVED = "improved"


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def extract_pdf_text(pdf_path: pathlib.Path) -> str:
    """Extract full text from a PDF using pymupdf.

    Returns concatenated text from all pages.
    """
    import fitz  # pymupdf

    doc = fitz.open(str(pdf_path))
    pages: list[str] = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    text = "\n".join(pages)
    # PostgreSQL text columns cannot store NUL (0x00) characters
    return text.replace("\x00", "")


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_pdf_text(text: str) -> str:
    """Clean text extracted from PDFs, decoding font escape sequences."""
    if not text:
        return text

    def decode_uni_sequence(match: re.Match) -> str:
        hex_str = match.group(1)
        try:
            code_point = int(hex_str, 16)
            if 0x20 <= code_point <= 0x10FFFF:
                return chr(code_point)
            return ""
        except (ValueError, OverflowError):
            return ""

    cleaned = re.sub(r"/uni([0-9a-fA-F]{8})", decode_uni_sequence, text)

    def decode_u_sequence(match: re.Match) -> str:
        hex_str = match.group(1)
        try:
            code_point = int(hex_str, 16)
            if 0x20 <= code_point <= 0x10FFFF:
                return chr(code_point)
            return ""
        except (ValueError, OverflowError):
            return ""

    cleaned = re.sub(r"/u([0-9a-fA-F]{4})", decode_u_sequence, cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_chars: int = 5000,
    overlap: int = 250,
) -> list[dict[str, Any]]:
    """Split *text* into overlapping character-based chunks.

    Returns a list of dicts with keys: text, chunk_index, char_start, char_end.
    """
    chunks: list[dict[str, Any]] = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        chunks.append(
            {
                "text": text[start:end],
                "chunk_index": idx,
                "char_start": start,
                "char_end": end,
            }
        )
        if end >= len(text):
            break
        start = end - overlap
        idx += 1
    return chunks


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

async def embed_texts(
    texts: list[str],
    client: AsyncOpenAI,
    model: str,
    batch_size: int = 20,
) -> list[list[float]]:
    """Embed a list of texts using the OpenAI-compatible embedding API."""
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = await client.embeddings.create(model=model, input=batch)
        batch_embs = sorted(response.data, key=lambda d: d.index)
        for item in batch_embs:
            all_embeddings.append(item.embedding)
    return all_embeddings


async def _embed_single(
    text: str,
    client: AsyncOpenAI,
    model: str,
) -> list[float]:
    """Embed a single text string and return its vector."""
    response = await client.embeddings.create(model=model, input=text)
    return response.data[0].embedding


# ---------------------------------------------------------------------------
# Indexing (chunk + embed + store)
# ---------------------------------------------------------------------------

async def index_paper_async(
    paper_id: int,
    pdf_path: str,
    embed_client: AsyncOpenAI,
    embed_model: str,
) -> dict[str, Any]:
    """Index a paper: extract text, chunk, embed, store in paper_chunks table.

    Also computes the document-level embedding (mean pooling of chunks) and
    persists it to papers.embedding.

    Returns {"indexed": True, "cached": bool, "chunks": int}.
    """
    from config import _get_rag_config

    if db.has_paper_chunks(paper_id):
        return {"indexed": True, "cached": True, "chunks": 0}

    config = _get_rag_config()
    full_text = extract_pdf_text(pathlib.Path(pdf_path))
    chunks = chunk_text(full_text, config["chunk_chars"], config["chunk_overlap"])

    if not chunks:
        return {"indexed": False, "cached": False, "chunks": 0}

    chunk_texts = [c["text"] for c in chunks]
    embeddings = await embed_texts(chunk_texts, embed_client, embed_model)

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb

    db.store_paper_chunks(paper_id, chunks)

    doc_embedding = np.mean(embeddings, axis=0).tolist()
    db.update_paper_embedding(paper_id, doc_embedding)

    return {"indexed": True, "cached": False, "chunks": len(chunks)}


# ---------------------------------------------------------------------------
# Improved retrieval internals
# ---------------------------------------------------------------------------

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{1,}")
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
    "what",
    "which",
    "where",
    "when",
    "how",
    "does",
    "about",
    "into",
    "using",
    "paper",
    "papers",
    "model",
    "models",
    "method",
    "methods",
}


def _extract_keywords(question: str, max_terms: int = 14) -> list[str]:
    q = question.lower()
    terms = WORD_RE.findall(q)
    filtered = [t for t in terms if t not in STOPWORDS and len(t) >= 3]

    # Domain expansions for LLM-RL queries
    expansions: list[str] = []
    if "grpo" in q:
        expansions += ["group", "relative", "policy", "optimization"]
    if "ppo" in q:
        expansions += ["proximal", "policy", "optimization", "clip", "kl"]
    if "dpo" in q:
        expansions += ["direct", "preference", "optimization"]
    if "rlhf" in q:
        expansions += ["reinforcement", "human", "feedback", "reward", "model"]
    if "sparse" in q or "length" in q:
        expansions += ["sparse", "length", "bias", "reward", "variance"]

    # preserve order + dedup
    seen = set()
    deduped = []
    for t in filtered + expansions:
        if t in seen:
            continue
        seen.add(t)
        deduped.append(t)
    return deduped[:max_terms]


def _lexical_overlap_score(text: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    lower = text.lower()
    hits = sum(1 for k in keywords if k in lower)
    return hits / max(1, len(keywords))


def _parse_year_from_paper(paper: Optional[dict[str, Any]]) -> Optional[int]:
    if not paper:
        return None
    published = paper.get("published_at")
    if not published:
        return None
    if isinstance(published, datetime):
        return published.year
    s = str(published)
    m = re.match(r"(\d{4})", s)
    return int(m.group(1)) if m else None


def _recency_score(year: Optional[int], now_year: Optional[int] = None, half_life_years: float = 4.0) -> float:
    if year is None:
        return 0.5
    if now_year is None:
        now_year = datetime.utcnow().year
    age = max(0, now_year - year)
    return math.exp(-math.log(2) * (age / max(0.5, half_life_years)))


def _source_quality_score(paper: Optional[dict[str, Any]]) -> float:
    if not paper:
        return 0.6
    arxiv_id = str(paper.get("arxiv_id") or "")
    if arxiv_id.startswith("local_"):
        return 0.75
    return 1.0


def _normalize_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-9:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _cosine(a: list[float], b: list[float]) -> float:
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    denom = float(np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
    if denom <= 1e-9:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


def _mmr_select(
    candidates: list[dict[str, Any]],
    query_embedding: list[float],
    k: int,
    lambda_mult: float = 0.75,
) -> list[dict[str, Any]]:
    if len(candidates) <= k:
        return candidates

    selected: list[dict[str, Any]] = []
    remaining = candidates.copy()

    # pick best first
    remaining.sort(key=lambda c: c.get("final_score", 0.0), reverse=True)
    selected.append(remaining.pop(0))

    while remaining and len(selected) < k:
        best_idx = 0
        best_mmr = -1e9
        for i, cand in enumerate(remaining):
            relevance = cand.get("final_score", 0.0)
            cand_emb = cand.get("embedding")
            if not cand_emb:
                diversity_penalty = 0.0
            else:
                max_sim_to_selected = 0.0
                for s in selected:
                    s_emb = s.get("embedding")
                    if s_emb:
                        max_sim_to_selected = max(max_sim_to_selected, _cosine(cand_emb, s_emb))
                diversity_penalty = max_sim_to_selected

            mmr = lambda_mult * relevance - (1 - lambda_mult) * diversity_penalty
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = i

        selected.append(remaining.pop(best_idx))

    return selected


def _build_chunk_citation(chunk: dict[str, Any], paper: Optional[dict[str, Any]]) -> str:
    paper_title = (paper or {}).get("title") or "Unknown paper"
    arxiv_id = (paper or {}).get("arxiv_id") or "unknown"
    chunk_index = chunk.get("chunk_index")
    char_start = chunk.get("char_start")
    char_end = chunk.get("char_end")
    year = _parse_year_from_paper(paper)
    year_part = f", {year}" if year else ""
    return f"{paper_title} [{arxiv_id}{year_part}] chunk#{chunk_index} chars({char_start}-{char_end})"


async def retrieve_paper_evidence_async(
    question: str,
    paper_id: int,
    embed_client: AsyncOpenAI,
    embed_model: str,
    evidence_k: int,
    mode: str = RAG_MODE_IMPROVED,
) -> list[dict[str, Any]]:
    """Two-step retrieval for a single paper.

    Step 1 (search): retrieve broad semantic candidates.
    Step 2 (get/rerank): rerank with hybrid features and diversify via MMR.
    """
    q_emb = await _embed_single(question, embed_client, embed_model)

    if mode == RAG_MODE_BASELINE:
        rows = db.search_paper_chunks(paper_id, q_emb, top_k=evidence_k)
        return [
            {
                "id": r.get("id"),
                "text": r["text"],
                "score": float(r.get("similarity") or 0.0),
                "semantic_score": float(r.get("similarity") or 0.0),
                "keyword_score": None,
                "temporal_score": None,
                "source_score": None,
                "final_score": float(r.get("similarity") or 0.0),
                "chunk_index": r.get("chunk_index"),
                "char_start": r.get("char_start"),
                "char_end": r.get("char_end"),
                "citation": None,
                "mode": RAG_MODE_BASELINE,
            }
            for r in rows
        ]

    # Improved mode
    candidate_multiplier = 4
    candidate_k = max(evidence_k * candidate_multiplier, evidence_k)
    rows = db.search_paper_chunks(paper_id, q_emb, top_k=candidate_k)
    if not rows:
        return []

    paper = db.get_paper_by_id(paper_id)
    keywords = _extract_keywords(question)
    semantic_raw = [float(r.get("similarity") or 0.0) for r in rows]
    keyword_raw = [_lexical_overlap_score(r.get("text") or "", keywords) for r in rows]

    semantic_scores = _normalize_scores(semantic_raw)
    keyword_scores = _normalize_scores(keyword_raw)

    recency = _recency_score(_parse_year_from_paper(paper))
    source_quality = _source_quality_score(paper)

    # final = semantic + lexical + temporal/source priors
    candidates: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        final_score = (
            0.56 * semantic_scores[i]
            + 0.32 * keyword_scores[i]
            + 0.08 * recency
            + 0.04 * source_quality
        )
        candidates.append(
            {
                "id": r.get("id"),
                "text": r.get("text") or "",
                "score": final_score,
                "semantic_score": semantic_scores[i],
                "keyword_score": keyword_scores[i],
                "temporal_score": recency,
                "source_score": source_quality,
                "final_score": final_score,
                "chunk_index": r.get("chunk_index"),
                "char_start": r.get("char_start"),
                "char_end": r.get("char_end"),
                "embedding": None,  # optional; kept for future dense MMR
                "citation": _build_chunk_citation(r, paper),
                "mode": RAG_MODE_IMPROVED,
            }
        )

    candidates.sort(key=lambda x: x["final_score"], reverse=True)

    # lightweight diversification using lexical fingerprint as anti-dup signal
    for c in candidates:
        c["embedding"] = [
            c["semantic_score"],
            c["keyword_score"],
            c["temporal_score"],
            c["source_score"],
        ]

    selected = _mmr_select(candidates, q_emb, k=evidence_k, lambda_mult=0.78)
    for c in selected:
        c.pop("embedding", None)
    return selected


# ---------------------------------------------------------------------------
# RAG answer
# ---------------------------------------------------------------------------

async def rag_answer_async(
    question: str,
    text: Optional[str] = None,
    pdf_path: Optional[str] = None,
    arxiv_id: Optional[str] = None,
    llm_client: Optional[AsyncOpenAI] = None,
    llm_model: Optional[str] = None,
    embed_client: Optional[AsyncOpenAI] = None,
    embed_model: Optional[str] = None,
    return_details: bool = False,
    rag_mode: str = RAG_MODE_IMPROVED,
) -> "str | dict[str, Any]":
    """Full RAG pipeline: retrieve evidence -> answer.

    Modes:
    - baseline: legacy top-k cosine retrieval
    - improved: hybrid rerank + MMR + temporal/source weighting + citations
    """
    from config import _get_rag_config

    assert llm_client is not None, "llm_client required"
    assert embed_client is not None, "embed_client required"

    config = _get_rag_config()
    evidence_k: int = config["evidence_k"]
    mode = rag_mode if rag_mode in {RAG_MODE_BASELINE, RAG_MODE_IMPROVED} else RAG_MODE_IMPROVED

    paper_id: Optional[int] = None
    if arxiv_id:
        paper = db.get_paper_by_arxiv_id(arxiv_id)
        if paper:
            paper_id = paper["id"]

    retrieved: list[dict[str, Any]] = []

    # DB-backed retrieval path
    if paper_id and db.has_paper_chunks(paper_id):
        retrieved = await retrieve_paper_evidence_async(
            question=question,
            paper_id=paper_id,
            embed_client=embed_client,
            embed_model=embed_model,
            evidence_k=evidence_k,
            mode=mode,
        )
    else:
        # Ephemeral fallback: preserve legacy behavior (semantic only)
        if pdf_path:
            full_text = extract_pdf_text(pathlib.Path(pdf_path))
        elif text:
            full_text = text
        else:
            raise HTTPException(
                status_code=400,
                detail="Provide text or pdf_path for RAG query.",
            )

        text_chunks = chunk_text(full_text, config["chunk_chars"], config["chunk_overlap"])
        chunk_texts = [c["text"] for c in text_chunks]
        all_embeddings = await embed_texts(chunk_texts + [question], embed_client, embed_model)
        chunk_embeddings = all_embeddings[: len(chunk_texts)]
        q_emb = all_embeddings[-1]

        retrieved = _retrieve_in_memory_evidence(
            question=question,
            chunk_texts=chunk_texts,
            chunk_embeddings=chunk_embeddings,
            query_embedding=q_emb,
            evidence_k=evidence_k,
            mode=mode,
        )

        # Persist chunks if paper exists in DB
        if paper_id:
            for chunk, emb in zip(text_chunks, chunk_embeddings):
                chunk["embedding"] = emb
            db.store_paper_chunks(paper_id, text_chunks)
            doc_emb = np.mean(chunk_embeddings, axis=0).tolist()
            db.update_paper_embedding(paper_id, doc_emb)

    context_parts = []
    for c in retrieved:
        text_part = c.get("text", "")
        citation = c.get("citation")
        if citation and mode == RAG_MODE_IMPROVED:
            context_parts.append(f"{text_part}\n\n[Source] {citation}")
        else:
            context_parts.append(text_part)

    context = "\n\n---\n\n".join(context_parts)
    answer = await _generate_answer(question, context, llm_client, llm_model)

    if return_details:
        return {
            "answer": answer,
            "chunks_retrieved": len(retrieved),
            "mode": mode,
            "chunks": [
                {
                    "text": clean_pdf_text(c.get("text", "")[:500]),
                    "text_raw_sample": c.get("text", "")[:500],
                    "score": c.get("score"),
                    "semantic_score": c.get("semantic_score"),
                    "keyword_score": c.get("keyword_score"),
                    "temporal_score": c.get("temporal_score"),
                    "source_score": c.get("source_score"),
                    "chunk_index": c.get("chunk_index"),
                    "char_start": c.get("char_start"),
                    "char_end": c.get("char_end"),
                    "citation": c.get("citation"),
                }
                for c in retrieved
            ],
        }
    return answer


# ---------------------------------------------------------------------------
# Ensure paper has a document-level embedding
# ---------------------------------------------------------------------------

async def ensure_paper_embedding(
    arxiv_id: str,
    embed_client: Optional[AsyncOpenAI] = None,
    embed_model: Optional[str] = None,
) -> bool:
    """Make sure papers.embedding is populated."""
    paper = db.get_paper_by_arxiv_id(arxiv_id)
    if not paper:
        return False

    if paper.get("embedding") is not None:
        return True

    paper_id = paper["id"]

    if db.has_paper_chunks(paper_id):
        chunks = db.get_paper_chunks(paper_id)
        embeddings = [c["embedding"] for c in chunks if c.get("embedding") is not None]
        if embeddings:
            doc_emb = np.mean(embeddings, axis=0).tolist()
            db.update_paper_embedding(paper_id, doc_emb)
            return True

    pdf_path = paper.get("pdf_path")
    if pdf_path and pathlib.Path(pdf_path).exists() and embed_client and embed_model:
        result = await index_paper_async(paper_id, pdf_path, embed_client, embed_model)
        return result.get("indexed", False)

    return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _in_memory_search(
    chunk_texts: list[str],
    chunk_embeddings: list[list[float]],
    query_embedding: list[float],
    top_k: int,
) -> list[dict[str, Any]]:
    """Return top-k chunks by cosine similarity (numpy, in-memory)."""
    if not chunk_embeddings:
        return []

    emb_array = np.array(chunk_embeddings, dtype=np.float32)
    q_array = np.array(query_embedding, dtype=np.float32)

    norms = np.linalg.norm(emb_array, axis=1) * np.linalg.norm(q_array)
    norms = np.maximum(norms, 1e-10)
    sims = np.dot(emb_array, q_array) / norms

    top_indices = np.argsort(sims)[-top_k:][::-1]
    return [{"text": chunk_texts[i], "score": float(sims[i]), "idx": int(i)} for i in top_indices]


def _retrieve_in_memory_evidence(
    question: str,
    chunk_texts: list[str],
    chunk_embeddings: list[list[float]],
    query_embedding: list[float],
    evidence_k: int,
    mode: str,
) -> list[dict[str, Any]]:
    """Retrieve evidence from in-memory chunks for baseline/improved modes."""
    if mode == RAG_MODE_BASELINE:
        hits = _in_memory_search(chunk_texts, chunk_embeddings, query_embedding, evidence_k)
        return [
            {
                "text": h["text"],
                "score": h["score"],
                "semantic_score": h["score"],
                "keyword_score": None,
                "temporal_score": None,
                "source_score": None,
                "final_score": h["score"],
                "chunk_index": h.get("idx"),
                "char_start": None,
                "char_end": None,
                "citation": None,
                "mode": RAG_MODE_BASELINE,
            }
            for h in hits
        ]

    if not chunk_embeddings:
        return []

    emb_array = np.array(chunk_embeddings, dtype=np.float32)
    q_array = np.array(query_embedding, dtype=np.float32)
    norms = np.linalg.norm(emb_array, axis=1) * np.linalg.norm(q_array)
    norms = np.maximum(norms, 1e-10)
    sims = np.dot(emb_array, q_array) / norms

    candidate_k = max(evidence_k * 4, evidence_k)
    candidate_idx = np.argsort(sims)[-candidate_k:][::-1]

    keywords = _extract_keywords(question)
    semantic_raw = [float(sims[i]) for i in candidate_idx]
    keyword_raw = [_lexical_overlap_score(chunk_texts[i], keywords) for i in candidate_idx]
    semantic_scores = _normalize_scores(semantic_raw)
    keyword_scores = _normalize_scores(keyword_raw)

    candidates: list[dict[str, Any]] = []
    for rank_i, idx in enumerate(candidate_idx):
        final_score = 0.58 * semantic_scores[rank_i] + 0.42 * keyword_scores[rank_i]
        candidates.append(
            {
                "text": chunk_texts[int(idx)],
                "score": final_score,
                "semantic_score": semantic_scores[rank_i],
                "keyword_score": keyword_scores[rank_i],
                "temporal_score": None,
                "source_score": None,
                "final_score": final_score,
                "chunk_index": int(idx),
                "char_start": None,
                "char_end": None,
                "citation": f"ephemeral chunk#{int(idx)}",
                "mode": RAG_MODE_IMPROVED,
                "embedding": [semantic_scores[rank_i], keyword_scores[rank_i], 0.0, 0.0],
            }
        )

    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    selected = _mmr_select(candidates, query_embedding, k=evidence_k, lambda_mult=0.8)
    for s in selected:
        s.pop("embedding", None)
    return selected


async def _generate_answer(
    question: str,
    context: str,
    llm_client: AsyncOpenAI,
    llm_model: str,
    max_context_chars: int = 18000,
) -> str:
    """Call the LLM to answer *question* given retrieved *context*."""
    if len(context) > max_context_chars:
        context = context[:max_context_chars] + "\n\n[Context truncated for length]"

    prompt = (
        "Answer the following question based ONLY on the provided context "
        "from research paper evidence. If the context does not contain enough "
        "information, explicitly say so.\n\n"
        "When source lines are provided, cite them inline like [Source].\n"
        "Prefer concrete comparisons, not generic summaries.\n"
        "If the question compares methods, include: (1) stability, (2) trade-offs, (3) when to use each.\n\n"
        "Context:\n"
        + context
        + "\n\nQuestion: "
        + question
        + "\n\nAnswer:"
    )

    prompt_tokens_est = len(prompt) // 4
    max_tokens = max(256, 8192 - prompt_tokens_est - 100)
    max_tokens = min(max_tokens, 2000)

    response = await llm_client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    return (response.choices[0].message.content or "").strip()
