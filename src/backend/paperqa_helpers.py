"""PaperQA2 indexing and query helpers."""
from __future__ import annotations

import asyncio
import hashlib
import os
import pathlib
import pickle
import re
from typing import Any, Optional

import numpy as np
from fastapi import HTTPException
from paperqa import Doc, Docs
from paperqa.settings import (
    AnswerSettings,
    MultimodalOptions,
    ParsingSettings,
    PromptSettings,
    Settings,
    make_default_litellm_model_list_settings,
)
from paperqa.readers import read_doc

import db
from config import _get_paperqa_config


def get_paperqa_index_path(arxiv_id: str) -> pathlib.Path:
    """Get path to stored PaperQA2 index for a paper."""
    index_dir = pathlib.Path("storage/paperqa_index")
    index_dir.mkdir(parents=True, exist_ok=True)
    safe_id = arxiv_id.replace("/", "_").replace(".", "_")
    return index_dir / f"{safe_id}.pkl"


def build_paperqa_settings(
    llm_model: str,
    embed_model: str,
    llm_base_url: str,
    embed_base_url: str,
    api_key: str,
) -> Settings:
    """Build PaperQA2 Settings with given models and endpoints."""
    pqa_config = _get_paperqa_config()

    llm_config = make_default_litellm_model_list_settings(llm_model)
    llm_config["model_list"][0]["litellm_params"].update(
        {"api_base": llm_base_url, "api_key": api_key}
    )
    summary_llm_config = make_default_litellm_model_list_settings(llm_model)
    summary_llm_config["model_list"][0]["litellm_params"].update(
        {"api_base": llm_base_url, "api_key": api_key}
    )

    reader_config = {
        "chunk_chars": pqa_config["chunk_chars"],
        "overlap": pqa_config["chunk_overlap"],
    }
    parsing_settings = ParsingSettings(
        reader_config=reader_config,
        use_doc_details=pqa_config["use_doc_details"],
        multimodal=MultimodalOptions.OFF,
    )
    answer_settings = AnswerSettings(
        evidence_k=pqa_config["evidence_k"],
        evidence_summary_length=pqa_config["evidence_summary_length"],
        evidence_skip_summary=pqa_config["evidence_skip_summary"],
        evidence_relevance_score_cutoff=pqa_config["evidence_relevance_score_cutoff"],
    )
    prompt_settings = PromptSettings(use_json=False)

    return Settings(
        llm=llm_model,
        llm_config=llm_config,
        summary_llm=llm_model,
        summary_llm_config=summary_llm_config,
        embedding=embed_model,
        embedding_config={
            "kwargs": {
                "api_base": embed_base_url,
                "api_key": api_key,
                "encoding_format": "float",
            },
        },
        parsing=parsing_settings,
        answer=answer_settings,
        prompts=prompt_settings,
    )


async def index_pdf_for_paperqa_async(
    pdf_path: str,
    arxiv_id: str,
    llm_base_url: str,
    embed_base_url: str,
    api_key: str,
    llm_model: str,
    embed_model: str,
) -> Docs:
    """Index a PDF and persist the Docs object for later QA queries (async)."""
    os.environ["OPENAI_API_BASE"] = llm_base_url
    os.environ["OPENAI_API_KEY"] = api_key

    docs = Docs()
    settings = build_paperqa_settings(
        llm_model, embed_model, llm_base_url, embed_base_url, api_key
    )
    await docs.aadd(pdf_path, settings=settings)

    index_path = get_paperqa_index_path(arxiv_id)
    with open(index_path, "wb") as f:
        pickle.dump(docs, f)

    doc_embedding = extract_document_embedding_from_paperqa(docs)
    if doc_embedding:
        paper = db.get_paper_by_arxiv_id(arxiv_id)
        if paper:
            db.update_paper_embedding(paper["id"], doc_embedding)

    return docs


def index_pdf_for_paperqa(
    pdf_path: str,
    arxiv_id: str,
    llm_base_url: str,
    embed_base_url: str,
    api_key: str,
    llm_model: str,
    embed_model: str,
) -> Docs:
    """Index a PDF and persist the Docs object for later QA queries (sync wrapper)."""
    return asyncio.run(index_pdf_for_paperqa_async(
        pdf_path, arxiv_id, llm_base_url, embed_base_url, api_key, llm_model, embed_model
    ))


def load_paperqa_index(arxiv_id: str) -> Optional[Docs]:
    """Load persisted PaperQA2 Docs if exists."""
    index_path = get_paperqa_index_path(arxiv_id)
    if not index_path.exists():
        return None
    try:
        with open(index_path, "rb") as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, EOFError, AttributeError, ValueError):
        return None


def extract_document_embedding_from_paperqa(docs: Docs) -> Optional[list[float]]:
    """Extract document-level embedding by mean pooling of PaperQA2 chunk embeddings.
    
    PaperQA2 stores text chunks with embeddings at docs.texts (top-level),
    not nested under individual documents.
    """
    all_chunk_embeddings: list[list[float]] = []

    # Primary source: docs.texts contains Text objects with embeddings
    if hasattr(docs, "texts") and docs.texts:
        for text_chunk in docs.texts:
            if hasattr(text_chunk, "embedding") and text_chunk.embedding is not None:
                emb = text_chunk.embedding
                if isinstance(emb, np.ndarray):
                    all_chunk_embeddings.append(emb.tolist())
                elif isinstance(emb, list):
                    all_chunk_embeddings.append(emb)

    # Fallback: check docs.docs for document-level embeddings (older versions)
    if not all_chunk_embeddings and hasattr(docs, "docs") and docs.docs:
        docs_iter = docs.docs.items() if isinstance(docs.docs, dict) else enumerate(docs.docs)
        for doc_key_or_doc in docs_iter:
            doc = doc_key_or_doc[1] if isinstance(docs.docs, dict) else doc_key_or_doc
            if hasattr(doc, "embedding") and doc.embedding is not None:
                emb = doc.embedding
                if isinstance(emb, np.ndarray):
                    all_chunk_embeddings.append(emb.tolist())
                elif isinstance(emb, list):
                    all_chunk_embeddings.append(emb)

    if not all_chunk_embeddings:
        return None

    mean_embedding = np.mean(all_chunk_embeddings, axis=0).tolist()
    return mean_embedding


async def ensure_paper_embedding(arxiv_id: str) -> bool:
    """Ensure a paper has a document-level embedding."""
    paper = db.get_paper_by_arxiv_id(arxiv_id)
    if not paper:
        return False

    if paper.get("embedding") is not None:
        return True

    docs = load_paperqa_index(arxiv_id)
    if docs:
        doc_embedding = extract_document_embedding_from_paperqa(docs)
        if doc_embedding:
            db.update_paper_embedding(paper["id"], doc_embedding)
            return True

    return False


async def paperqa_answer_async(
    text: Optional[str],
    pdf_path: Optional[str],
    question: str,
    llm_base_url: str,
    embed_base_url: str,
    api_key: str,
    llm_model: str,
    embed_model: str,
    arxiv_id: Optional[str] = None,
) -> str:
    """Query PaperQA2 for an answer (async). Uses cached index if available."""
    assert text or pdf_path, "Provide text or pdf_path."
    os.environ["OPENAI_API_BASE"] = llm_base_url
    os.environ["OPENAI_API_KEY"] = api_key

    docs: Optional[Docs] = None
    if arxiv_id:
        docs = load_paperqa_index(arxiv_id)

    if docs is None:
        docs = Docs()
        content_path: Optional[pathlib.Path] = None
        if pdf_path:
            content_path = pathlib.Path(pdf_path)
        elif text is not None:
            inputs_dir = pathlib.Path("storage/inputs")
            inputs_dir.mkdir(parents=True, exist_ok=True)
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            content_path = inputs_dir / f"paperqa_{text_hash}.txt"
            content_path.write_text(text, encoding="utf-8")
        if content_path is None or not content_path.exists():
            raise HTTPException(status_code=404, detail="Content path not found.")

        settings = build_paperqa_settings(
            llm_model, embed_model, llm_base_url, embed_base_url, api_key
        )
        await docs.aadd(str(content_path), settings=settings)

        if arxiv_id:
            index_path = get_paperqa_index_path(arxiv_id)
            with open(index_path, "wb") as f:
                pickle.dump(docs, f)

            doc_embedding = extract_document_embedding_from_paperqa(docs)
            if doc_embedding:
                paper = db.get_paper_by_arxiv_id(arxiv_id)
                if paper:
                    db.update_paper_embedding(paper["id"], doc_embedding)

    settings = build_paperqa_settings(
        llm_model, embed_model, llm_base_url, embed_base_url, api_key
    )
    result = await docs.aquery(question, settings=settings)
    answer = str(result.answer) if hasattr(result, "answer") else str(result)
    answer = clean_paperqa_citations(answer)
    return answer


def paperqa_answer(
    text: Optional[str],
    pdf_path: Optional[str],
    question: str,
    llm_base_url: str,
    embed_base_url: str,
    api_key: str,
    llm_model: str,
    embed_model: str,
    arxiv_id: Optional[str] = None,
) -> str:
    """Query PaperQA2 for an answer (sync wrapper)."""
    return asyncio.run(paperqa_answer_async(
        text, pdf_path, question, llm_base_url, embed_base_url, api_key,
        llm_model, embed_model, arxiv_id
    ))


def clean_paperqa_citations(text: str) -> str:
    """Remove PaperQA2 citation markers from text."""
    pattern = r"\s*\([A-Za-z0-9_]+\s+pages?\s+[\d,\-–\s]+(?:,\s*[A-Za-z0-9_]+\s+pages?\s+[\d,\-–\s]+)*\)"
    cleaned = re.sub(pattern, "", text)
    cleaned = re.sub(r"  +", " ", cleaned)
    return cleaned.strip()


async def paperqa_extract_pdf_async(pdf_path: pathlib.Path) -> dict[str, Any]:
    """Extract text from PDF using PaperQA2's native parser (async version)."""
    pqa_config = _get_paperqa_config()
    reader_config = {
        "chunk_chars": pqa_config["chunk_chars"],
        "overlap": pqa_config["chunk_overlap"],
    }
    parsing_settings = ParsingSettings(
        reader_config=reader_config,
        use_doc_details=pqa_config["use_doc_details"],
        multimodal=MultimodalOptions.OFF,
    )
    settings = Settings(parsing=parsing_settings)

    doc = Doc(docname=pdf_path.stem, dockey=pdf_path.stem, citation="Local PDF")
    parsed_text = await read_doc(
        str(pdf_path),
        doc,
        parsed_text_only=True,
        parse_pdf=settings.parsing.parse_pdf,
        **settings.parsing.reader_config,
    )
    text = parsed_text.reduce_content()
    return {"text": text, "parser": "paperqa"}


def paperqa_extract_pdf(pdf_path: pathlib.Path) -> dict[str, Any]:
    """Extract text from PDF using PaperQA2's native parser (sync wrapper)."""
    return asyncio.run(paperqa_extract_pdf_async(pdf_path))
