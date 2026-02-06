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


def clean_pdf_text(text: str) -> str:
    """Clean text extracted from PDFs, attempting to decode font escape sequences.
    
    Some PDFs use embedded fonts where glyphs are stored as /uniXXXXXXXX sequences.
    This function attempts to decode them to actual unicode characters.
    """
    if not text:
        return text
    
    def decode_uni_sequence(match):
        """Decode a /uniXXXXXXXX sequence to unicode character."""
        hex_str = match.group(1)
        try:
            # Try to convert 8-digit hex to unicode
            code_point = int(hex_str, 16)
            # Only decode if it's a printable character
            if 0x20 <= code_point <= 0x10FFFF:
                return chr(code_point)
            return ""  # Skip control characters
        except (ValueError, OverflowError):
            return ""  # Remove if can't decode
    
    # Try to decode /uni escape sequences
    cleaned = re.sub(r'/uni([0-9a-fA-F]{8})', decode_uni_sequence, text)
    
    # Also handle standard /uXXXX format
    def decode_u_sequence(match):
        hex_str = match.group(1)
        try:
            code_point = int(hex_str, 16)
            if 0x20 <= code_point <= 0x10FFFF:
                return chr(code_point)
            return ""
        except (ValueError, OverflowError):
            return ""
    
    cleaned = re.sub(r'/u([0-9a-fA-F]{4})', decode_u_sequence, cleaned)
    
    # Clean up excessive whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    return cleaned.strip()
    
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
    """Ensure a paper has a document-level embedding.
    
    Tries in order:
    1. Return True if embedding already exists
    2. Extract from cached PaperQA index
    3. Generate proxy embedding using text similarity with existing papers
    """
    paper = db.get_paper_by_arxiv_id(arxiv_id)
    if not paper:
        return False

    if paper.get("embedding") is not None:
        return True

    # Try to extract from cached PaperQA index
    docs = load_paperqa_index(arxiv_id)
    if docs:
        doc_embedding = extract_document_embedding_from_paperqa(docs)
        if doc_embedding:
            db.update_paper_embedding(paper["id"], doc_embedding)
            return True
    
    # Fallback: generate proxy embedding using text similarity
    proxy_embedding = generate_proxy_embedding(paper)
    if proxy_embedding:
        db.update_paper_embedding(paper["id"], proxy_embedding)
        print(f"  Generated proxy embedding for {arxiv_id}")
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
    return_details: bool = False,
) -> str | dict[str, Any]:
    """Query PaperQA2 for an answer (async). Uses cached index if available.
    
    Args:
        return_details: If True, returns dict with answer and chunk details for debugging
    """
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
    
    if return_details:
        # Extract chunk details from PaperQA result
        chunks_info = []
        if hasattr(result, "contexts") and result.contexts:
            for ctx in result.contexts:
                # Get text content - handle PaperQA Text objects
                text_content = ""
                if hasattr(ctx, "text"):
                    text_obj = ctx.text
                    # Handle Text objects with .text attribute
                    if hasattr(text_obj, "text"):
                        raw_text = str(text_obj.text)
                    else:
                        raw_text = str(text_obj)
                else:
                    raw_text = str(ctx)
                
                # Clean the text to decode font escape sequences
                cleaned_text = clean_pdf_text(raw_text)
                text_content = cleaned_text[:500] if cleaned_text else raw_text[:500]
                
                chunk_info = {
                    "text": text_content,
                    "text_raw_sample": raw_text[:200] if raw_text != cleaned_text else None,  # Show raw if different
                    "score": getattr(ctx, "score", None),
                }
                # Try to get chunk ID/name
                if hasattr(ctx, "name"):
                    chunk_info["name"] = str(ctx.name) if ctx.name else None
                if hasattr(ctx, "doc"):
                    doc = ctx.doc
                    chunk_info["doc_name"] = str(getattr(doc, "docname", "")) if doc else None
                chunks_info.append(chunk_info)
        
        return {
            "answer": answer,
            "chunks_retrieved": len(chunks_info),
            "chunks": chunks_info,
        }
    
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


def generate_proxy_embedding(paper: dict[str, Any]) -> Optional[list[float]]:
    """Generate a proxy embedding for a paper using text similarity with existing papers.
    
    This is a fallback for when the embedding service is unavailable.
    It computes TF-IDF similarity between the paper's text and existing papers,
    then creates a weighted average of similar papers' embeddings.
    """
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Get all papers with embeddings
    papers_with_embeddings = [p for p in db.get_all_papers() if p.get("embedding") is not None and p["id"] != paper.get("id")]
    
    if len(papers_with_embeddings) < 5:
        return None
    
    # Prepare text for TF-IDF (use title + abstract + summary)
    def get_paper_text(p: dict) -> str:
        parts = []
        if p.get("title"):
            parts.append(p["title"])
        if p.get("abstract"):
            parts.append(p["abstract"])
        if p.get("summary"):
            parts.append(p["summary"][:500])  # Truncate summary
        return " ".join(parts)
    
    target_text = get_paper_text(paper)
    if not target_text:
        return None
    
    corpus_texts = [get_paper_text(p) for p in papers_with_embeddings]
    
    # Compute TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform([target_text] + corpus_texts)
    except ValueError:
        return None
    
    # Compute similarity between target (index 0) and all others
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    # Get top-10 most similar papers
    top_k = min(10, len(similarities))
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    top_similarities = similarities[top_indices]
    
    # Normalize similarities to weights
    weights = top_similarities / (top_similarities.sum() + 1e-8)
    
    # Compute weighted average of embeddings
    embeddings = []
    for idx in top_indices:
        emb = papers_with_embeddings[idx].get("embedding")
        if emb:
            embeddings.append(np.array(emb, dtype=np.float32))
    
    if not embeddings:
        return None
    
    embeddings_array = np.stack(embeddings)
    proxy_embedding = np.average(embeddings_array, axis=0, weights=weights[:len(embeddings)])
    
    # L2 normalize
    norm = np.linalg.norm(proxy_embedding)
    if norm > 0:
        proxy_embedding = proxy_embedding / norm
    
    return proxy_embedding.tolist()
