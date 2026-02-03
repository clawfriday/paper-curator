"""Configuration and prompt loading helpers."""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
from functools import lru_cache
from typing import Any

import yaml
from fastapi import HTTPException

_prompts_cache: dict[str, Any] | None = None
_prompts_mtime: float = 0


@lru_cache(maxsize=4)
def _load_config_cached(config_mtime: float) -> dict[str, Any]:
    config_path = pathlib.Path("config/config.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return config


def _load_config() -> dict[str, Any]:
    config_path = pathlib.Path("config/config.yaml")
    if not config_path.exists():
        raise HTTPException(status_code=500, detail="Config file not found: config/config.yaml")
    mtime = config_path.stat().st_mtime
    return _load_config_cached(mtime)


def _load_prompts() -> dict[str, Any]:
    """Load prompts from JSON file with caching based on file modification time."""
    global _prompts_cache, _prompts_mtime

    prompts_path = pathlib.Path(__file__).parent / "prompts" / "prompts.json"
    if not prompts_path.exists():
        raise HTTPException(status_code=500, detail="Prompts file not found: prompts/prompts.json")

    current_mtime = prompts_path.stat().st_mtime
    if _prompts_cache is None or current_mtime > _prompts_mtime:
        _prompts_cache = json.loads(prompts_path.read_text(encoding="utf-8"))
        _prompts_mtime = current_mtime

    return _prompts_cache


def get_prompt(prompt_id: str, **kwargs: Any) -> str:
    """Get a prompt by ID and format it with the provided variables."""
    prompts = _load_prompts()
    if prompt_id not in prompts:
        raise HTTPException(status_code=500, detail=f"Prompt not found: {prompt_id}")

    template = prompts[prompt_id]["template"]
    return template.format(**kwargs)


def _load_prompt() -> tuple[str, str, str]:
    """Load paper_summary prompt from JSON prompts file."""
    prompts = _load_prompts()
    if "paper_summary" not in prompts:
        raise HTTPException(status_code=500, detail="Prompt 'paper_summary' not found in prompts.json")

    prompt_data = prompts["paper_summary"]
    prompt_id = prompt_data.get("id", "paper_summary_v1")
    prompt_body = prompt_data["template"]
    prompt_hash = hashlib.sha256(prompt_body.encode("utf-8")).hexdigest()
    return prompt_id, prompt_hash, prompt_body


def _convert_localhost_for_docker(url: str) -> str:
    """Convert localhost URLs to host.docker.internal when running in Docker."""
    if not url:
        return url
    in_docker = os.environ.get("DATABASE_URL", "").startswith("postgresql://")
    if in_docker:
        url = url.replace("://localhost:", "://host.docker.internal:")
        url = url.replace("://127.0.0.1:", "://host.docker.internal:")
    return url


def _get_endpoint_config() -> dict[str, str]:
    """Get endpoint configuration."""
    config = _load_config()
    endpoints = config.get("endpoints", {})
    llm_url = endpoints.get("llm_base_url", config.get("openai_api_base", ""))
    embed_url = endpoints.get("embedding_base_url", config.get("openai_api_base3", ""))
    return {
        "llm_base_url": _convert_localhost_for_docker(llm_url),
        "embedding_base_url": _convert_localhost_for_docker(embed_url),
        "api_key": endpoints.get("api_key", config.get("openai_api_key", "local-key")),
    }


def _get_paperqa_config() -> dict[str, Any]:
    """Get PaperQA2 configuration."""
    config = _load_config()
    pqa = config.get("paperqa", {})
    return {
        "chunk_chars": int(pqa.get("chunk_chars", config.get("paperqa_chunk_chars", 5000))),
        "chunk_overlap": int(pqa.get("chunk_overlap", config.get("paperqa_chunk_overlap", 250))),
        "use_doc_details": bool(pqa.get("use_doc_details", config.get("paperqa_use_doc_details", True))),
        "evidence_k": int(pqa.get("evidence_k", config.get("paperqa_evidence_k", 10))),
        "evidence_summary_length": str(pqa.get("evidence_summary_length", config.get("paperqa_evidence_summary_length", "about 100 words"))),
        "evidence_skip_summary": bool(pqa.get("evidence_skip_summary", config.get("paperqa_evidence_skip_summary", False))),
        "evidence_relevance_score_cutoff": float(pqa.get("evidence_relevance_score_cutoff", config.get("paperqa_evidence_relevance_score_cutoff", 1))),
    }


def _get_ui_config() -> dict[str, Any]:
    """Get UI configuration."""
    config = _load_config()
    ui = config.get("ui", {})
    return {
        "hover_debounce_ms": int(ui.get("hover_debounce_ms", 500)),
        "max_similar_papers": int(ui.get("max_similar_papers", 5)),
        "tree_auto_save_interval_ms": int(ui.get("tree_auto_save_interval_ms", 30000)),
    }


def _get_classification_config() -> dict[str, Any]:
    config = _load_config()
    classification = config.get("classification", {})
    return {
        "branching_factor": int(classification.get("branching_factor", 5)),
        "rebuild_on_ingest": bool(classification.get("rebuild_on_ingest", True)),
    }


def _get_external_apis_config() -> dict[str, Any]:
    """Get external API configuration."""
    config = _load_config()
    apis = config.get("external_apis", {})
    return {
        "papers_with_code_enabled": bool(apis.get("papers_with_code_enabled", True)),
        "github_search_enabled": bool(apis.get("github_search_enabled", True)),
        "semantic_scholar_enabled": bool(apis.get("semantic_scholar_enabled", True)),
        "github_token": apis.get("github_token"),
        "semantic_scholar_api_key": apis.get("semantic_scholar_api_key"),
    }
