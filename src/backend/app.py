from __future__ import annotations

import asyncio
import hashlib
import json
import os
import pathlib
import pickle
import re
import uuid
from functools import lru_cache
from typing import Any, Optional

import arxiv
import httpx
import yaml
from fastapi import FastAPI, HTTPException
from openai import OpenAI, AsyncOpenAI
from paperqa import Docs
from paperqa.readers import read_doc
from paperqa.settings import (
    AnswerSettings,
    MultimodalOptions,
    ParsingSettings,
    PromptSettings,
    Settings,
    make_default_litellm_model_list_settings,
)
from paperqa.types import Doc
from pydantic import BaseModel, Field

import db

app = FastAPI(title="paper-curator-backend")


# =============================================================================
# Shared HTTP Client Pool (connection reuse for external APIs)
# =============================================================================

# Shared clients for external APIs - reuse connections instead of creating new ones
_http_clients: dict[str, httpx.AsyncClient] = {}


def _get_http_client(name: str, timeout: float = 10.0, headers: dict | None = None) -> httpx.AsyncClient:
    """Get or create a shared httpx.AsyncClient for an external API.
    
    Reuses connections to eliminate TLS handshake overhead on repeated calls.
    """
    if name not in _http_clients:
        _http_clients[name] = httpx.AsyncClient(
            timeout=timeout,
            headers=headers or {},
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )
    return _http_clients[name]


@app.on_event("shutdown")
async def shutdown_http_clients():
    """Close all shared HTTP clients on shutdown."""
    for client in _http_clients.values():
        await client.aclose()
    _http_clients.clear()


# =============================================================================
# Request/Response Models
# =============================================================================

class ArxivResolveRequest(BaseModel):
    arxiv_id: Optional[str] = Field(default=None, description="arXiv ID, e.g. 1706.03762")
    url: Optional[str] = Field(default=None, description="arXiv URL")


class ArxivDownloadRequest(BaseModel):
    arxiv_id: Optional[str] = Field(default=None, description="arXiv ID, e.g. 1706.03762")
    url: Optional[str] = Field(default=None, description="arXiv URL")
    output_dir: Optional[str] = Field(default=None, description="Directory to store downloads")


class PdfExtractRequest(BaseModel):
    pdf_path: str = Field(description="Local PDF file path")


class SummarizeRequest(BaseModel):
    text: Optional[str] = Field(default=None, description="Full paper text or extracted sections")
    pdf_path: Optional[str] = Field(default=None, description="Local PDF file path")
    arxiv_id: Optional[str] = Field(default=None, description="arXiv ID for persisting index")


class StructuredSummarizeRequest(BaseModel):
    pdf_path: str = Field(description="Local PDF file path")
    arxiv_id: Optional[str] = Field(default=None, description="arXiv ID for persisting index")


class EmbedAbstractRequest(BaseModel):
    text: str = Field(description="Abstract text to embed for similarity search")


class EmbedFulltextRequest(BaseModel):
    arxiv_id: str = Field(description="arXiv ID of paper")
    pdf_path: str = Field(description="Path to PDF file")


class QaRequest(BaseModel):
    arxiv_id: Optional[str] = Field(default=None, description="arXiv ID for cached index lookup")
    context: Optional[str] = Field(default=None, description="Context text to answer from")
    question: str = Field(description="Question to answer")
    pdf_path: Optional[str] = Field(default=None, description="Local PDF file path")


class StructuredQaRequest(BaseModel):
    arxiv_id: str = Field(description="arXiv ID for cached index lookup (paper must be indexed)")
    pdf_path: Optional[str] = Field(default=None, description="PDF path fallback if index not found")


class ClassifyRequest(BaseModel):
    title: str = Field(description="Paper title")
    abstract: str = Field(description="Paper abstract or summary")
    existing_categories: list[str] = Field(default=[], description="Existing categories in the tree")


class SavePaperRequest(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: Optional[str] = None
    summary: Optional[str] = None
    pdf_path: Optional[str] = None
    latex_path: Optional[str] = None
    pdf_url: Optional[str] = None
    published_at: Optional[str] = None
    category: str
    abbreviation: Optional[str] = None  # Short display name for tree node


class BatchIngestRequest(BaseModel):
    directory: Optional[str] = Field(default=None, description="Local directory containing PDF files")
    slack_channel: Optional[str] = Field(default=None, description="Slack channel identifier (#channel-name, channel ID, or URL)")
    slack_token: Optional[str] = Field(default=None, description="Slack User OAuth Token (xoxp-...) - not persisted")
    skip_existing: bool = Field(default=True, description="Skip PDFs already in database")


class TreeNodeRequest(BaseModel):
    node_id: str
    name: str
    node_type: str  # 'category' or 'paper'
    parent_id: Optional[str] = None
    paper_id: Optional[int] = None
    position: int = 0


class RepoSearchRequest(BaseModel):
    arxiv_id: str
    title: str


class ReferencesRequest(BaseModel):
    arxiv_id: str


class ExplainReferenceRequest(BaseModel):
    reference_id: int
    source_paper_title: str
    cited_title: str
    citation_context: Optional[str] = None


class SimilarPapersRequest(BaseModel):
    arxiv_id: str


# =============================================================================
# Config Loading
# =============================================================================

@lru_cache(maxsize=4)
def _load_config_cached(config_mtime: float) -> dict[str, Any]:
    config_path = pathlib.Path("config/paperqa.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return config


def _load_config() -> dict[str, Any]:
    config_path = pathlib.Path("config/paperqa.yaml")
    if not config_path.exists():
        raise HTTPException(status_code=500, detail="Config file not found: config/paperqa.yaml")
    mtime = config_path.stat().st_mtime
    return _load_config_cached(mtime)


# =============================================================================
# Prompt Management
# =============================================================================

_prompts_cache: dict[str, Any] | None = None
_prompts_mtime: float = 0


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


def _convert_localhost_for_docker(url: str) -> str:
    """Convert localhost URLs to host.docker.internal when running in Docker.
    
    This allows users to always write 'localhost' in config, and it will
    automatically work both locally and inside Docker containers.
    """
    if not url:
        return url
    # Check if running in Docker (DATABASE_URL is set by compose.yml)
    in_docker = os.environ.get("DATABASE_URL", "").startswith("postgresql://")
    if in_docker:
        # Replace localhost variants with host.docker.internal
        url = url.replace("://localhost:", "://host.docker.internal:")
        url = url.replace("://127.0.0.1:", "://host.docker.internal:")
    return url


def _get_endpoint_config() -> dict[str, str]:
    """Get endpoint configuration."""
    config = _load_config()
    endpoints = config.get("endpoints", {})
    # Support both new and legacy config format
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
    """Get classification configuration."""
    config = _load_config()
    classification = config.get("classification", {})
    return {
        "category_threshold": int(classification.get("category_threshold", 10)),
        "auto_reclassify_enabled": bool(classification.get("auto_reclassify_enabled", True)),
    }


def _get_external_apis_config() -> dict[str, Any]:
    """Get external APIs configuration."""
    config = _load_config()
    apis = config.get("external_apis", {})
    return {
        "papers_with_code_enabled": bool(apis.get("papers_with_code_enabled", True)),
        "github_search_enabled": bool(apis.get("github_search_enabled", True)),
        "semantic_scholar_enabled": bool(apis.get("semantic_scholar_enabled", True)),
        "github_token": apis.get("github_token"),
    }


# =============================================================================
# Helper Functions
# =============================================================================

def _require_identifier(arxiv_id: Optional[str], url: Optional[str]) -> str:
    """Extract arXiv ID from provided arxiv_id or URL."""
    if arxiv_id:
        return arxiv_id
    if url:
        # Match /abs/, /pdf/, or /html/ paths
        match = re.search(r"arxiv\.org/(?:abs|pdf|html)/([^/\s?]+)", url)
        if match:
            return match.group(1).replace(".pdf", "")
        raise HTTPException(status_code=400, detail=f"Could not extract arXiv ID from URL: {url}")
    raise HTTPException(status_code=400, detail="Provide arxiv_id or url.")


def _extract_arxiv_ids_from_text(text: str) -> list[str]:
    """Extract all arXiv IDs/URLs from text.
    
    Returns list of arXiv IDs (format: YYYY.NNNNN or YYYY.NNNNNvN).
    Handles:
    - arXiv URLs: https://arxiv.org/abs/1234.5678
    - arXiv PDF URLs: https://arxiv.org/pdf/1234.5678.pdf
    - Plain arXiv IDs: 1234.5678 or 1234.5678v1
    """
    arxiv_ids = set()
    
    # Match arXiv URLs
    url_pattern = r"arxiv\.org/(?:abs|pdf|html)/([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)"
    for match in re.finditer(url_pattern, text, re.IGNORECASE):
        arxiv_id = match.group(1)
        # Remove .pdf extension if present
        arxiv_id = arxiv_id.replace(".pdf", "")
        arxiv_ids.add(arxiv_id)
    
    # Match standalone arXiv IDs (format: YYYY.NNNNN or YYYY.NNNNNvN)
    id_pattern = r"\b([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)\b"
    for match in re.finditer(id_pattern, text):
        arxiv_id = match.group(1)
        arxiv_ids.add(arxiv_id)
    
    return sorted(list(arxiv_ids))


def _resolve_slack_channel_id(client, channel_input: str) -> str:
    """Resolve Slack channel ID from various input formats.
    
    Supports:
    - Channel ID: C1234567890
    - Channel name: #general or general
    - Slack URL: https://workspace.slack.com/archives/C1234567890
    """
    channel_input = channel_input.strip()
    
    # If it's already a channel ID (starts with C)
    if channel_input.startswith("C") and len(channel_input) > 1:
        return channel_input
    
    # Extract channel ID from URL (supports multiple Slack URL formats)
    # Format 1: https://workspace.slack.com/archives/C1234567890
    url_match = re.search(r"/archives/(C[A-Z0-9]+)", channel_input)
    if url_match:
        return url_match.group(1)
    
    # Format 2: https://join.slack.com/share/... (Slack Connect share link - extract workspace/channel from it)
    # Note: These share links don't directly give us channel ID, but we can try to resolve via name
    
    # Extract channel name (remove # if present, handle spaces)
    channel_name = channel_input.lstrip("#").strip()
    
    # First, try to access the channel directly by name using conversations.info
    # Try multiple variants of the channel name to handle formatting differences
    channel_name_variants = [
        channel_name,  # Original (already stripped #)
        f"#{channel_name}",  # With # prefix
        channel_name.replace(" ", ""),  # Without spaces
        channel_name.replace("-", "_"),  # Hyphens to underscores
        channel_name.replace("_", "-"),  # Underscores to hyphens
    ]
    
    for variant in channel_name_variants:
        try:
            info_response = client.conversations_info(channel=variant)
            if info_response.get("ok") and info_response.get("channel"):
                channel_data = info_response["channel"]
                # Check if channel is archived
                if channel_data.get("is_archived"):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Channel '{channel_input}' is archived. Archived channels cannot be accessed."
                    )
                return channel_data["id"]
            elif not info_response.get("ok"):
                error = info_response.get("error", "Unknown error")
                if error == "missing_scope":
                    raise HTTPException(
                        status_code=403,
                        detail=f"Slack token missing required scope 'channels:read' or 'groups:read'"
                    )
                # For other errors (like channel_not_found), try next variant
        except HTTPException:
            raise
        except Exception:
            # If this variant fails, try next one
            continue
    
    # List all channels and find matching one
    try:
        response = client.conversations_list(types="public_channel,private_channel", limit=1000, exclude_archived=True)
        if not response["ok"]:
            error_msg = response.get("error", "Unknown error")
            # Provide helpful error messages for common issues
            if error_msg == "missing_scope":
                raise HTTPException(
                    status_code=403,
                    detail=f"Slack token missing required scopes. Please add 'channels:read' and 'groups:read' scopes to your Slack app. Error: {error_msg}"
                )
            elif error_msg == "invalid_auth":
                raise HTTPException(
                    status_code=401,
                    detail=f"Invalid Slack token. Please check your token and ensure it's a valid User OAuth Token (xoxp-...). Error: {error_msg}"
                )
            else:
                raise HTTPException(status_code=400, detail=f"Slack API error: {error_msg}")
        
        channels = response.get("channels", [])
        if not channels:
            raise HTTPException(
                status_code=404,
                detail=f"No channels found. This might indicate insufficient permissions. Channel name: '{channel_input}'"
            )
        
        # Normalize channel name for matching (remove special chars, lowercase)
        def normalize_name(name: str) -> str:
            return name.lower().replace("-", "").replace("_", "").replace(" ", "")
        
        normalized_target = normalize_name(channel_name)
        
        # Try exact match first
        for channel in channels:
            if channel["name"] == channel_name:
                return channel["id"]
        
        # Try case-insensitive match
        for channel in channels:
            if channel["name"].lower() == channel_name.lower():
                return channel["id"]
        
        # Try normalized match (handles hyphens, underscores, spaces, case differences)
        for channel in channels:
            if normalize_name(channel["name"]) == normalized_target:
                return channel["id"]
        
        # Try fuzzy match - check if target is contained in any channel name
        for channel in channels:
            if normalized_target in normalize_name(channel["name"]) or normalize_name(channel["name"]) in normalized_target:
                return channel["id"]
        
        # If still not found, provide helpful error with available channels and search for similar ones
        available_channels = [ch["name"] for ch in channels[:20]]  # Show first 20
        similar_channels = [
            ch["name"] for ch in channels 
            if normalized_target[:5] in normalize_name(ch["name"]) or normalize_name(ch["name"])[:5] in normalized_target
        ]
        
        # Also try to find channels containing "model", "training", "paper", or "weights"
        keyword_matches = [
            ch["name"] for ch in channels 
            if any(keyword in normalize_name(ch["name"]) for keyword in ["model", "training", "paper", "weight"])
        ]
        
        error_msg = f"Channel '{channel_input}' not found in accessible channels. "
        if similar_channels:
            error_msg += f"Similar channels: {', '.join(similar_channels[:5])}. "
        if keyword_matches:
            error_msg += f"Channels with related keywords: {', '.join(keyword_matches[:5])}. "
        error_msg += f"Available channels (first 20): {', '.join(available_channels)}. "
        error_msg += "\n\nTroubleshooting:\n"
        error_msg += "1. Verify the channel name is correct (check for typos, spaces, or different formatting)\n"
        error_msg += "2. IMPORTANT: If using a User OAuth Token (xoxp-...), the USER whose token it is must be a member of the channel.\n"
        error_msg += "   Adding the app/bot to the channel is not enough - the user must also join the channel.\n"
        error_msg += "3. Alternatively, use a Bot Token (xoxb-...) and ensure the bot is added to the channel\n"
        error_msg += "4. Check if the channel is archived (archived channels may not appear)\n"
        error_msg += "5. Verify you're using the token for the correct Slack workspace\n"
        error_msg += "6. Try using the channel ID instead: Right-click channel → Copy link → Use the C1234567890 ID from the URL"
        
        raise HTTPException(status_code=404, detail=error_msg)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resolving Slack channel: {e}")


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


def _get_openai_client(base_url: str, api_key: str) -> OpenAI:
    """Create OpenAI client with ngrok header support if needed."""
    # Add ngrok-skip-browser-warning header for ngrok free tier
    if "ngrok" in base_url.lower():
        import httpx
        # Use both default_headers and http_client to ensure header is sent
        http_client = httpx.Client(
            headers={"ngrok-skip-browser-warning": "true"},
            timeout=30.0,
            follow_redirects=True,
        )
        return OpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=http_client,
            default_headers={"ngrok-skip-browser-warning": "true"},
        )
    return OpenAI(base_url=base_url, api_key=api_key)


def _get_async_openai_client(base_url: str, api_key: str) -> AsyncOpenAI:
    """Create AsyncOpenAI client with ngrok header support if needed."""
    # Add ngrok-skip-browser-warning header for ngrok free tier
    if "ngrok" in base_url.lower():
        import httpx
        # Use both default_headers and http_client to ensure header is sent
        http_client = httpx.AsyncClient(
            headers={"ngrok-skip-browser-warning": "true"},
            timeout=30.0,
            follow_redirects=True,
        )
        return AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=http_client,
            default_headers={"ngrok-skip-browser-warning": "true"},
        )
    return AsyncOpenAI(base_url=base_url, api_key=api_key)


@lru_cache(maxsize=3)
def _resolve_model(base_url: str, api_key: str) -> str:
    """Resolve model name from endpoint, with ngrok free tier workaround."""
    client = _get_openai_client(base_url, api_key)
    try:
        models = client.models.list()
        model_ids = sorted([model.id for model in models.data if getattr(model, "id", None)])
        assert model_ids, "No models returned by OpenAI-compatible endpoint."
        return model_ids[0]
    except Exception as e:
        error_msg = str(e)
        # Check if it's an ngrok browser warning issue
        if "ngrok" in base_url.lower() and (
            "ERR_NGROK_3004" in error_msg 
            or "gateway error" in error_msg.lower() 
            or "invalid or incomplete HTTP response" in error_msg.lower()
            or "ngrok gateway error" in error_msg.lower()
        ):
            help_msg = (
                "ngrok ERR_NGROK_3004: Browser warning page blocking programmatic access.\n\n"
                "SOLUTIONS (choose one):\n\n"
                "1. Configure ngrok to skip browser warning (RECOMMENDED):\n"
                "   Restart ngrok with:\n"
                "   ngrok http 8001 --request-header-add 'ngrok-skip-browser-warning: true'\n\n"
                "   OR add to ~/.ngrok2/ngrok.yml:\n"
                "   tunnels:\n"
                "     llm:\n"
                "       addr: 8001\n"
                "       request_header:\n"
                "         add: ['ngrok-skip-browser-warning: true']\n\n"
                "2. Upgrade to ngrok paid plan (has Edge request headers)\n\n"
                "3. Use alternative tunneling:\n"
                "   - Cloudflare Tunnel (free, no browser warning)\n"
                "   - localtunnel (free, simple)\n"
                "   - serveo (free, SSH-based)\n\n"
                f"Current endpoint: {base_url}\n"
                f"Original error: {error_msg}"
            )
            raise Exception(help_msg)
        raise


def _get_paperqa_index_path(arxiv_id: str) -> pathlib.Path:
    """Get path to stored PaperQA2 index for a paper."""
    index_dir = pathlib.Path("storage/paperqa_index")
    index_dir.mkdir(parents=True, exist_ok=True)
    # Sanitize arxiv_id for filename
    safe_id = arxiv_id.replace("/", "_").replace(".", "_")
    return index_dir / f"{safe_id}.pkl"


def _reset_litellm_callbacks() -> None:
    """Reset LiteLLM callbacks to prevent accumulation.
    
    LiteLLM accumulates callbacks with each LLM call. When the limit (30) is
    reached, it causes instability. This resets all callback lists.
    """
    import litellm
    litellm.input_callback = []
    litellm.success_callback = []
    litellm.failure_callback = []
    litellm._async_success_callback = []
    litellm._async_failure_callback = []


def _build_paperqa_settings(
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
    # Disable JSON mode to avoid parsing issues with non-OpenAI models
    # PaperQA2 expects specific JSON format that some models don't produce correctly
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


async def _index_pdf_for_paperqa_async(
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
    settings = _build_paperqa_settings(
        llm_model, embed_model, llm_base_url, embed_base_url, api_key
    )
    await docs.aadd(pdf_path, settings=settings)
    
    # Persist to filesystem using pickle
    index_path = _get_paperqa_index_path(arxiv_id)
    with open(index_path, "wb") as f:
        pickle.dump(docs, f)
    
    return docs


def _index_pdf_for_paperqa(
    pdf_path: str,
    arxiv_id: str,
    llm_base_url: str,
    embed_base_url: str,
    api_key: str,
    llm_model: str,
    embed_model: str,
) -> Docs:
    """Index a PDF and persist the Docs object for later QA queries (sync wrapper)."""
    return asyncio.run(_index_pdf_for_paperqa_async(
        pdf_path, arxiv_id, llm_base_url, embed_base_url, api_key, llm_model, embed_model
    ))


def _load_paperqa_index(arxiv_id: str) -> Optional[Docs]:
    """Load persisted PaperQA2 Docs if exists."""
    index_path = _get_paperqa_index_path(arxiv_id)
    if not index_path.exists():
        return None
    try:
        with open(index_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        # If loading fails, return None to trigger re-indexing
        return None


async def _paperqa_answer_async(
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
    
    # Try to load cached index if arxiv_id provided
    if arxiv_id:
        docs = _load_paperqa_index(arxiv_id)
    
    # If no cached index, create new one
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
        
        settings = _build_paperqa_settings(
            llm_model, embed_model, llm_base_url, embed_base_url, api_key
        )
        await docs.aadd(str(content_path), settings=settings)
        
        # Persist if arxiv_id provided
        if arxiv_id:
            index_path = _get_paperqa_index_path(arxiv_id)
            with open(index_path, "wb") as f:
                pickle.dump(docs, f)
    
    settings = _build_paperqa_settings(
        llm_model, embed_model, llm_base_url, embed_base_url, api_key
    )
    result = await docs.aquery(question, settings=settings)
    answer = str(result.answer) if hasattr(result, "answer") else str(result)
    # Clean up PaperQA2 citation markers
    answer = _clean_paperqa_citations(answer)
    return answer


def _paperqa_answer(
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
    """Query PaperQA2 for an answer (sync wrapper). Uses cached index if available."""
    return asyncio.run(_paperqa_answer_async(
        text, pdf_path, question, llm_base_url, embed_base_url, api_key,
        llm_model, embed_model, arxiv_id
    ))


def _clean_paperqa_citations(text: str) -> str:
    """Remove PaperQA2 citation markers from text.
    
    PaperQA2 adds citations like:
    - "(docname pages 1-2)"
    - "(docname page 5)"
    - "(An2025 pages 1-2, An2025 pages 9-12)"
    which are not useful for end users.
    """
    # Pattern matches parentheses containing citation-like content
    # e.g., (An2025 pages 1-2) or (An2025 pages 1-2, An2025 pages 9-12)
    pattern = r'\s*\([A-Za-z0-9_]+\s+pages?\s+[\d,\-–\s]+(?:,\s*[A-Za-z0-9_]+\s+pages?\s+[\d,\-–\s]+)*\)'
    cleaned = re.sub(pattern, '', text)
    # Also clean up any double spaces left behind
    cleaned = re.sub(r'  +', ' ', cleaned)
    return cleaned.strip()


async def _download_arxiv_pdf(arxiv_id: str, pdf_url: str) -> str:
    """Download arXiv PDF and return local path with retry logic."""
    output_dir = pathlib.Path("storage/downloads")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use arxiv library to download with retry
    max_retries = 3
    last_error = None
    
    for attempt in range(max_retries):
        try:
            client = arxiv.Client()
            search = arxiv.Search(id_list=[arxiv_id])
            results = list(client.results(search))
            
            if not results:
                raise HTTPException(status_code=404, detail=f"arXiv paper not found: {arxiv_id}")
            
            result = results[0]
            pdf_path = result.download_pdf(dirpath=str(output_dir))
            return str(pdf_path)
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                import asyncio
                await asyncio.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                continue
            else:
                # Last attempt failed, raise with context
                error_msg = str(e)
                if "retrieval incomplete" in error_msg or "urlopen error" in error_msg:
                    raise Exception(f"PDF download failed after {max_retries} attempts: {error_msg}. The PDF may be too large or the connection was interrupted.")
                raise Exception(f"PDF download failed after {max_retries} attempts: {error_msg}")
    
    # Should never reach here, but just in case
    raise Exception(f"PDF download failed: {last_error}")


async def _paperqa_extract_pdf_async(pdf_path: pathlib.Path) -> dict[str, Any]:
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


def _paperqa_extract_pdf(pdf_path: pathlib.Path) -> dict[str, Any]:
    """Extract text from PDF using PaperQA2's native parser (sync wrapper)."""
    return asyncio.run(_paperqa_extract_pdf_async(pdf_path))


# =============================================================================
# Core Endpoints
# =============================================================================

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/config")
def get_config() -> dict[str, Any]:
    """Return UI configuration for the frontend."""
    return _get_ui_config()


@app.post("/arxiv/resolve")
def arxiv_resolve(payload: ArxivResolveRequest) -> dict[str, Any]:
    identifier = _require_identifier(payload.arxiv_id, payload.url)
    client = arxiv.Client()
    search = arxiv.Search(id_list=[identifier])
    results = list(client.results(search))
    assert results, f"No arXiv result found for: {identifier}"
    result = results[0]
    return {
        "arxiv_id": result.get_short_id(),
        "title": result.title,
        "authors": [author.name for author in result.authors],
        "published": result.published.isoformat() if result.published else None,
        "summary": result.summary,
        "pdf_url": result.pdf_url,
        "entry_id": result.entry_id,
        "comment": result.comment,
    }


@app.post("/arxiv/download")
def arxiv_download(payload: ArxivDownloadRequest) -> dict[str, Any]:
    identifier = _require_identifier(payload.arxiv_id, payload.url)
    output_dir = payload.output_dir or os.getenv("ARXIV_DOWNLOAD_DIR", "storage/downloads")
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    client = arxiv.Client()
    search = arxiv.Search(id_list=[identifier])
    results = list(client.results(search))
    if not results:
        raise HTTPException(status_code=404, detail="No arXiv result found.")
    result = results[0]

    pdf_path = result.download_pdf(dirpath=output_dir)
    latex_path = result.download_source(dirpath=output_dir) if result.source_url() else None

    return {
        "arxiv_id": result.get_short_id(),
        "pdf_path": pdf_path,
        "latex_path": latex_path,
    }


@app.post("/pdf/extract")
def pdf_extract(payload: PdfExtractRequest) -> dict[str, Any]:
    """Extract text from PDF using PaperQA2's native parser."""
    pdf_path = pathlib.Path(payload.pdf_path)
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF not found.")
    return _paperqa_extract_pdf(pdf_path)


@app.post("/summarize")
def summarize(payload: SummarizeRequest) -> dict[str, Any]:
    """Summarize a paper. Also indexes PDF for faster QA queries if arxiv_id provided."""
    prompt_id, prompt_hash, prompt_body = _load_prompt()
    endpoint_config = _get_endpoint_config()
    base_url = endpoint_config["llm_base_url"]
    embed_base_url = endpoint_config["embedding_base_url"]
    api_key = endpoint_config["api_key"]
    
    # Check LLM endpoint
    try:
        model = f"openai/{_resolve_model(base_url, api_key)}"
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM endpoint not available at {base_url}: {e}")
    
    # Check embedding endpoint
    try:
        embed_model = f"openai/{_resolve_model(embed_base_url, api_key)}"
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Embedding endpoint not available at {embed_base_url}. Please start your embedding service on port 8004.")
    
    summary = _paperqa_answer(
        payload.text,
        payload.pdf_path,
        prompt_body,
        base_url,
        embed_base_url,
        api_key,
        model,
        embed_model,
        arxiv_id=payload.arxiv_id,  # Persist index for QA reuse
    )
    return {
        "summary": summary.strip(),
        "prompt_id": prompt_id,
        "prompt_hash": prompt_hash,
        "model": model,
        "indexed": payload.arxiv_id is not None,
    }


@app.post("/summarize/structured")
async def summarize_structured(payload: StructuredSummarizeRequest) -> dict[str, Any]:
    """Generate a structured summary with components and detailed analysis.
    
    Flow:
    1. Extract key components/concepts from the paper
    2. For each component, query 4 aspects in parallel:
       - Logical steps / pseudo-code
       - Main benefit area
       - Rationale behind benefit
       - Quantifiable results
    3. Return structured sections
    
    Uses: PaperQA2 for document indexing and LLM queries
    """
    endpoint_config = _get_endpoint_config()
    base_url = endpoint_config["llm_base_url"]
    embed_base_url = endpoint_config["embedding_base_url"]
    api_key = endpoint_config["api_key"]
    
    model = f"openai/{_resolve_model(base_url, api_key)}"
    embed_model = f"openai/{_resolve_model(embed_base_url, api_key)}"
    
    _reset_litellm_callbacks()
    
    # Step 1: Extract components using PaperQA2
    extract_prompt = get_prompt("extract_components")
    components_raw = await _paperqa_answer_async(
        None,
        payload.pdf_path,
        extract_prompt,
        base_url,
        embed_base_url,
        api_key,
        model,
        embed_model,
        arxiv_id=payload.arxiv_id,
    )
    
    # Parse components JSON - extract array from response
    match = re.search(r'\[.*\]', components_raw, re.DOTALL)
    components = json.loads(match.group()) if match else [components_raw.strip().strip('"\'')]
    components = components or ["Main Contribution"]
    
    # Step 2: For each component, query 4 aspects sequentially
    async def analyze_component(component: str) -> dict[str, Any]:
        """Analyze a single component with 4 sequential queries."""
        result = {"component": component}
        
        aspects = [
            ("steps", get_prompt("component_steps", component=component)),
            ("benefits", get_prompt("component_benefits", component=component)),
            ("rationale", get_prompt("component_rationale", component=component)),
            ("results", get_prompt("component_results", component=component)),
        ]
        
        for aspect, prompt in aspects:
            _reset_litellm_callbacks()
            answer = await _paperqa_answer_async(
                None,
                payload.pdf_path,
                prompt,
                base_url,
                embed_base_url,
                api_key,
                model,
                embed_model,
                arxiv_id=payload.arxiv_id,
            )
            result[aspect] = answer.strip()
        
        return result
    
    # Analyze components sequentially to avoid callback accumulation
    sections = []
    for component in components:
        _reset_litellm_callbacks()
        section = await analyze_component(component)
        sections.append(section)
    
    return {
        "components": list(components),
        "sections": list(sections),
        "model": model,
        "indexed": payload.arxiv_id is not None,
    }


@app.post("/embed/abstract")
async def embed_abstract(payload: EmbedAbstractRequest) -> dict[str, Any]:
    """Embed abstract text for pgvector similarity search."""
    endpoint_config = _get_endpoint_config()
    base_url = endpoint_config["embedding_base_url"]
    api_key = endpoint_config["api_key"]
    model = _resolve_model(base_url, api_key)
    client = _get_async_openai_client(base_url, api_key)
    response = await client.embeddings.create(model=model, input=payload.text)
    vector = response.data[0].embedding
    return {"embedding": vector, "model": model}


@app.post("/embed/fulltext")
def embed_fulltext(payload: EmbedFulltextRequest) -> dict[str, Any]:
    """Index full PDF for PaperQA2 queries. Persists index for reuse."""
    endpoint_config = _get_endpoint_config()
    llm_base_url = endpoint_config["llm_base_url"]
    embed_base_url = endpoint_config["embedding_base_url"]
    api_key = endpoint_config["api_key"]
    
    # Check if already indexed
    index_path = _get_paperqa_index_path(payload.arxiv_id)
    if index_path.exists():
        return {"indexed": True, "cached": True, "arxiv_id": payload.arxiv_id}
    
    try:
        llm_model = f"openai/{_resolve_model(llm_base_url, api_key)}"
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM endpoint not available: {e}")
    
    try:
        embed_model = f"openai/{_resolve_model(embed_base_url, api_key)}"
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Embedding endpoint not available: {e}")
    
    _index_pdf_for_paperqa(
        payload.pdf_path,
        payload.arxiv_id,
        llm_base_url,
        embed_base_url,
        api_key,
        llm_model,
        embed_model,
    )
    return {"indexed": True, "cached": False, "arxiv_id": payload.arxiv_id}


# Keep old endpoint for backwards compatibility
@app.post("/embed")
async def embed(payload: EmbedAbstractRequest) -> dict[str, Any]:
    """Embed text (backwards compatible, use /embed/abstract instead)."""
    return await embed_abstract(payload)


@app.post("/qa")
def qa(payload: QaRequest) -> dict[str, Any]:
    """Answer a question about a paper. Uses cached index if arxiv_id provided.
    
    Persists the query and answer to the database for later retrieval.
    """
    endpoint_config = _get_endpoint_config()
    base_url = endpoint_config["llm_base_url"]
    embed_base_url = endpoint_config["embedding_base_url"]
    api_key = endpoint_config["api_key"]
    model = f"openai/{_resolve_model(base_url, api_key)}"
    embed_model = f"openai/{_resolve_model(embed_base_url, api_key)}"
    
    # Check if we have a cached index
    has_cache = payload.arxiv_id and _get_paperqa_index_path(payload.arxiv_id).exists()
    
    answer = _paperqa_answer(
        payload.context,
        payload.pdf_path,
        payload.question,
        base_url,
        embed_base_url,
        api_key,
        model,
        embed_model,
        arxiv_id=payload.arxiv_id,
    )
    
    # Persist query to database if arxiv_id provided
    if payload.arxiv_id:
        paper = db.get_paper_by_arxiv_id(payload.arxiv_id)
        if paper:
            db.add_query(paper["id"], payload.question, answer, model)
    
    return {"answer": answer, "used_cache": has_cache}


@app.post("/qa/structured")
async def qa_structured(payload: StructuredQaRequest) -> dict[str, Any]:
    """Run structured analysis on an already-indexed paper.
    
    Uses cached PaperQA index for efficient querying. Runs ALL aspect queries
    in parallel using asyncio.gather for maximum throughput.
    
    Flow:
    1. Extract key components from paper (sequential - need results first)
    2. For ALL components, query ALL 4 aspects in parallel
    3. Return structured sections
    """
    # Check for cached index
    index_path = _get_paperqa_index_path(payload.arxiv_id)
    if not index_path.exists() and not payload.pdf_path:
        raise HTTPException(
            status_code=404, 
            detail=f"No cached index for {payload.arxiv_id}. Ingest the paper first."
        )
    
    endpoint_config = _get_endpoint_config()
    base_url = endpoint_config["llm_base_url"]
    embed_base_url = endpoint_config["embedding_base_url"]
    api_key = endpoint_config["api_key"]
    model = f"openai/{_resolve_model(base_url, api_key)}"
    embed_model = f"openai/{_resolve_model(embed_base_url, api_key)}"
    
    def reset_callbacks():
        try:
            import litellm
            litellm.input_callback = []
            litellm.success_callback = []
            litellm.failure_callback = []
            litellm._async_success_callback = []
            litellm._async_failure_callback = []
        except Exception:
            pass
    
    async def run_query_async(question: str) -> str:
        """Run a single PaperQA query asynchronously."""
        reset_callbacks()
        return await _paperqa_answer_async(
            None,
            payload.pdf_path,
            question,
            base_url,
            embed_base_url,
            api_key,
            model,
            embed_model,
            arxiv_id=payload.arxiv_id,
        )
    
    # Step 1: Extract components (sequential - need results before step 2)
    reset_callbacks()
    extract_prompt = get_prompt("extract_components")
    components_raw = await run_query_async(extract_prompt)
    
    # Parse components JSON
    try:
        match = re.search(r'\[.*\]', components_raw, re.DOTALL)
        if match:
            components = json.loads(match.group())
        else:
            components = [components_raw.strip().strip('"\'')]
    except json.JSONDecodeError:
        components = [components_raw.strip().strip('"\'')]
    
    if not components:
        components = ["Main Contribution"]
    
    # Limit components to prevent excessive runtime
    components = components[:5]
    
    # Step 2: Build all queries and run them in parallel
    # Create list of (component_index, aspect_name, question) tuples
    query_tasks = []
    query_metadata = []  # Track which component and aspect each query belongs to
    
    for idx, component in enumerate(components):
        for aspect, prompt_name in [
            ("steps", "component_steps"),
            ("benefits", "component_benefits"),
            ("rationale", "component_rationale"),
            ("results", "component_results"),
        ]:
            question = get_prompt(prompt_name, component=component)
            query_tasks.append(run_query_async(question))
            query_metadata.append((idx, aspect))
    
    # Run ALL queries in parallel
    results = await asyncio.gather(*query_tasks, return_exceptions=True)
    
    # Reassemble results into sections
    sections = [{"component": comp} for comp in components]
    for (idx, aspect), result in zip(query_metadata, results):
        if isinstance(result, Exception):
            sections[idx][aspect] = f"Error: {result}"
        else:
            sections[idx][aspect] = result
    
    # Persist structured analysis to database
    structured_result = {
        "components": components,
        "sections": sections,
        "model": model,
    }
    
    if payload.arxiv_id:
        paper = db.get_paper_by_arxiv_id(payload.arxiv_id)
        if paper:
            db.update_paper_structured_summary(paper["id"], structured_result)
    
    return structured_result


class MergeSummaryRequest(BaseModel):
    arxiv_id: str = Field(description="arXiv ID of the paper")
    selected_qa: list[dict[str, str]] = Field(description="List of {question, answer} pairs to merge")


class DedupSummaryRequest(BaseModel):
    arxiv_id: str = Field(description="arXiv ID of the paper")


@app.post("/summary/merge")
async def merge_qa_to_summary(payload: MergeSummaryRequest) -> dict[str, Any]:
    """Merge selected Q&A content into the paper's summary.
    
    Uses LLM to intelligently integrate Q&A content into appropriate
    sections of the existing summary.
    """
    # Get paper from DB
    paper = db.get_paper_by_arxiv_id(payload.arxiv_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    
    current_summary = paper.get("summary", "")
    if not current_summary:
        raise HTTPException(status_code=400, detail="Paper has no summary to merge into")
    
    if not payload.selected_qa:
        raise HTTPException(status_code=400, detail="No Q&A pairs provided")
    
    # Format Q&A content
    qa_content = "\n\n".join([
        f"Q: {qa['question']}\nA: {qa['answer']}"
        for qa in payload.selected_qa
    ])
    
    # Get LLM config
    endpoint_config = _get_endpoint_config()
    base_url = endpoint_config["llm_base_url"]
    api_key = endpoint_config["api_key"]
    model = _resolve_model(base_url, api_key)
    client = _get_async_openai_client(base_url, api_key)
    
    # Get merge prompt
    prompt = get_prompt(
        "merge_qa_to_summary",
        current_summary=current_summary,
        qa_content=qa_content,
    )
    
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4000,
        temperature=0.3,
    )
    updated_summary = response.choices[0].message.content.strip()
    
    # Update summary in database
    db.update_paper_summary(paper["id"], updated_summary)
    
    return {
        "summary": updated_summary,
        "merged_count": len(payload.selected_qa),
        "model": model,
    }


@app.post("/summary/dedup")
async def dedup_summary(payload: DedupSummaryRequest) -> dict[str, Any]:
    """Remove duplicated content from a paper's summary.
    
    Uses LLM to identify and remove redundant information while
    preserving the most detailed version.
    """
    # Get paper from DB
    paper = db.get_paper_by_arxiv_id(payload.arxiv_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    
    current_summary = paper.get("summary", "")
    if not current_summary:
        raise HTTPException(status_code=400, detail="Paper has no summary")
    
    # Get LLM config
    endpoint_config = _get_endpoint_config()
    base_url = endpoint_config["llm_base_url"]
    api_key = endpoint_config["api_key"]
    model = _resolve_model(base_url, api_key)
    client = _get_async_openai_client(base_url, api_key)
    
    # Get dedup prompt
    prompt = get_prompt("dedup_summary", summary=current_summary)
    
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4000,
        temperature=0.1,
    )
    deduped_summary = response.choices[0].message.content.strip()
    
    # Update summary in database
    db.update_paper_summary(paper["id"], deduped_summary)
    
    return {
        "summary": deduped_summary,
        "original_length": len(current_summary),
        "new_length": len(deduped_summary),
        "model": model,
    }


@app.post("/classify")
async def classify(payload: ClassifyRequest) -> dict[str, Any]:
    """Classify a paper into a category using LLM."""
    endpoint_config = _get_endpoint_config()
    base_url = endpoint_config["llm_base_url"]
    api_key = endpoint_config["api_key"]
    model = _resolve_model(base_url, api_key)
    client = _get_async_openai_client(base_url, api_key)

    existing_str = ", ".join(payload.existing_categories) if payload.existing_categories else "None yet"
    prompt = get_prompt(
        "classify",
        existing_categories=existing_str,
        title=payload.title,
        abstract=payload.abstract[:2000],
    )

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0.1,
    )
    category = response.choices[0].message.content.strip()
    return {"category": category, "model": model}


class AbbreviateRequest(BaseModel):
    title: str


@app.post("/abbreviate")
async def abbreviate(payload: AbbreviateRequest) -> dict[str, Any]:
    """Generate a short abbreviation for a paper title."""
    endpoint_config = _get_endpoint_config()
    base_url = endpoint_config["llm_base_url"]
    api_key = endpoint_config["api_key"]
    model = _resolve_model(base_url, api_key)
    client = _get_async_openai_client(base_url, api_key)
    
    prompt = get_prompt("abbreviate", title=payload.title)
    
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=20,
        temperature=0.1,
    )
    abbrev = response.choices[0].message.content.strip()
    # Clean up any quotes or extra formatting
    abbrev = abbrev.strip('"\'').strip()
    return {"abbreviation": abbrev, "model": model}


class ReabbreviateRequest(BaseModel):
    arxiv_id: str


@app.post("/papers/reabbreviate")
async def reabbreviate_paper(payload: ReabbreviateRequest) -> dict[str, Any]:
    """Re-generate abbreviation for an existing paper and update tree node."""
    # Get paper from DB
    paper = db.get_paper_by_arxiv_id(payload.arxiv_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    
    # Generate new abbreviation
    endpoint_config = _get_endpoint_config()
    base_url = endpoint_config["llm_base_url"]
    api_key = endpoint_config["api_key"]
    model = _resolve_model(base_url, api_key)
    client = _get_async_openai_client(base_url, api_key)
    
    prompt = get_prompt("abbreviate", title=paper["title"])
    
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=20,
        temperature=0.1,
    )
    abbrev = response.choices[0].message.content.strip().strip('"\'').strip()
    
    # Update tree node name
    node_id = f"paper_{payload.arxiv_id.replace('.', '_')}"
    db.update_tree_node_name(node_id, abbrev)
    
    return {"abbreviation": abbrev, "node_id": node_id}


@app.post("/papers/reabbreviate-all")
async def reabbreviate_all_papers() -> dict[str, Any]:
    """Re-generate abbreviations for ALL papers in parallel."""
    # Get all papers
    papers = db.get_all_papers()
    if not papers:
        return {"updated": 0, "results": []}
    
    endpoint_config = _get_endpoint_config()
    base_url = endpoint_config["llm_base_url"]
    api_key = endpoint_config["api_key"]
    model = _resolve_model(base_url, api_key)
    client = _get_async_openai_client(base_url, api_key)
    
    async def abbreviate_one(paper: dict) -> dict[str, Any]:
        prompt = get_prompt("abbreviate", title=paper["title"])
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.1,
        )
        abbrev = response.choices[0].message.content.strip().strip('"\'').strip()
        node_id = f"paper_{paper['arxiv_id'].replace('.', '_')}"
        db.update_tree_node_name(node_id, abbrev)
        return {"arxiv_id": paper["arxiv_id"], "abbreviation": abbrev}
    
    # Run all abbreviations in parallel
    results = await asyncio.gather(*[abbreviate_one(p) for p in papers])
    
    return {"updated": len(results), "results": results}


# =============================================================================
# Database & Tree Endpoints
# =============================================================================

@app.post("/papers/save")
async def save_paper(payload: SavePaperRequest) -> dict[str, Any]:
    """Save a paper to the database and add it to the tree.
    
    If the target category exceeds the threshold, triggers auto-reclassification.
    """
    # Create paper in DB
    paper_id = db.create_paper(
        arxiv_id=payload.arxiv_id,
        title=payload.title,
        authors=payload.authors,
        abstract=payload.abstract,
        summary=payload.summary,
        pdf_path=payload.pdf_path,
        latex_path=payload.latex_path,
        pdf_url=payload.pdf_url,
        published_at=payload.published_at,
    )
    
    # Check if category exists, if not create it
    tree = db.get_tree()
    category_exists = any(n["name"] == payload.category and n["node_type"] == "category" for n in tree)
    
    if not category_exists:
        category_node_id = f"cat_{uuid.uuid4().hex[:8]}"
        db.add_tree_node(
            node_id=category_node_id,
            name=payload.category,
            node_type="category",
            parent_id="root",
        )
    else:
        category_node_id = next(n["node_id"] for n in tree if n["name"] == payload.category and n["node_type"] == "category")
    
    # Add paper node - use abbreviation if provided, otherwise fallback to truncated title
    paper_node_id = f"paper_{payload.arxiv_id.replace('.', '_')}"
    display_name = payload.abbreviation if payload.abbreviation else (
        payload.title[:30] + ".." if len(payload.title) > 30 else payload.title
    )
    db.add_tree_node(
        node_id=paper_node_id,
        name=display_name,
        node_type="paper",
        parent_id=category_node_id,
        paper_id=paper_id,
    )
    
    # Check if auto-reclassification is needed
    reclassified = []
    classification_config = _get_classification_config()
    if classification_config["auto_reclassify_enabled"]:
        paper_count = db.get_category_paper_count_by_id(category_node_id)
        threshold = classification_config["category_threshold"]
        
        if paper_count > threshold:
            # Get all papers in the overcrowded category
            papers_in_category = db.get_papers_in_category(category_node_id)
            
            # Get existing categories (excluding the current one)
            existing_categories = [
                n["name"] for n in tree 
                if n["node_type"] == "category" and n["node_id"] != category_node_id
            ]
            
            # Reclassify each paper
            endpoint_config = _get_endpoint_config()
            base_url = endpoint_config["llm_base_url"]
            api_key = endpoint_config["api_key"]
            model = _resolve_model(base_url, api_key)
            client = _get_async_openai_client(base_url, api_key)
            
            for paper in papers_in_category:
                # Ask LLM for more specific category
                prompt = get_prompt(
                    "classify",
                    existing_categories=", ".join(existing_categories) if existing_categories else "None yet",
                    title=paper["title"],
                    abstract=(paper.get("abstract") or "")[:2000],
                )
                
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=50,
                    temperature=0.1,
                )
                new_category = response.choices[0].message.content.strip()
                
                # Only move if category is different
                if new_category != payload.category:
                    # Check if new category exists
                    new_cat_node = next(
                        (n for n in db.get_tree() if n["name"] == new_category and n["node_type"] == "category"),
                        None
                    )
                    
                    if not new_cat_node:
                        # Create new category
                        new_cat_node_id = f"cat_{uuid.uuid4().hex[:8]}"
                        db.add_tree_node(
                            node_id=new_cat_node_id,
                            name=new_category,
                            node_type="category",
                            parent_id="root",
                        )
                    else:
                        new_cat_node_id = new_cat_node["node_id"]
                    
                    # Move paper to new category
                    db.move_paper_to_category(paper["node_id"], new_cat_node_id)
                    reclassified.append({
                        "arxiv_id": paper["arxiv_id"],
                        "old_category": payload.category,
                        "new_category": new_category,
                    })
                    
                    # Add new category to existing list for subsequent papers
                    if new_category not in existing_categories:
                        existing_categories.append(new_category)
    
    return {
        "paper_id": paper_id,
        "node_id": paper_node_id,
        "display_name": display_name,
        "reclassified": reclassified,
    }


async def _fetch_slack_messages(client, channel_id: str) -> list[dict[str, Any]]:
    """Fetch all messages from a Slack channel with pagination.
    
    Returns list of message objects containing 'text' field.
    Uses asyncio.to_thread to avoid blocking the event loop.
    """
    all_messages = []
    cursor = None
    
    while True:
        try:
            # Wrap blocking Slack API call in thread executor
            response = await asyncio.to_thread(
                client.conversations_history,
                channel=channel_id,
                cursor=cursor,
                limit=200,  # Max per request
            )
            
            if not response["ok"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Slack API error: {response.get('error', 'Unknown error')}"
                )
            
            messages = response.get("messages", [])
            all_messages.extend(messages)
            
            # Check if there are more messages
            response_metadata = response.get("response_metadata", {})
            cursor = response_metadata.get("next_cursor")
            
            if not cursor:
                break
            
            # Rate limiting: Slack allows 1 request per second for conversations.history
            await asyncio.sleep(1.1)
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching Slack messages: {e}")
    
    return all_messages


@app.post("/papers/batch-ingest")
async def batch_ingest(payload: BatchIngestRequest) -> dict[str, Any]:
    """Batch ingest PDFs from a local directory or arXiv papers from a Slack channel.
    
    For directory:
    - For each PDF:
      1. Extract text to get title (from first page)
      2. Classify into category
      3. Generate abbreviation
      4. Summarize
      5. Save to database
    
    For Slack channel:
    - Fetch all messages from channel
    - Extract arXiv IDs/URLs from messages
    - Ingest each arXiv paper found
    """
    import glob
    import shutil
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
    
    results = []
    endpoint_config = _get_endpoint_config()
    base_url = endpoint_config["llm_base_url"]
    embed_base_url = endpoint_config["embedding_base_url"]
    api_key = endpoint_config["api_key"]
    
    # Verify endpoints before starting
    try:
        model = _resolve_model(base_url, api_key)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM endpoint not available: {e}")
    
    try:
        embed_model = _resolve_model(embed_base_url, api_key)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Embedding endpoint not available: {e}")
    
    llm_client = _get_async_openai_client(base_url, api_key)
    
    # Progress logging for Slack ingestion
    progress_log = []
    
    def log_progress(message: str):
        """Log progress message (for debugging and user feedback)."""
        progress_log.append(message)
        print(f"[Slack Ingest] {message}")  # Also print to Docker logs
    
    # Determine source type
    if payload.slack_channel:
        # Slack channel ingestion - wrap entire block to capture progress_log in errors
        try:
            log_progress("Starting Slack channel ingestion...")
            if not payload.slack_token:
                raise HTTPException(status_code=400, detail="Slack token is required for Slack channel ingestion")
            
            # Initialize Slack client
            log_progress("Initializing Slack client...")
            slack_client = WebClient(token=payload.slack_token)
            
            # Test token validity first
            log_progress("Testing Slack token validity...")
            try:
                auth_test = await asyncio.to_thread(slack_client.auth_test)
                if not auth_test["ok"]:
                    raise HTTPException(status_code=401, detail=f"Invalid Slack token: {auth_test.get('error', 'Unknown error')}")
                log_progress(f"✓ Token valid (workspace: {auth_test.get('team', 'Unknown')})")
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=401, detail=f"Failed to authenticate with Slack: {e}")
            
            # Resolve channel ID (wrap blocking call)
            log_progress(f"Resolving channel: {payload.slack_channel}")
            try:
                channel_id = await asyncio.to_thread(
                    _resolve_slack_channel_id,
                    slack_client,
                    payload.slack_channel
                )
                log_progress(f"✓ Channel resolved: {channel_id}")
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error resolving Slack channel: {e}")
            
            # Fetch all messages
            log_progress("Fetching messages from Slack channel (this may take a while for large channels)...")
            try:
                messages = await _fetch_slack_messages(slack_client, channel_id)
                log_progress(f"✓ Fetched {len(messages)} messages from channel")
                if not messages:
                    return {
                        "total": 0,
                        "success": 0,
                        "skipped": 0,
                        "errors": 0,
                        "results": [],
                        "progress_log": progress_log,
                        "message": "No messages found in channel",
                    }
            except HTTPException:
                raise
            except Exception as e:
                import traceback
                error_detail = f"Error fetching Slack messages: {str(e)}\n{traceback.format_exc()}"
                log_progress(f"✗ Error fetching messages: {error_detail}")
                raise HTTPException(status_code=500, detail=error_detail)
            
            # Extract arXiv IDs from all messages
            log_progress("Extracting arXiv IDs from messages...")
            all_arxiv_ids = set()
            try:
                for i, message in enumerate(messages):
                    if (i + 1) % 100 == 0:
                        log_progress(f"  Processing message {i+1}/{len(messages)}...")
                    text = message.get("text", "")
                    arxiv_ids = _extract_arxiv_ids_from_text(text)
                    all_arxiv_ids.update(arxiv_ids)
                log_progress(f"✓ Found {len(all_arxiv_ids)} unique arXiv papers")
            except Exception as e:
                import traceback
                error_detail = f"Error extracting arXiv IDs from messages: {str(e)}\n{traceback.format_exc()}"
                log_progress(f"✗ Error extracting arXiv IDs: {error_detail}")
                raise HTTPException(status_code=500, detail=error_detail)
            
            if not all_arxiv_ids:
                return {
                    "total": 0,
                    "success": 0,
                    "skipped": 0,
                    "errors": 0,
                    "results": [],
                    "progress_log": progress_log,
                }
            
            # Process each arXiv ID
            arxiv_ids_list = sorted(list(all_arxiv_ids))
            log_progress(f"Starting ingestion of {len(arxiv_ids_list)} papers...")
            for idx, arxiv_id in enumerate(arxiv_ids_list):
                log_progress(f"Processing paper {idx+1}/{len(arxiv_ids_list)}: {arxiv_id}")
                # Rate limiting: process in batches with delay
                await asyncio.sleep(0.5)  # Small delay between papers
                
                # Reset LiteLLM callbacks
                try:
                    import litellm
                    litellm.input_callback = []
                    litellm.success_callback = []
                    litellm.failure_callback = []
                    litellm._async_success_callback = []
                    litellm._async_failure_callback = []
                except Exception:
                    pass
                
                try:
                    # Check if already exists
                    log_progress(f"  [{arxiv_id}] Checking if paper already exists...")
                    if payload.skip_existing and db.get_paper_by_arxiv_id(arxiv_id):
                        log_progress(f"  [{arxiv_id}] ⏭ Skipped (already exists)")
                        results.append({
                            "file": arxiv_id,
                            "status": "skipped",
                            "reason": "already exists",
                        })
                        continue
                    
                    # Download and ingest arXiv paper (reuse existing single ingest logic)
                    log_progress(f"  [{arxiv_id}] Fetching metadata from arXiv...")
                    arxiv_client = arxiv.Client()
                    search = arxiv.Search(id_list=[arxiv_id])
                    arxiv_results = list(arxiv_client.results(search))
                    
                    if not arxiv_results:
                        log_progress(f"  [{arxiv_id}] ✗ arXiv paper not found")
                        results.append({
                            "file": arxiv_id,
                            "status": "error",
                            "reason": f"arXiv paper not found: {arxiv_id}",
                        })
                        continue
                    
                    arxiv_result = arxiv_results[0]
                    title = arxiv_result.title
                    authors = [author.name for author in arxiv_result.authors]
                    abstract = arxiv_result.summary
                    pdf_url = arxiv_result.pdf_url
                    log_progress(f"  [{arxiv_id}] ✓ Found: {title}")
                    
                    # Download PDF
                    log_progress(f"  [{arxiv_id}] Downloading PDF...")
                    pdf_path = await _download_arxiv_pdf(arxiv_id, pdf_url)
                    log_progress(f"  [{arxiv_id}] ✓ PDF downloaded")
                    
                    # Get existing categories
                    log_progress(f"  [{arxiv_id}] Classifying paper...")
                    tree = db.get_tree()
                    existing_categories = [n["name"] for n in tree if n["node_type"] == "category"]
                    
                    # Classify
                    classify_prompt = get_prompt("classify",
                        title=title,
                        abstract=abstract[:500],
                        existing_categories=", ".join(existing_categories) if existing_categories else "None"
                    )
                    classify_resp = await llm_client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": classify_prompt}],
                        max_tokens=50,
                        temperature=0.1,
                    )
                    category = classify_resp.choices[0].message.content.strip().strip('"\'')
                    log_progress(f"  [{arxiv_id}] ✓ Classified as: {category}")
                    
                    # Abbreviate
                    log_progress(f"  [{arxiv_id}] Generating abbreviation...")
                    abbrev_prompt = get_prompt("abbreviate", title=title)
                    abbrev_resp = await llm_client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": abbrev_prompt}],
                        max_tokens=20,
                        temperature=0.1,
                    )
                    abbreviation = abbrev_resp.choices[0].message.content.strip().strip('"\'')
                    log_progress(f"  [{arxiv_id}] ✓ Abbreviation: {abbreviation}")
                    
                    # Summarize
                    log_progress(f"  [{arxiv_id}] Summarizing paper (this may take a while)...")
                    prompt_id, prompt_hash, prompt_body = _load_prompt()
                    summary = await _paperqa_answer_async(
                        None,
                        pdf_path,
                        prompt_body,
                        base_url,
                        embed_base_url,
                        api_key,
                        f"openai/{model}",
                        f"openai/{embed_model}",
                        arxiv_id=arxiv_id,
                    )
                    log_progress(f"  [{arxiv_id}] ✓ Summary generated")
                    
                    # Save to database
                    log_progress(f"  [{arxiv_id}] Saving to database...")
                    db_paper_id = db.create_paper(
                        arxiv_id=arxiv_id,
                        title=title,
                        authors=authors,
                        abstract=abstract[:1000],
                        summary=summary,
                        pdf_path=pdf_path,
                    )
                    
                    # Add to tree
                    category_exists = any(n["name"] == category and n["node_type"] == "category" for n in tree)
                    if not category_exists:
                        category_node_id = f"cat_{uuid.uuid4().hex[:8]}"
                        db.add_tree_node(
                            node_id=category_node_id,
                            name=category,
                            node_type="category",
                            parent_id="root",
                        )
                        log_progress(f"  [{arxiv_id}] ✓ Created new category: {category}")
                    else:
                        category_node_id = next(n["node_id"] for n in tree if n["name"] == category and n["node_type"] == "category")
                    
                    paper_node_id = f"paper_{arxiv_id.replace('.', '_')}"
                    db.add_tree_node(
                        node_id=paper_node_id,
                        name=abbreviation,
                        node_type="paper",
                        parent_id=category_node_id,
                        paper_id=db_paper_id,
                    )
                    
                    log_progress(f"  [{arxiv_id}] ✓ Successfully ingested!")
                    results.append({
                        "file": arxiv_id,
                        "status": "success",
                        "paper_id": arxiv_id,
                        "title": title,
                        "category": category,
                        "abbreviation": abbreviation,
                    })
                    
                except Exception as e:
                    import traceback
                    error_msg = str(e)
                    log_progress(f"  [{arxiv_id}] ✗ Error: {error_msg}")
                    # Include traceback for debugging, but truncate if too long
                    tb = traceback.format_exc()
                    if len(tb) > 500:
                        tb = tb[:500] + "... (truncated)"
                    results.append({
                        "file": arxiv_id,
                        "status": "error",
                        "reason": f"{error_msg}\n{tb}",
                    })
        except HTTPException as e:
            # Include progress_log in HTTPException response
            import json
            detail = e.detail
            if isinstance(detail, str):
                detail = {"message": detail, "progress_log": progress_log}
            elif isinstance(detail, dict):
                detail["progress_log"] = progress_log
            else:
                detail = {"message": str(detail), "progress_log": progress_log}
            raise HTTPException(status_code=e.status_code, detail=detail)
        except Exception as e:
            # Catch any unexpected errors and include progress_log
            import traceback
            error_detail = f"Unexpected error during Slack ingestion: {str(e)}\n{traceback.format_exc()}"
            log_progress(f"✗ Fatal error: {error_detail}")
            raise HTTPException(
                status_code=500,
                detail={
                    "message": error_detail,
                    "progress_log": progress_log
                }
            )
    
    elif payload.directory:
        # Directory ingestion (existing logic)
        # Handle paths: /Users/hyl/xxx becomes /host_home/xxx in Docker
        dir_path = payload.directory
        home_dir = os.environ.get("HOME", "/root")
        
        # Check if running in Docker (home is /root) and path starts with typical macOS path
        if home_dir == "/root" and dir_path.startswith("/Users/"):
            # Extract the path after /Users/username/
            parts = dir_path.split("/")
            if len(parts) > 3:
                # /Users/username/rest -> /host_home/rest
                docker_path = "/host_home/" + "/".join(parts[3:])
                directory = pathlib.Path(docker_path)
            else:
                directory = pathlib.Path(dir_path)
        else:
            directory = pathlib.Path(dir_path)
        
        if not directory.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Directory not found: {payload.directory} (mapped to {directory})"
            )
        
        # Find all PDFs
        pdf_files = list(directory.glob("*.pdf")) + list(directory.glob("*.PDF"))
        if not pdf_files:
            raise HTTPException(status_code=400, detail=f"No PDF files found in: {payload.directory}")
        
        for pdf_file in pdf_files:
            # Reset LiteLLM callbacks to prevent MAX_CALLBACKS limit (30) accumulation
            try:
                import litellm
                litellm.input_callback = []
                litellm.success_callback = []
                litellm.failure_callback = []
                litellm._async_success_callback = []
                litellm._async_failure_callback = []
            except Exception:
                pass  # Ignore if litellm not available or attributes don't exist
            
            try:
                # Generate a unique ID based on filename
                filename = pdf_file.stem  # filename without extension
                paper_id = f"local_{hashlib.md5(filename.encode()).hexdigest()[:12]}"
                
                # Check if already exists
                if payload.skip_existing and db.get_paper_by_arxiv_id(paper_id):
                    results.append({
                        "file": pdf_file.name,
                        "status": "skipped",
                        "reason": "already exists",
                    })
                    continue
                
                # Copy PDF to storage
                storage_dir = pathlib.Path("storage/downloads")
                storage_dir.mkdir(parents=True, exist_ok=True)
                dest_path = storage_dir / f"{paper_id}.pdf"
                shutil.copy2(pdf_file, dest_path)
                pdf_path = str(dest_path)
                
                # Extract text from PDF using the async version
                try:
                    extract_result = await _paperqa_extract_pdf_async(pdf_file)
                    first_chunk = extract_result.get("text", "")[:2000]
                except Exception as ex:
                    results.append({
                        "file": pdf_file.name,
                        "status": "error",
                        "reason": f"Could not extract text from PDF: {ex}",
                    })
                    continue
                
                if not first_chunk:
                    results.append({
                        "file": pdf_file.name,
                        "status": "error",
                        "reason": "PDF extraction returned empty text",
                    })
                    continue
                
                # Use filename as title (clean it up)
                title = filename.replace("_", " ").replace("-", " ").strip()
                
                # Get existing categories for classification
                tree = db.get_tree()
                existing_categories = [n["name"] for n in tree if n["node_type"] == "category"]
                
                # Classify
                classify_prompt = get_prompt("classify", 
                    title=title, 
                    abstract=first_chunk[:500],
                    existing_categories=", ".join(existing_categories) if existing_categories else "None"
                )
                classify_resp = await llm_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": classify_prompt}],
                    max_tokens=50,
                    temperature=0.1,
                )
                category = classify_resp.choices[0].message.content.strip().strip('"\'')
                
                # Abbreviate
                abbrev_prompt = get_prompt("abbreviate", title=title)
                abbrev_resp = await llm_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": abbrev_prompt}],
                    max_tokens=20,
                    temperature=0.1,
                )
                abbreviation = abbrev_resp.choices[0].message.content.strip().strip('"\'')
                
                # Summarize (this also indexes the PDF) - use async version
                prompt_id, prompt_hash, prompt_body = _load_prompt()
                summary = await _paperqa_answer_async(
                    None,
                    pdf_path,
                    prompt_body,
                    base_url,
                    embed_base_url,
                    api_key,
                    f"openai/{model}",
                    f"openai/{embed_model}",
                    arxiv_id=paper_id,
                )
                
                # Save to database
                db_paper_id = db.create_paper(
                    arxiv_id=paper_id,
                    title=title,
                    authors=[],  # No author info from local PDFs
                    abstract=first_chunk[:1000],
                    summary=summary,
                    pdf_path=pdf_path,
                )
                
                # Add to tree
                category_exists = any(n["name"] == category and n["node_type"] == "category" for n in tree)
                if not category_exists:
                    category_node_id = f"cat_{uuid.uuid4().hex[:8]}"
                    db.add_tree_node(
                        node_id=category_node_id,
                        name=category,
                        node_type="category",
                        parent_id="root",
                    )
                else:
                    category_node_id = next(n["node_id"] for n in tree if n["name"] == category and n["node_type"] == "category")
                
                paper_node_id = f"paper_{paper_id.replace('.', '_')}"
                db.add_tree_node(
                    node_id=paper_node_id,
                    name=abbreviation,
                    node_type="paper",
                    parent_id=category_node_id,
                    paper_id=db_paper_id,
                )
                
                results.append({
                    "file": pdf_file.name,
                    "status": "success",
                    "paper_id": paper_id,
                    "title": title,
                    "category": category,
                    "abbreviation": abbreviation,
                })
                
            except Exception as e:
                results.append({
                    "file": pdf_file.name,
                    "status": "error",
                    "reason": str(e),
                })
    else:
        # Neither slack_channel nor directory provided
        raise HTTPException(status_code=400, detail="Either 'directory' or 'slack_channel' must be provided")
    
    # Calculate summary statistics
    success_count = sum(1 for r in results if r["status"] == "success")
    skip_count = sum(1 for r in results if r["status"] == "skipped")
    error_count = sum(1 for r in results if r["status"] == "error")
    
    # Determine total count based on source
    if payload.slack_channel:
        # For Slack, total is the number of unique arXiv IDs found
        total = len(set(r.get("file", "") for r in results if r["status"] != "skipped"))
    elif payload.directory:
        total = len(pdf_files)
    else:
        total = 0
    
    # Add final summary to progress log
    if payload.slack_channel:
        log_progress(f"✓ Completed: {success_count} success, {skip_count} skipped, {error_count} errors")
    
    return {
        "total": total,
        "success": success_count,
        "skipped": skip_count,
        "errors": error_count,
        "results": results,
        "progress_log": progress_log if payload.slack_channel else [],
    }


@app.post("/categories/rebalance")
async def rebalance_categories() -> dict[str, Any]:
    """Rebalance all categories that exceed the threshold.
    
    Iterates through all crowded categories and reclassifies papers
    to create more specific subcategories.
    """
    classification_config = _get_classification_config()
    if not classification_config["auto_reclassify_enabled"]:
        return {"message": "Auto-reclassification is disabled", "reclassified": []}
    
    threshold = classification_config["category_threshold"]
    categories = db.get_all_categories_with_counts()
    crowded_categories = [c for c in categories if c["paper_count"] > threshold]
    
    if not crowded_categories:
        return {"message": "No crowded categories found", "reclassified": []}
    
    endpoint_config = _get_endpoint_config()
    base_url = endpoint_config["llm_base_url"]
    api_key = endpoint_config["api_key"]
    model = _resolve_model(base_url, api_key)
    client = _get_async_openai_client(base_url, api_key)
    
    all_reclassified = []
    
    for category in crowded_categories:
        category_node_id = category["node_id"]
        category_name = category["name"]
        papers_in_category = db.get_papers_in_category(category_node_id)
        
        # Get existing categories (excluding the current one)
        tree = db.get_tree()
        existing_categories = [
            n["name"] for n in tree 
            if n["node_type"] == "category" and n["node_id"] != category_node_id
        ]
        
        for paper in papers_in_category:
            prompt = get_prompt(
                "classify",
                existing_categories=", ".join(existing_categories) if existing_categories else "None yet",
                title=paper["title"],
                abstract=(paper.get("abstract") or "")[:2000],
            )
            
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.1,
            )
            new_category = response.choices[0].message.content.strip()
            
            # Only move if category is different
            if new_category != category_name:
                new_cat_node = next(
                    (n for n in db.get_tree() if n["name"] == new_category and n["node_type"] == "category"),
                    None
                )
                
                if not new_cat_node:
                    new_cat_node_id = f"cat_{uuid.uuid4().hex[:8]}"
                    db.add_tree_node(
                        node_id=new_cat_node_id,
                        name=new_category,
                        node_type="category",
                        parent_id="root",
                    )
                else:
                    new_cat_node_id = new_cat_node["node_id"]
                
                db.move_paper_to_category(paper["node_id"], new_cat_node_id)
                all_reclassified.append({
                    "arxiv_id": paper["arxiv_id"],
                    "old_category": category_name,
                    "new_category": new_category,
                })
                
                if new_category not in existing_categories:
                    existing_categories.append(new_category)
    
    return {
        "message": f"Rebalanced {len(crowded_categories)} categories",
        "categories_processed": [c["name"] for c in crowded_categories],
        "reclassified": all_reclassified,
    }


@app.get("/papers/{arxiv_id}/cached-data")
def get_paper_cached_data(arxiv_id: str) -> dict[str, Any]:
    """Get all cached data for a paper (repos, refs, similar, queries, structured_summary).
    
    Used to restore state when selecting a paper in the GUI.
    """
    paper = db.get_paper_by_arxiv_id(arxiv_id)
    if not paper:
        raise HTTPException(status_code=404, detail=f"Paper {arxiv_id} not found")
    
    cached_data = db.get_paper_cached_data(paper["id"])
    structured_summary = db.get_paper_structured_summary(paper["id"])
    
    return {
        "arxiv_id": arxiv_id,
        "paper_id": paper["id"],
        "structured_summary": structured_summary,
        **cached_data,
    }


@app.get("/tree")
def get_tree() -> dict[str, Any]:
    """Get the full tree structure."""
    nodes = db.get_tree()
    
    # Build tree structure
    def build_tree(parent_id: Optional[str]) -> list[dict[str, Any]]:
        children = []
        for node in nodes:
            if node["parent_id"] == parent_id:
                child = {
                    "node_id": node["node_id"],
                    "name": node["name"],
                    "node_type": node["node_type"],
                }
                if node["node_type"] == "paper" and node["paper_id"]:
                    child["attributes"] = {
                        "arxivId": node.get("arxiv_id"),
                        "title": node.get("paper_title"),
                        "authors": node.get("authors") or [],
                        "summary": node.get("summary"),
                        "pdfPath": node.get("pdf_path"),
                    }
                grandchildren = build_tree(node["node_id"])
                if grandchildren:
                    child["children"] = grandchildren
                children.append(child)
        return children
    
    # Find root and build from there
    root = next((n for n in nodes if n["node_type"] == "root"), None)
    if not root:
        return {"name": "AI Papers", "children": []}
    
    return {
        "name": root["name"],
        "children": build_tree(root["node_id"]),
    }


@app.post("/tree/node")
def add_tree_node(payload: TreeNodeRequest) -> dict[str, str]:
    """Add a node to the tree."""
    db.add_tree_node(
        node_id=payload.node_id,
        name=payload.name,
        node_type=payload.node_type,
        parent_id=payload.parent_id,
        paper_id=payload.paper_id,
        position=payload.position,
    )
    return {"status": "ok"}


@app.delete("/tree/node/{node_id}")
def delete_tree_node(node_id: str) -> dict[str, str]:
    """Delete a node from the tree."""
    db.delete_tree_node(node_id)
    return {"status": "ok"}


@app.delete("/papers/{arxiv_id}")
def delete_paper(arxiv_id: str) -> dict[str, Any]:
    """Delete a paper and its tree node.
    
    Removes the paper from the database (which cascades to delete
    all associated data like references, similar papers, queries, repos)
    and removes the tree node.
    """
    paper = db.get_paper_by_arxiv_id(arxiv_id)
    if not paper:
        raise HTTPException(status_code=404, detail=f"Paper {arxiv_id} not found")
    
    # Construct node_id (format: paper_XXXX_XXXXXvX where dots become underscores)
    node_id = f"paper_{arxiv_id.replace('.', '_')}"
    
    # Delete paper (cascades to all related data)
    db.delete_paper(paper["id"])
    
    # Delete tree node
    db.delete_tree_node(node_id)
    
    return {
        "status": "ok",
        "deleted_arxiv_id": arxiv_id,
        "deleted_paper_id": paper["id"],
    }


class PrefetchRequest(BaseModel):
    arxiv_id: str
    title: str


@app.post("/papers/prefetch")
async def prefetch_paper_data(payload: PrefetchRequest) -> dict[str, Any]:
    """Prefetch repos, references, and similar papers in parallel.
    
    Call this after paper ingest to cache auxiliary data for faster UI loading.
    """
    apis_config = _get_external_apis_config()
    api_key = apis_config.get("semantic_scholar_api_key")
    
    # Run all three fetches in parallel
    repos_task = _prefetch_repos(payload.arxiv_id, payload.title, apis_config)
    refs_task = _get_semantic_scholar_references(payload.arxiv_id, api_key)
    similar_task = _get_semantic_scholar_recommendations(payload.arxiv_id, 10, api_key)
    
    repos, refs, similar = await asyncio.gather(
        repos_task, refs_task, similar_task,
        return_exceptions=True
    )
    
    return {
        "repos": repos if not isinstance(repos, Exception) else [],
        "references": refs if not isinstance(refs, Exception) else [],
        "similar": similar if not isinstance(similar, Exception) else [],
        "repos_error": str(repos) if isinstance(repos, Exception) else None,
        "refs_error": str(refs) if isinstance(refs, Exception) else None,
        "similar_error": str(similar) if isinstance(similar, Exception) else None,
    }


async def _prefetch_repos(arxiv_id: str, title: str, apis_config: dict) -> list[dict]:
    """Helper to prefetch repos for prefetch endpoint."""
    paper = db.get_paper_by_arxiv_id(arxiv_id)
    if paper:
        cached = db.get_cached_repos(paper["id"])
        if cached:
            return cached
    
    repos = []
    if apis_config["papers_with_code_enabled"]:
        pwc_repos = await _search_papers_with_code(title)
        repos.extend(pwc_repos)
    
    if apis_config["github_search_enabled"]:
        has_official = any(r.get("is_official") for r in repos)
        if not has_official:
            github_repos = await _search_github(title, apis_config.get("github_token"))
            repos.extend(github_repos)
    
    # Cache results
    if paper and repos:
        for repo in repos:
            db.cache_repo(
                paper_id=paper["id"],
                source=repo.get("source", "unknown"),
                repo_url=repo.get("url"),
                repo_name=repo.get("name"),
                stars=repo.get("stars"),
                is_official=repo.get("is_official", False),
            )
    
    return repos


# =============================================================================
# Repository Search (Feature 4)
# =============================================================================

async def _search_papers_with_code(title: str) -> list[dict[str, Any]]:
    """Search Papers With Code for repositories."""
    client = _get_http_client("paperswithcode", timeout=10.0)
    # Search for paper
    search_url = "https://paperswithcode.com/api/v1/papers/"
    response = await client.get(search_url, params={"q": title[:100]})
    if response.status_code != 200:
        return []
    
    data = response.json()
    results = data.get("results", [])
    if not results:
        return []
    
    repos = []
    for paper in results[:3]:  # Check top 3 matches
        paper_id = paper.get("id")
        if not paper_id:
            continue
        
        # Get repos for this paper
        repos_url = f"https://paperswithcode.com/api/v1/papers/{paper_id}/repositories/"
        repos_response = await client.get(repos_url)
        if repos_response.status_code == 200:
            repos_data = repos_response.json()
            for repo in repos_data.get("results", []):
                repos.append({
                    "source": "paperswithcode",
                    "repo_url": repo.get("url"),
                    "repo_name": repo.get("url", "").split("/")[-1] if repo.get("url") else None,
                    "stars": repo.get("stars"),
                    "is_official": repo.get("is_official", False),
                })
    return repos


async def _search_github(title: str, github_token: Optional[str] = None) -> list[dict[str, Any]]:
    """Search GitHub for repositories by paper title."""
    headers = {"Accept": "application/vnd.github.v3+json"}
    if github_token:
        headers["Authorization"] = f"token {github_token}"
    
    client = _get_http_client("github", timeout=10.0, headers={"Accept": "application/vnd.github.v3+json"})
    # Clean title for search
    clean_title = re.sub(r'[^\w\s]', '', title)[:50]
    search_url = "https://api.github.com/search/repositories"
    response = await client.get(
        search_url,
        params={"q": clean_title, "sort": "stars", "order": "desc", "per_page": 5},
        headers=headers,  # Override with auth if provided
    )
    if response.status_code != 200:
        return []
    
    data = response.json()
    repos = []
    for item in data.get("items", []):
        repos.append({
            "source": "github",
            "repo_url": item.get("html_url"),
            "repo_name": item.get("full_name"),
            "stars": item.get("stargazers_count"),
            "is_official": False,
        })
    return repos


@app.post("/repos/search")
async def search_repos(payload: RepoSearchRequest) -> dict[str, Any]:
    """Search for GitHub repositories associated with a paper."""
    apis_config = _get_external_apis_config()
    
    # Check cache first
    paper = db.get_paper_by_arxiv_id(payload.arxiv_id)
    if paper:
        cached = db.get_cached_repos(paper["id"])
        if cached:
            return {"repos": cached, "from_cache": True}
    
    repos = []
    
    # 1. Try Papers With Code first
    if apis_config["papers_with_code_enabled"]:
        pwc_repos = await _search_papers_with_code(payload.title)
        repos.extend(pwc_repos)
    
    # 2. Fall back to GitHub search if no official repos found
    if apis_config["github_search_enabled"]:
        has_official = any(r.get("is_official") for r in repos)
        if not has_official:
            github_repos = await _search_github(payload.title, apis_config.get("github_token"))
            repos.extend(github_repos)
    
    # Cache results
    if paper and repos:
        for repo in repos:
            db.cache_repo(
                paper_id=paper["id"],
                source=repo["source"],
                repo_url=repo.get("repo_url"),
                repo_name=repo.get("repo_name"),
                stars=repo.get("stars"),
                is_official=repo.get("is_official", False),
            )
    
    return {"repos": repos, "from_cache": False}


# =============================================================================
# References (Feature 5)
# =============================================================================

async def _get_semantic_scholar_references(arxiv_id: str, api_key: Optional[str] = None) -> list[dict[str, Any]]:
    """Get references from Semantic Scholar API."""
    # Clean arXiv ID (remove version suffix)
    clean_id = re.sub(r'v\d+$', '', arxiv_id)
    
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    
    client = _get_http_client("semantic_scholar", timeout=15.0)
    # First get the paper ID
    paper_url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{clean_id}"
    params = {"fields": "references.title,references.authors,references.year,references.externalIds"}
    response = await client.get(paper_url, params=params, headers=headers)
    
    if response.status_code == 429:
        print("Semantic Scholar rate limited. Consider adding an API key to config/paperqa.yaml")
        return []
    
    if response.status_code != 200:
        print(f"Semantic Scholar returned {response.status_code}: {response.text[:200]}")
        return []
    
    data = response.json()
    references = []
    for ref in data.get("references", []):
        arxiv_ref_id = None
        external_ids = ref.get("externalIds", {})
        if external_ids and "ArXiv" in external_ids:
            arxiv_ref_id = external_ids["ArXiv"]
        
        authors = [a.get("name", "") for a in ref.get("authors", [])]
        references.append({
            "cited_title": ref.get("title", "Unknown"),
            "cited_arxiv_id": arxiv_ref_id,
            "cited_authors": authors,
            "cited_year": ref.get("year"),
        })
    return references


@app.post("/references/fetch")
async def fetch_references(payload: ReferencesRequest) -> dict[str, Any]:
    """Fetch references for a paper."""
    paper = db.get_paper_by_arxiv_id(payload.arxiv_id)
    
    # Check cache first
    if paper:
        cached = db.get_references(paper["id"])
        if cached:
            return {"references": cached, "from_cache": True}
    
    apis_config = _get_external_apis_config()
    references = []
    
    # Try Semantic Scholar first
    if apis_config["semantic_scholar_enabled"]:
        api_key = apis_config.get("semantic_scholar_api_key")
        references = await _get_semantic_scholar_references(payload.arxiv_id, api_key)
    
    # TODO: Add LaTeX parsing fallback
    # TODO: Add PDF text parsing fallback
    
    # Cache references
    if paper and references:
        for ref in references:
            db.add_reference(
                source_paper_id=paper["id"],
                cited_title=ref["cited_title"],
                cited_arxiv_id=ref.get("cited_arxiv_id"),
                cited_authors=ref.get("cited_authors"),
                cited_year=ref.get("cited_year"),
            )
        # Refresh from DB to get IDs
        references = db.get_references(paper["id"])
    
    return {"references": references, "from_cache": False}


@app.post("/references/explain")
async def explain_reference(payload: ExplainReferenceRequest) -> dict[str, Any]:
    """Generate an explanation for a reference using LLM."""
    # Check cache
    refs = db.get_references(0)  # We need to query by ref ID
    ref = next((r for r in refs if r.get("id") == payload.reference_id), None) if refs else None
    if ref and ref.get("explanation"):
        return {"explanation": ref["explanation"], "from_cache": True}
    
    endpoint_config = _get_endpoint_config()
    base_url = endpoint_config["llm_base_url"]
    api_key = endpoint_config["api_key"]
    model = _resolve_model(base_url, api_key)
    client = _get_async_openai_client(base_url, api_key)
    
    context_section = ""
    if payload.citation_context:
        context_section = f"\nCitation context: \"{payload.citation_context}\""
    
    prompt = get_prompt(
        "explain_reference",
        source_title=payload.source_paper_title,
        cited_title=payload.cited_title,
        context_section=context_section,
    )
    
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.3,
    )
    explanation = response.choices[0].message.content.strip()
    
    # Cache the explanation
    if payload.reference_id:
        db.update_reference_explanation(payload.reference_id, explanation)
    
    return {"explanation": explanation, "from_cache": False}


# =============================================================================
# Similar Papers (Feature 6) - Uses Semantic Scholar Recommendations API
# =============================================================================

async def _get_semantic_scholar_recommendations(arxiv_id: str, limit: int = 10, api_key: Optional[str] = None) -> list[dict[str, Any]]:
    """Get paper recommendations from Semantic Scholar API (searches the internet)."""
    # Clean arXiv ID (remove version suffix like v1, v2)
    clean_id = re.sub(r'v\d+$', '', arxiv_id)
    
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    
    client = _get_http_client("semantic_scholar", timeout=15.0)
    # Use the Recommendations API to find similar papers from Semantic Scholar's 200M+ paper database
    rec_url = f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/arXiv:{clean_id}"
    params = {
        "fields": "title,abstract,authors,year,externalIds,citationCount,url",
        "limit": limit,
        "from": "all-cs",  # Search all CS papers, use "recent" for recent papers only
    }
    
    response = await client.get(rec_url, params=params, headers=headers)
    
    if response.status_code != 200:
        print(f"Semantic Scholar Recommendations API returned {response.status_code}: {response.text}")
        return []
    
    data = response.json()
    recommendations = []
    
    for paper in data.get("recommendedPapers", []):
        arxiv_ref_id = None
        external_ids = paper.get("externalIds", {})
        if external_ids and "ArXiv" in external_ids:
            arxiv_ref_id = external_ids["ArXiv"]
        
        authors = [a.get("name", "") for a in paper.get("authors", [])]
        recommendations.append({
            "arxiv_id": arxiv_ref_id,
            "title": paper.get("title", "Unknown"),
            "abstract": paper.get("abstract", ""),
            "authors": authors,
            "year": paper.get("year"),
            "citation_count": paper.get("citationCount", 0),
            "url": paper.get("url", ""),
            "source": "semantic_scholar",
        })
    
    return recommendations


@app.post("/papers/similar")
async def find_similar_papers(payload: SimilarPapersRequest) -> dict[str, Any]:
    """Find similar papers using Semantic Scholar Recommendations API (searches internet)."""
    # Get UI config for limit
    ui_config = _get_ui_config()
    limit = ui_config.get("max_similar_papers", 10)
    
    # Check cache first (optional - paper may not be in local DB)
    paper = db.get_paper_by_arxiv_id(payload.arxiv_id)
    if paper:
        cached = db.get_cached_similar_papers(paper["id"])
        if cached:
            return {"similar_papers": cached, "from_cache": True}
    
    # Query Semantic Scholar Recommendations API (searches the internet)
    apis_config = _get_external_apis_config()
    api_key = apis_config.get("semantic_scholar_api_key")
    similar = await _get_semantic_scholar_recommendations(payload.arxiv_id, limit, api_key)
    
    # Cache results if paper is in our DB
    if paper and similar:
        for s in similar:
            db.cache_similar_paper(
                paper_id=paper["id"],
                similar_arxiv_id=s.get("arxiv_id"),
                similar_title=s["title"],
                similarity_score=None,  # No similarity score from API
            )
    
    return {"similar_papers": similar, "from_cache": False}
