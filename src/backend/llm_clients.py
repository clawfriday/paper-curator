"""LLM client helpers and model resolution."""
from __future__ import annotations

from functools import lru_cache

from openai import AsyncOpenAI, OpenAI


def get_openai_client(base_url: str, api_key: str) -> OpenAI:
    """Create OpenAI client with ngrok header support if needed."""
    if "ngrok" in base_url.lower():
        import httpx
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


def get_async_openai_client(base_url: str, api_key: str) -> AsyncOpenAI:
    """Create AsyncOpenAI client with ngrok header support if needed."""
    if "ngrok" in base_url.lower():
        import httpx
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
def resolve_model(base_url: str, api_key: str) -> str:
    """Resolve model name from endpoint, with ngrok free tier workaround."""
    client = get_openai_client(base_url, api_key)
    try:
        models = client.models.list()
        model_ids = sorted([model.id for model in models.data if getattr(model, "id", None)])
        assert model_ids, "No models returned by OpenAI-compatible endpoint."
        return model_ids[0]
    except Exception as e:
        error_msg = str(e)
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


def reset_litellm_callbacks() -> None:
    """Reset LiteLLM callbacks to prevent accumulation.
    
    LiteLLM has a MAX_CALLBACKS limit of 30. PaperQA2 adds callbacks for each
    query, which can accumulate and hit this limit during batch operations.
    This function clears all callback lists to prevent the warning.
    """
    try:
        import litellm
    except ImportError:
        return
    
    # Clear main callback lists
    litellm.input_callback = []
    litellm.success_callback = []
    litellm.failure_callback = []
    litellm._async_success_callback = []
    litellm._async_failure_callback = []
    
    # Also clear the logging callback manager if it exists
    try:
        if hasattr(litellm, 'logging_callback_manager'):
            manager = litellm.logging_callback_manager
            if hasattr(manager, 'callbacks'):
                manager.callbacks = []
            if hasattr(manager, '_callbacks'):
                manager._callbacks = []
    except Exception:
        pass  # Ignore if attribute structure is different


def get_llm_config() -> dict[str, str]:
    """Load LLM configuration from config (with DB overrides).
    
    Returns:
        dict with 'base_url', 'api_key', 'model'
    """
    from config import _get_endpoint_config
    
    # Get config from centralized config module (supports DB overrides)
    endpoint_config = _get_endpoint_config()
    
    base_url = endpoint_config["llm_base_url"]
    api_key = endpoint_config["api_key"]
    model = resolve_model(base_url, api_key)
    
    return {"base_url": base_url, "api_key": api_key, "model": model}
