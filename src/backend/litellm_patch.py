"""LiteLLM patch to prevent callback accumulation and logging worker issues.

This module MUST be imported before any PaperQA2 or LiteLLM imports.
It disables LiteLLM's internal logging and callback mechanisms that cause:
1. MAX_CALLBACKS limit (30) warnings during batch operations
2. Async task cleanup warnings from LoggingWorker
3. Memory leaks from accumulated callbacks

Usage:
    # At the very top of app.py, before any other imports:
    from litellm_patch import patch_litellm
    patch_litellm()
"""
from __future__ import annotations

import os


def patch_litellm() -> None:
    """Patch LiteLLM to disable problematic logging and callbacks.
    
    Must be called BEFORE importing litellm, paperqa, or any module that
    uses them.
    """
    # Set environment variables before litellm is imported
    # These prevent litellm from initializing logging infrastructure
    os.environ["LITELLM_LOG"] = "ERROR"  # Only log errors
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"  # Avoid external calls
    
    try:
        import litellm
        
        # Disable all logging and telemetry
        litellm.suppress_debug_info = True
        litellm.set_verbose = False
        litellm.drop_params = True  # Don't raise errors for unsupported params
        
        # Disable callbacks entirely
        litellm.input_callback = []
        litellm.success_callback = []
        litellm.failure_callback = []
        litellm._async_success_callback = []
        litellm._async_failure_callback = []
        
        # Patch the logging callback manager to prevent callback additions
        if hasattr(litellm, 'logging_callback_manager'):
            manager = litellm.logging_callback_manager
            
            # Clear existing callbacks
            for attr in ['callbacks', '_callbacks', 'callback_list',
                         'success_callbacks', 'failure_callbacks']:
                if hasattr(manager, attr):
                    setattr(manager, attr, [])
            
            # Monkey-patch add_callback to be a no-op
            # This prevents PaperQA2 from adding new callbacks
            def no_op_add_callback(*args, **kwargs):
                pass
            
            if hasattr(manager, 'add_callback'):
                manager.add_callback = no_op_add_callback
            if hasattr(manager, 'add_callbacks'):
                manager.add_callbacks = no_op_add_callback
        
        # Also patch the callbacks module if it exists
        try:
            from litellm import callbacks as litellm_callbacks
            if hasattr(litellm_callbacks, 'add_callback'):
                litellm_callbacks.add_callback = lambda *args, **kwargs: None
        except (ImportError, AttributeError):
            pass
        
        print("[litellm_patch] LiteLLM patched: logging and callbacks disabled")
        
    except ImportError:
        # LiteLLM not installed, nothing to patch
        pass


def reset_litellm_callbacks() -> None:
    """Reset LiteLLM callbacks - call this between batch operations.
    
    This is a fallback for any callbacks that slip through the patch.
    """
    try:
        import litellm
    except ImportError:
        return
    
    # Clear all callback lists
    litellm.input_callback = []
    litellm.success_callback = []
    litellm.failure_callback = []
    litellm._async_success_callback = []
    litellm._async_failure_callback = []
    
    # Clear callback manager
    if hasattr(litellm, 'logging_callback_manager'):
        manager = litellm.logging_callback_manager
        for attr in ['callbacks', '_callbacks', 'callback_list']:
            if hasattr(manager, attr):
                setattr(manager, attr, [])
