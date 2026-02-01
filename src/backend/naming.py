"""Contrastive naming for tree nodes based on children and siblings.

This module provides LLM-powered naming for the hierarchical paper classification tree.
It works with the frontend-format tree structure from clustering.py:

Tree structure:
{
    "name": "AI Papers",
    "children": [
        {
            "name": "Cluster node_xxx",  # Will be renamed by LLM
            "node_id": "node_xxx",
            "node_type": "category",
            "children": [...]
        },
        {
            "name": "Paper 123",
            "node_id": "node_yyy",
            "node_type": "paper",
            "paper_id": 123
        }
    ]
}

Naming process:
1. Process tree levels bottom-up (deepest categories first)
2. For each category node, gather:
   - Children content (paper summaries for leaf level, category names for higher levels)
   - Sibling content (other categories at same level)
3. Call LLM to generate distinguishing name using contrastive prompt
4. Update node name in database and in-memory tree
5. Proceed to next level (names from previous level are now available)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
import db

logger = logging.getLogger(__name__)


def _get_prompt(prompt_name: str, **kwargs: Any) -> str:
    """Load prompt template from prompts.json."""
    import json
    import os
    from pathlib import Path
    
    prompt_paths = [
        Path("prompts/prompts.json"),
        Path("../prompts/prompts.json"),
        Path("../../prompts/prompts.json"),
    ]
    
    for path in prompt_paths:
        if path.exists():
            with open(path, "r") as f:
                prompts = json.load(f)
                if prompt_name in prompts:
                    template = prompts[prompt_name]["template"]
                    # Replace variables
                    for key, value in kwargs.items():
                        template = template.replace(f"{{{key}}}", str(value))
                    return template
    
    raise ValueError(f"Prompt '{prompt_name}' not found")


def _get_node_summaries_or_names(
    node_ids: list[str], 
    is_leaf_level: bool,
    tree: dict[str, Any] | None = None
) -> list[str]:
    """Get summaries (for leaf level) or names (for higher levels) for nodes.
    
    For leaf-level category nodes (parents of paper nodes):
    - Returns paper summaries from the database
    
    For higher-level category nodes:
    - Returns the (already named) category names of children
    
    Args:
        node_ids: List of node IDs (node_xxx format)
        is_leaf_level: True if children are paper nodes, False if children are category nodes
        tree: Optional tree structure to search in (if None, loads from database)
        
    Returns:
        List of summaries or names for the requested nodes
    """
    if tree is None:
        tree = db.get_tree()
    
    result = []
    
    def find_node(node: dict[str, Any], target_id: str) -> dict[str, Any] | None:
        """Recursively find node by node_id."""
        if node.get("node_id") == target_id:
            return node
        if node.get("children"):
            for child in node["children"]:
                found = find_node(child, target_id)
                if found:
                    return found
        return None
    
    for node_id in node_ids:
        node = find_node(tree, node_id)
        if not node:
            continue
        
        if is_leaf_level and node.get("node_type") == "paper":
            # Get paper summary
            if node.get("paper_id"):
                paper = db.get_paper_by_id(node["paper_id"])
                if paper and paper.get("summary"):
                    result.append(paper["summary"])
        else:
            # Get node name
            if node.get("name"):
                result.append(node["name"])
    
    return result


async def name_tree_nodes() -> dict[str, Any]:
    """Name all tree nodes using contrastive naming (bottom-up).
    
    Process:
    1. Traverse tree level by level from bottom to top
    2. For each parent node at level i:
       - Collect summaries (if leaf level) or names (if higher level) of children
       - Collect summaries/names of sibling nodes (same grandparent)
       - Call LLM with contrastive prompt to generate distinguishing name
       - Update node name in database
    
    Returns:
        Dictionary with results:
        {
            "nodes_named": int,
            "levels_processed": int
        }
    """
    # Import functions from app module at function level to avoid circular imports
    import app as app_module
    _get_endpoint_config = app_module._get_endpoint_config
    _resolve_model = app_module._resolve_model
    _get_async_openai_client = app_module._get_async_openai_client
    
    tree = db.get_tree()
    
    # Build tree structure to traverse level by level
    def get_tree_levels() -> list[list[dict[str, Any]]]:
        """Get tree nodes organized by level (bottom-up) from nested structure."""
        # Collect all category nodes with their depths
        nodes_with_depth = []
        
        def traverse(node: dict[str, Any], depth: int = 0) -> None:
            """Recursively traverse tree and collect category nodes with depths."""
            if node.get("node_type") == "category":
                nodes_with_depth.append((node, depth))
            
            if node.get("children"):
                for child in node["children"]:
                    traverse(child, depth + 1)
        
        traverse(tree)
        
        if not nodes_with_depth:
            return []
        
        # Find max depth
        max_depth = max(depth for _, depth in nodes_with_depth)
        
        # Group nodes by depth (bottom-up, so reverse)
        levels = [[] for _ in range(max_depth + 1)]
        for node, depth in nodes_with_depth:
            levels[max_depth - depth].append(node)  # Reverse: deepest first
        
        return [level for level in levels if level]  # Remove empty levels
    
    levels = get_tree_levels()
    
    if not levels:
        return {
            "nodes_named": 0,
            "levels_processed": 0,
            "message": "No category nodes found to name",
        }
    
    # Get LLM config
    endpoint_config = _get_endpoint_config()
    base_url = endpoint_config["llm_base_url"]
    api_key = endpoint_config["api_key"]
    model = _resolve_model(base_url, api_key)
    client = _get_async_openai_client(base_url, api_key)
    
    nodes_named = 0
    
    # Helper function to update name in in-memory tree
    def update_name_in_memory(node: dict[str, Any], target_id: str, new_name: str) -> None:
        """Update name in in-memory tree structure."""
        if node.get("node_id") == target_id:
            node["name"] = new_name
        if node.get("children"):
            for child in node["children"]:
                update_name_in_memory(child, target_id, new_name)
    
    # Helper function to find node and its parent
    def find_node_and_parent(target_id: str, tree_node: dict[str, Any], parent: dict[str, Any] | None = None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Find node and its parent."""
        if tree_node.get("node_id") == target_id:
            return tree_node, parent
        if tree_node.get("children"):
            for child in tree_node["children"]:
                result = find_node_and_parent(target_id, child, tree_node)
                if result[0]:
                    return result
        return None, None
    
    # Process each level from bottom to top (sequentially)
    for level_idx, level_nodes in enumerate(levels):
        is_leaf_level = (level_idx == 0)  # First level (deepest) is leaf level
        
        print(f"\nProcessing level {level_idx + 1}/{len(levels)} ({len(level_nodes)} nodes)...")
        
        # Define async function to name a single node
        # Note: This function captures is_leaf_level, tree, client, model from outer scope
        async def name_single_node(node: dict[str, Any], leaf_level: bool = is_leaf_level) -> tuple[str, bool, str]:
            """Name a single node. Returns (node_id, success, name_or_error)."""
            node_id = node["node_id"]
            
            # Get children of this node
            children_ids = []
            if node.get("children"):
                children_ids = [c["node_id"] for c in node["children"]]
            
            if not children_ids:
                return (node_id, False, "No children")
            
            # Get summaries/names of children (use current tree state)
            children_content = _get_node_summaries_or_names(children_ids, leaf_level, tree)
            
            if not children_content:
                return (node_id, False, "No children content")
            
            # Find siblings (nodes with same parent)
            node_found, parent_node = find_node_and_parent(node_id, tree)
            
            siblings_ids = []
            if parent_node and parent_node.get("children"):
                siblings_ids = [
                    s["node_id"] for s in parent_node["children"]
                    if s.get("node_id") != node_id and s.get("node_type") == "category"
                ]
            
            # Get summaries/names of siblings (use current tree state)
            siblings_content = _get_node_summaries_or_names(siblings_ids, leaf_level, tree)
            
            # Build prompt
            children_text = "\n\n".join([
                f"Child {i+1}:\n{content[:500]}"  # Truncate to avoid token limits
                for i, content in enumerate(children_content)
            ])
            
            siblings_text = "\n\n".join([
                f"Sibling {i+1}:\n{content[:500]}"
                for i, content in enumerate(siblings_content)
            ]) if siblings_content else "None"
            
            # Retry logic for LLM naming
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    prompt = _get_prompt(
                        "node_naming",
                        children_summaries=children_text,
                        sibling_summaries=siblings_text,
                    )
                    
                    response = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=50,
                        temperature=0.1,
                    )
                    
                    new_name = response.choices[0].message.content.strip().strip('"\'')
                    
                    # Validate that we got a meaningful name (not empty, not generic)
                    if new_name and not new_name.startswith("Category_") and len(new_name) > 3:
                        # Update node name in database
                        db.update_tree_node_name(node_id, new_name)
                        # Update name in in-memory tree to keep it in sync
                        update_name_in_memory(tree, node_id, new_name)
                        return (node_id, True, new_name)
                    else:
                        raise ValueError(f"Invalid name generated: '{new_name}'")
                    
                except Exception as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        await asyncio.sleep(1)  # Brief delay before retry
                    else:
                        # Try to generate a fallback name based on children
                        try:
                            if children_content:
                                # Use first few words from first child summary as fallback
                                first_child_words = children_content[0].split()[:3]
                                fallback_name = " ".join(first_child_words).title()[:30]
                                if fallback_name:
                                    db.update_tree_node_name(node_id, fallback_name)
                                    update_name_in_memory(tree, node_id, fallback_name)
                                    return (node_id, True, fallback_name)
                        except Exception as fallback_error:
                            return (node_id, False, f"Fallback failed: {fallback_error}")
                        return (node_id, False, str(e))
            
            return (node_id, False, "Max retries exceeded")
        
        # Process all nodes in this level in parallel
        results = await asyncio.gather(*[name_single_node(node) for node in level_nodes])
        
        # Count successes and log results
        level_named = 0
        for node_id, success, name_or_error in results:
            if success:
                level_named += 1
                nodes_named += 1
                print(f"  ✓ Named {node_id} as '{name_or_error}'")
            else:
                print(f"  ✗ Failed to name {node_id}: {name_or_error}")
        
        print(f"Level {level_idx + 1} complete: {level_named}/{len(level_nodes)} nodes named")
        
        # After processing this level, tree is updated in-memory
        # For next level, siblings will have updated names
    
    return {
        "nodes_named": nodes_named,
        "levels_processed": len(levels),
        "message": f"Named {nodes_named} nodes across {len(levels)} levels",
    }
