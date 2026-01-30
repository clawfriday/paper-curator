"""Test clustering functionality."""

import sys
from pathlib import Path
from typing import Any

# Add src/backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "backend"))

from clustering import build_tree_from_clusters, _get_clustering_config
import db


def test_tree_building():
    """Test that tree building works correctly with all assertions."""
    print("=" * 60)
    print("Running tree building test...")
    print("=" * 60)
    
    # 1. Assert: All papers have their embedding
    print("\n1. Checking that all papers have embeddings...")
    papers = db.get_all_papers()
    
    if len(papers) == 0:
        print("  ⚠ No papers in database, skipping test")
        return True
    
    papers_with_embeddings = [p for p in papers if p.get("embedding") is not None]
    papers_without_embeddings = [p for p in papers if p.get("embedding") is None]
    
    if papers_without_embeddings:
        print(f"  ✗ Found {len(papers_without_embeddings)} papers without embeddings:")
        for p in papers_without_embeddings[:5]:  # Show first 5
            print(f"    - Paper ID {p['id']}: {p.get('title', 'No title')[:50]}")
        if len(papers_without_embeddings) > 5:
            print(f"    ... and {len(papers_without_embeddings) - 5} more")
        raise AssertionError(f"{len(papers_without_embeddings)} papers are missing embeddings")
    
    print(f"  ✓ All {len(papers)} papers have embeddings")
    
    # 2. Build the tree with all papers
    print("\n2. Building tree from all papers...")
    result = build_tree_from_clusters()
    
    assert "total_papers" in result, "Result should contain total_papers"
    assert "total_clusters" in result, "Result should contain total_clusters"
    
    if result["total_papers"] < 2:
        print(f"  ⚠ Only {result['total_papers']} papers available, need at least 2 for clustering")
        return True
    
    # Extract tree structure (frontend format: {name, children: [...]})
    tree_structure = {
        "name": result.get("name", "AI Papers"),
        "children": result.get("children", []),
    }
    total_papers = result["total_papers"]
    branching_factor = _get_clustering_config()["branching_factor"]
    
    print(f"  ✓ Built tree with {total_papers} papers")
    print(f"  ✓ Branching factor: {branching_factor}")
    
    # 3. Assert: In the tree that is built, it contains all papers
    print("\n3. Verifying all papers are in the tree...")
    
    def collect_all_paper_ids(node: dict[str, Any]) -> list[int]:
        """Recursively collect all paper IDs from the frontend format tree."""
        paper_ids = []
        # If this is a paper node, collect its paper_id
        if node.get("node_type") == "paper" and node.get("paper_id"):
            paper_ids.append(node["paper_id"])
        # Recursively collect from children
        if node.get("children"):
            for child in node["children"]:
                paper_ids.extend(collect_all_paper_ids(child))
        return paper_ids
    
    all_paper_ids_in_tree = collect_all_paper_ids(tree_structure)
    all_paper_ids_in_db = [p["id"] for p in papers_with_embeddings]
    
    missing_papers = set(all_paper_ids_in_db) - set(all_paper_ids_in_tree)
    extra_papers = set(all_paper_ids_in_tree) - set(all_paper_ids_in_db)
    
    if missing_papers:
        print(f"  ✗ Missing papers in tree: {missing_papers}")
        raise AssertionError(f"Tree is missing {len(missing_papers)} papers: {list(missing_papers)[:10]}")
    
    if extra_papers:
        print(f"  ✗ Extra papers in tree: {extra_papers}")
        raise AssertionError(f"Tree contains {len(extra_papers)} papers not in DB: {list(extra_papers)[:10]}")
    
    print(f"  ✓ All {len(all_paper_ids_in_tree)} papers are present in the tree")
    
    # 4. Assert: Each node has at most branching_factor number of child nodes
    print("\n4. Verifying branching factor constraint...")
    
    def check_branching_factor(node: dict[str, Any], max_children: int) -> list[dict]:
        """Recursively check that no node has more than max_children."""
        violations = []
        # Count children in this node
        children = node.get("children", [])
        if len(children) > max_children:
            violations.append({
                "node_id": node.get("node_id", "unknown"),
                "name": node.get("name", "unknown"),
                "children_count": len(children),
                "max_allowed": max_children
            })
        # Recursively check children
        for child in children:
            violations.extend(check_branching_factor(child, max_children))
        return violations
    
    violations = check_branching_factor(tree_structure, branching_factor)
    
    if violations:
        print(f"  ✗ Found {len(violations)} nodes violating branching factor:")
        for v in violations[:5]:
            print(f"    - {v['cluster_id']}: {v['children_count']} children (max: {v['max_allowed']})")
        if len(violations) > 5:
            print(f"    ... and {len(violations) - 5} more")
        raise AssertionError(f"{len(violations)} nodes exceed branching factor of {branching_factor}")
    
    print(f"  ✓ All nodes respect branching factor (max {branching_factor} children)")
    
    # 5. Assert: Each node is not empty
    print("\n5. Verifying no empty nodes...")
    
    def check_empty_nodes(node: dict[str, Any]) -> list[dict]:
        """Recursively check that no node is empty."""
        empty_nodes = []
        node_type = node.get("node_type")
        children = node.get("children", [])
        
        if node_type == "paper":
            # Paper nodes must have paper_id
            if not node.get("paper_id"):
                empty_nodes.append({
                    "node_id": node.get("node_id", "unknown"),
                    "name": node.get("name", "unknown"),
                    "type": "paper",
                    "issue": "missing paper_id"
                })
        elif node_type == "category":
            # Category nodes must have children
            if len(children) == 0:
                empty_nodes.append({
                    "node_id": node.get("node_id", "unknown"),
                    "name": node.get("name", "unknown"),
                    "type": "category",
                    "issue": "no children"
                })
        
        # Recursively check children
        for child in children:
            empty_nodes.extend(check_empty_nodes(child))
        
        return empty_nodes
    
    empty_nodes = check_empty_nodes(tree_structure)
    
    if empty_nodes:
        print(f"  ✗ Found {len(empty_nodes)} empty or invalid nodes:")
        for n in empty_nodes[:5]:
            print(f"    - {n['cluster_id']} ({n['type']}): {n['issue']}")
        if len(empty_nodes) > 5:
            print(f"    ... and {len(empty_nodes) - 5} more")
        raise AssertionError(f"Found {len(empty_nodes)} empty or invalid nodes")
    
    print(f"  ✓ All nodes are non-empty and valid")
    
    # Summary
    print("\n" + "=" * 60)
    print("✓ All assertions passed!")
    print(f"  - Papers with embeddings: {len(papers_with_embeddings)}")
    print(f"  - Papers in tree: {len(all_paper_ids_in_tree)}")
    print(f"  - Branching factor: {branching_factor}")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    test_tree_building()

