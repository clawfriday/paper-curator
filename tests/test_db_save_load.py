"""Test database save/load functionality."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "backend"))

import clustering
import db


def test_db_save_load():
    """Test saving and loading tree from database."""
    print("=" * 60)
    print("Testing Database Save/Load Functionality")
    print("=" * 60)

    # Build tree
    print("\n1. Building tree...")
    result = clustering.build_tree_from_clusters()
    tree_structure = {
        "name": result.get("name", "AI Papers"),
        "children": result.get("children", []),
    }

    # Save to database
    print("2. Saving tree to database...")
    clustering.write_tree_to_database(tree_structure)

    # Load from database
    print("3. Loading tree from database...")
    loaded_tree = db.get_tree()

    # Verify structure
    assert loaded_tree.get("name") == tree_structure.get("name"), "Root name mismatch"
    assert "children" in loaded_tree, "Missing children field"

    # Count papers in loaded tree
    def count_papers(node):
        count = 0
        if node.get("node_type") == "paper" and node.get("paper_id"):
            count = 1
        if node.get("children"):
            for child in node["children"]:
                count += count_papers(child)
        return count

    original_count = result.get("total_papers", 0)
    loaded_count = count_papers(loaded_tree)

    print(f"4. Verifying paper count...")
    print(f"   Original papers: {original_count}")
    print(f"   Loaded papers: {loaded_count}")
    assert original_count == loaded_count, f"Paper count mismatch: {original_count} != {loaded_count}"

    print("\n" + "=" * 60)
    print("✓ Database save/load test passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_db_save_load()
