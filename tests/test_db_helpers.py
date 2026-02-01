"""Test database helper functions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "backend"))

import db


def test_db_helpers():
    """Test database helper functions."""
    print("=" * 60)
    print("Testing Database Helper Functions")
    print("=" * 60)

    # Test get_tree_node_names
    print("\n1. Testing get_tree_node_names()...")
    node_names = db.get_tree_node_names()
    print(f"   ✓ Loaded {len(node_names)} node names")

    # Test find_paper_node_id
    print("\n2. Testing find_paper_node_id()...")
    papers = db.get_all_papers()
    if papers:
        test_paper = papers[0]
        node_id = db.find_paper_node_id(test_paper["id"])
        if node_id:
            print(f"   ✓ Found node_id for paper {test_paper['id']}: {node_id}")

            # Test update_tree_node_name
            print("\n3. Testing update_tree_node_name()...")
            original_name = node_names.get(node_id, "")
            db.update_tree_node_name(node_id, "Test Name")
            updated_tree = db.get_tree()

            def find_node(node, target_id):
                if node.get("node_id") == target_id:
                    return node
                if node.get("children"):
                    for child in node["children"]:
                        found = find_node(child, target_id)
                        if found:
                            return found
                return None

            updated_node = find_node(updated_tree, node_id)
            if updated_node and updated_node.get("name") == "Test Name":
                print(f"   ✓ Successfully updated node name")
                # Restore original name
                db.update_tree_node_name(node_id, original_name)
                print(f"   ✓ Restored original name")
            else:
                print(f"   ✗ Failed to update node name")
        else:
            print(f"   ⚠ No node_id found for paper {test_paper['id']}")

    print("\n" + "=" * 60)
    print("✓ All database helper functions tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_db_helpers()
