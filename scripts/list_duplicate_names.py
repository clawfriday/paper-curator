import json
from collections import defaultdict


def main() -> None:
    with open("schemas/named_tree.json") as f:
        tree = json.load(f)

    name_to_nodes: dict[str, list[str]] = defaultdict(list)

    def walk(node: dict) -> None:
        name = (node.get("name") or "").strip()
        node_id = node.get("node_id") or "(missing node_id)"
        if name:
            name_to_nodes[name].append(node_id)
        for child in node.get("children", []) or []:
            walk(child)

    walk(tree)

    duplicates = {k: v for k, v in name_to_nodes.items() if len(v) > 1}
    if not duplicates:
        print("No duplicate node names found.")
        return

    print("Duplicate node names:")
    for name, ids in duplicates.items():
        print(f"- {name}:")
        for node_id in ids:
            print(f"  - {node_id}")


if __name__ == "__main__":
    main()
