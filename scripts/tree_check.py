import json
from collections import Counter


def main() -> None:
    with open("schemas/named_tree.json") as f:
        tree = json.load(f)

    node_ids: list[str] = []
    node_names: list[str] = []
    missing_names: list[str] = []
    mixed_children: list[str] = []
    missing_node_id = 0
    missing_node_type = 0
    cycle_paths: list[list[str]] = []

    def walk(node: dict, path: list[str]) -> None:
        nonlocal missing_node_id, missing_node_type
        node_id = node.get("node_id")
        node_type = node.get("node_type")
        name = (node.get("name") or "").strip()

        if node_id in path:
            cycle_paths.append(path + [node_id])
            return

        if not node_id:
            missing_node_id += 1
        else:
            node_ids.append(node_id)

        if not node_type:
            missing_node_type += 1

        if not name:
            missing_names.append(node_id or "(missing id)")
        else:
            node_names.append(name)

        children = node.get("children") or []
        if node_type == "category" and children:
            types = {c.get("node_type") for c in children}
            if "paper" in types and "category" in types:
                mixed_children.append(node_id)

        for child in children:
            walk(child, path + ([node_id] if node_id else []))

    walk(tree, [])

    node_id_counts = Counter(node_ids)
    node_name_counts = Counter(node_names)
    node_id_dupes = {k: v for k, v in node_id_counts.items() if v > 1}
    node_name_dupes = {k: v for k, v in node_name_counts.items() if v > 1}

    lines: list[str] = []
    lines.append("Tree checks:")
    lines.append(f"- Total nodes: {len(node_ids)}")
    lines.append(f"- Missing node_id: {missing_node_id}")
    lines.append(f"- Missing node_type: {missing_node_type}")
    lines.append(f"- Missing node name: {len(missing_names)}")
    lines.append(f"- Duplicate node_id: {len(node_id_dupes)}")
    lines.append(f"- Duplicate node name: {len(node_name_dupes)}")
    lines.append(f"- Mixed children (paper+category): {len(mixed_children)}")
    lines.append(f"- Cycles detected: {len(cycle_paths)}")

    if node_name_dupes:
        lines.append("")
        lines.append("Sample duplicate node names:")
        for name, count in list(node_name_dupes.items())[:10]:
            lines.append(f"  {name}: {count}")

    if cycle_paths:
        lines.append("")
        lines.append("Sample cycle paths:")
        for path in cycle_paths[:3]:
            lines.append("  -> ".join(path))

    print("\n".join(lines))


if __name__ == "__main__":
    main()
