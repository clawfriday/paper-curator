#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build JSON payload for summarize/embed endpoints.")
    parser.add_argument("--extract-json", required=True, help="GROBID extract JSON path.")
    parser.add_argument("--max-chars", type=int, default=12000, help="Max characters for payload text.")
    args = parser.parse_args()

    path = Path(args.extract_json)
    data = json.loads(path.read_text(encoding="utf-8"))
    abstract = data.get("abstract") or ""
    sections = data.get("sections") or []
    section_texts = [section.get("text") for section in sections if section.get("text")]
    full_text = "\n\n".join([abstract] + section_texts).strip()
    if not full_text:
        raise SystemExit("No text extracted from GROBID output.")

    payload = {"text": full_text[: args.max_chars]}
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
