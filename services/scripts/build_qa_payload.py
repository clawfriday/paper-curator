#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build JSON payload for QA endpoint.")
    parser.add_argument("--context-file", required=True, help="Path to context text file.")
    parser.add_argument("--question", required=True, help="Question to ask.")
    args = parser.parse_args()

    context_text = Path(args.context_file).read_text(encoding="utf-8")
    payload = {"context": context_text, "question": args.question}
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
