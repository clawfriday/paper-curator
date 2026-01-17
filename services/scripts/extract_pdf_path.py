#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract pdf_path from arxiv download response JSON.")
    parser.add_argument("--download-json", required=True, help="Path to arxiv download response JSON.")
    args = parser.parse_args()

    path = Path(args.download_json)
    data = json.loads(path.read_text(encoding="utf-8"))
    pdf_path = data.get("pdf_path")
    if not pdf_path:
        raise SystemExit("pdf_path missing in download response")
    print(pdf_path)


if __name__ == "__main__":
    main()
