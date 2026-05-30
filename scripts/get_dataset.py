"""Download a real, openly-licensed document into data/ for the demo.

Default dataset
---------------
NIST AI Risk Management Framework 1.0 (NIST AI 100-1), January 2023.

- Source:  https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf
- DOI:     https://doi.org/10.6028/NIST.AI.100-1
- License: U.S. Government work. NIST publications are not subject to
           copyright protection in the United States (17 U.S.C. ยง 105) and
           are free to use, reproduce, and distribute. See:
           https://www.nist.gov/director/licensing

This document was picked because it (1) is unambiguously openly licensed,
(2) is large enough that retrieval differences are measurable (48 pages,
~100 chunks at default settings), and (3) is full of concrete, factual
claims that make for clean evaluation questions.

Usage
-----
    python scripts/get_dataset.py
    # or, inside Docker:
    docker compose run --rm rag python scripts/get_dataset.py
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path


DEFAULT_URL = "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf"
DEFAULT_FILENAME = "nist_ai_rmf_1.0.pdf"


def download(url: str, dest: Path) -> int:
    """Download ``url`` to ``dest``. Returns the file size in bytes."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "rag-pipeline-dataset-fetcher/1.0"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status} fetching {url}")
        data = resp.read()
    dest.write_bytes(data)
    return len(data)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Source URL to download (default: {DEFAULT_URL}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data") / DEFAULT_FILENAME,
        help=f"Destination path (default: data/{DEFAULT_FILENAME}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the destination already exists.",
    )
    args = parser.parse_args()

    if args.out.exists() and not args.force:
        size = args.out.stat().st_size
        print(f"Already present: {args.out} ({size:,} bytes). Use --force to re-download.")
        return 0

    print(f"Downloading {args.url} -> {args.out} ...")
    try:
        size = download(args.url, args.out)
    except Exception as e:
        print(f"ERROR: download failed: {e}", file=sys.stderr)
        print(
            "Tip: NIST sometimes reorganizes URLs. If this 404s, grab the PDF "
            "manually from https://doi.org/10.6028/NIST.AI.100-1 and save it as "
            f"{args.out}.",
            file=sys.stderr,
        )
        return 1

    print(f"OK: {size:,} bytes written.")
    print()
    print("Next:")
    print(f"  docker compose run --rm rag python -m src.cli clear --yes")
    print(f"  docker compose run --rm rag python -m src.cli ingest {args.out.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
