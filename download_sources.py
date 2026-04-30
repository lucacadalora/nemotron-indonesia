#!/usr/bin/env python3
"""Download public data/model sources for Nemotron-Indonesia.

The script is intentionally conservative: it only downloads the source(s) you
request. Some sources are very large. Start with --dry-run, then pull one source
at a time on the training box.

Examples:
  python download_sources.py --list
  python download_sources.py --sources core --dry-run
  python download_sources.py --sources indo4b_hf indonlu indobert
  HF_TOKEN=hf_xxx python download_sources.py --sources culturax_id
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List

try:
    from huggingface_hub import snapshot_download
except Exception:  # pragma: no cover
    snapshot_download = None


SOURCES: Dict[str, dict] = {
    "indo4b_hf": {
        "kind": "hf_dataset",
        "repo_id": "taufiqdp/Indo4B-hf",
        "repo_type": "dataset",
        "allow_patterns": ["data/*.parquet", "README.md"],
        "target": "data/raw/indo4b-hf",
        "url": "https://huggingface.co/datasets/taufiqdp/Indo4B-hf",
        "role": "Core Indonesian continued-pretraining corpus; HF parquet mirror of Indo4B.",
        "access": "public",
    },
    "sea_pile_id": {
        "kind": "hf_dataset",
        "repo_id": "aisingapore/SEA-PILE-v1",
        "repo_type": "dataset",
        "allow_patterns": ["sea-pile-mc4/id/*.jsonl.gz", "README.md", "LICENSE"],
        "target": "data/raw/sea-pile-id",
        "url": "https://huggingface.co/datasets/aisingapore/SEA-PILE-v1",
        "role": "Curated Southeast Asian web corpus; Indonesian mC4 subset.",
        "access": "public",
    },
    "mc4_id": {
        "kind": "hf_dataset",
        "repo_id": "allenai/c4",
        "repo_type": "dataset",
        "allow_patterns": ["multilingual/c4-id*.json.gz", "README.md"],
        "target": "data/raw/mc4-id",
        "url": "https://huggingface.co/datasets/allenai/c4",
        "role": "Large Indonesian mC4 web corpus fallback/replacement when OSCAR is unavailable.",
        "access": "public",
    },
    "cc100_id": {
        "kind": "direct",
        "urls": ["https://data.statmt.org/cc-100/id.txt.xz"],
        "target": "data/raw/cc100/id.txt.xz",
        "url": "https://data.statmt.org/cc-100/id.txt.xz",
        "role": "CC100 Indonesian web corpus from CC-Net/XLM-R pipeline.",
        "access": "public",
    },
    "cc100_local_languages": {
        "kind": "direct_many",
        "urls": [
            "https://data.statmt.org/cc-100/jv.txt.xz",
            "https://data.statmt.org/cc-100/su.txt.xz",
        ],
        "target": "data/raw/cc100-local-languages",
        "url": "https://data.statmt.org/cc-100/",
        "role": "Javanese and Sundanese CC100 subsets for local-language coverage.",
        "access": "public",
    },
    "wikipedia_id": {
        "kind": "hf_dataset",
        "repo_id": "wikimedia/wikipedia",
        "repo_type": "dataset",
        "allow_patterns": ["20231101.id/*.parquet", "README.md"],
        "target": "data/raw/wikipedia-id",
        "url": "https://huggingface.co/datasets/wikimedia/wikipedia",
        "role": "Structured Indonesian encyclopedia corpus.",
        "access": "public",
    },
    "indonlu": {
        "kind": "hf_dataset",
        "repo_id": "indonlp/indonlu",
        "repo_type": "dataset",
        "allow_patterns": ["*"],
        "target": "data/eval/indonlu",
        "url": "https://huggingface.co/datasets/indonlp/indonlu",
        "role": "Indonesian NLU benchmark/evaluation suite; also small supervised seed.",
        "access": "public",
    },
    "nusax_senti": {
        "kind": "hf_dataset",
        "repo_id": "indonlp/NusaX-senti",
        "repo_type": "dataset",
        "allow_patterns": ["*"],
        "target": "data/eval/nusax-senti",
        "url": "https://huggingface.co/datasets/indonlp/NusaX-senti",
        "role": "Local-language sentiment benchmark for Indonesian regional languages.",
        "access": "public",
    },
    "indobert": {
        "kind": "hf_model",
        "repo_id": "indobenchmark/indobert-base-p1",
        "repo_type": "model",
        "allow_patterns": ["*"],
        "target": "models/baselines/indobert-base-p1",
        "url": "https://huggingface.co/indobenchmark/indobert-base-p1",
        "role": "IndoBERT baseline model for benchmark comparison and data-quality helpers.",
        "access": "public",
    },
    "culturax_id": {
        "kind": "hf_dataset",
        "repo_id": "uonlp/CulturaX",
        "repo_type": "dataset",
        "allow_patterns": ["id/*.parquet", "README.md"],
        "target": "data/raw/culturax-id",
        "url": "https://huggingface.co/datasets/uonlp/CulturaX",
        "role": "Large deduplicated Indonesian corpus from mC4/OSCAR-derived sources; optional.",
        "access": "hf-auto-gated: accept terms on Hugging Face, then run with HF_TOKEN.",
    },
}

CORE_SOURCES = [
    "indo4b_hf",
    "sea_pile_id",
    "mc4_id",
    "cc100_id",
    "wikipedia_id",
    "indonlu",
    "nusax_senti",
    "indobert",
]


def expand_sources(names: List[str]) -> List[str]:
    expanded: List[str] = []
    for name in names:
        if name == "core":
            expanded.extend(CORE_SOURCES)
        elif name == "all":
            expanded.extend(SOURCES.keys())
        else:
            expanded.append(name)
    seen = set()
    result = []
    for name in expanded:
        if name not in SOURCES:
            raise SystemExit(f"Unknown source: {name}. Run --list.")
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def print_source(name: str, src: dict) -> None:
    print(f"\n{name}")
    print(f"  role:   {src['role']}")
    print(f"  access: {src['access']}")
    print(f"  url:    {src['url']}")
    print(f"  target: {src['target']}")
    if src["kind"].startswith("hf_"):
        patterns = ", ".join(src.get("allow_patterns", []))
        print(f"  hf:     {src['repo_id']} ({src['repo_type']}), include: {patterns}")
    else:
        for url in src.get("urls", []):
            print(f"  file:   {url}")


def verify_url(url: str, access: str) -> str:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "nemotron-indonesia-link-check"})
    try:
        with urllib.request.urlopen(req, timeout=25) as res:
            return f"OK {res.status}"
    except urllib.error.HTTPError as e:
        if "gated" in access and e.code in (401, 403):
            return f"REQUIRES_AUTH {e.code}"
        return f"HTTP_ERROR {e.code}"
    except Exception as e:
        return f"ERROR {type(e).__name__}: {e}"


def download_hf(name: str, src: dict, root: Path, dry_run: bool) -> None:
    target = root / src["target"]
    print_source(name, src)
    if dry_run:
        patterns = " ".join(f"--include '{p}'" for p in src.get("allow_patterns", []))
        print(
            "  command: huggingface-cli download "
            f"{src['repo_id']} --repo-type {src['repo_type']} {patterns} --local-dir {target}"
        )
        return
    if snapshot_download is None:
        raise SystemExit("huggingface_hub is missing. Install with: pip install huggingface-hub")
    target.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")
    snapshot_download(
        repo_id=src["repo_id"],
        repo_type=src["repo_type"],
        allow_patterns=src.get("allow_patterns"),
        local_dir=str(target),
        token=token,
    )


def download_direct(name: str, src: dict, root: Path, dry_run: bool) -> None:
    print_source(name, src)
    urls = src.get("urls", [src["url"]])
    if dry_run:
        for url in urls:
            print(f"  command: curl -L -C - --fail -O {url}")
        return
    target = root / src["target"]
    if src["kind"] == "direct":
        target.parent.mkdir(parents=True, exist_ok=True)
        out_paths = [target]
    else:
        target.mkdir(parents=True, exist_ok=True)
        out_paths = [target / Path(url).name for url in urls]
    for url, out_path in zip(urls, out_paths):
        print(f"Downloading {url} -> {out_path}")
        subprocess.run(["curl", "-L", "-C", "-", "--fail", "--output", str(out_path), url], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Nemotron-Indonesia data/model sources")
    parser.add_argument("--list", action="store_true", help="List sources and exit")
    parser.add_argument("--sources", nargs="+", default=[], help="Source names, or core/all")
    parser.add_argument("--root", default=".", help="Repository/data root")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without downloading")
    parser.add_argument("--verify-links", action="store_true", help="HEAD-check source landing/direct URLs")
    args = parser.parse_args()

    if args.list:
        for name, src in SOURCES.items():
            print_source(name, src)
        return 0

    if not args.sources:
        parser.error("provide --sources, or use --list")

    names = expand_sources(args.sources)
    root = Path(args.root).resolve()

    if args.verify_links:
        for name in names:
            src = SOURCES[name]
            urls = src.get("urls", [src["url"]])
            for url in urls:
                print(f"{name}: {verify_url(url, src['access'])} {url}")
        if args.dry_run:
            return 0

    for name in names:
        src = SOURCES[name]
        if src["kind"].startswith("hf_"):
            download_hf(name, src, root, args.dry_run)
        else:
            download_direct(name, src, root, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
