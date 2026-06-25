"""
Build an arXiv source package for CSMT-GNN.

The builder verifies arxiv_metadata.json and marks the manifest upload-ready
only when public author metadata is complete.
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]


SOURCE_FILES = [
    "main.tex",
    "main.bbl",
    "references.bib",
]

def metadata_complete(metadata: Dict) -> bool:
    authors = metadata.get("authors", [])
    if not authors:
        return False
    for author in authors:
        if any(not str(author.get(key, "")).strip() for key in ("name", "email", "affiliation")):
            return False
    return True


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def zip_dir(source_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in source_dir.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(source_dir))


def build(output_dir: Path, zip_path: Path) -> None:
    metadata_path = ROOT / "arxiv_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    complete = metadata_complete(metadata)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    copied: List[str] = []
    for rel in SOURCE_FILES:
        src = ROOT / "paper" / rel
        if not src.exists():
            continue
        copy_file(src, output_dir / rel)
        copied.append(rel)

    copy_file(metadata_path, output_dir / "arxiv_metadata.json")
    copied.append("arxiv_metadata.json")

    upload_note = {
        "package": "csmt-gnn-arxiv-source",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "upload_ready": complete,
        "blocker": None if complete else "Fill real author metadata in arxiv_metadata.json before uploading.",
        "primary_category": metadata.get("primary_category"),
        "cross_list": metadata.get("cross_list", []),
        "files": sorted(copied),
    }
    (output_dir / "ARXIV_PACKAGE_MANIFEST.json").write_text(json.dumps(upload_note, indent=2), encoding="utf-8")
    zip_dir(output_dir, zip_path)
    print(json.dumps({"arxiv_dir": str(output_dir), "zip": str(zip_path), "upload_ready": complete}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build arXiv source package.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build" / "arxiv_source")
    parser.add_argument("--zip-path", type=Path, default=ROOT / "build" / "csmt_gnn_arxiv_source.zip")
    args = parser.parse_args()
    build(args.output_dir, args.zip_path)


if __name__ == "__main__":
    main()
