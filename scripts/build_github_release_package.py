"""Build the public CSMT-GNN GitHub release package for the arXiv artifact."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List


ROOT = Path(__file__).resolve().parents[1]


FILES = [
    ".zenodo.json",
    "00_START_HERE.md",
    "README.md",
    "SUBMISSION_NOTES.md",
    "ENGINEERING_NOTES.md",
    "LICENSE",
    "CITATION.cff",
    "arxiv_metadata.json",
    "pyproject.toml",
    "requirements.txt",
    "ast_preprocessor.py",
    "inference_ast.py",
    "diagnostics.py",
    "csmt_gnn.py",
    "transformer_baseline.py",
    "train.py",
]

DIRS = [
    "paper",
    "scripts",
    "tests",
]

RESULT_FILES = [
    "results/submission_status.json",
    "results/lowcompute_validation_summary.md",
    "results/lowcompute_validation_summary.json",
    "results/data_pipeline_validation.json",
    "results/structural_probe_eval.json",
    "results/cvd_mask_audit.json",
    "results/architecture_cost_table.json",
    "results/diagnostic_poc_lowcompute.json",
    "results/diagnostic_poc_transformer.json",
    "results/diagnostic_poc_transformer_b4.json",
    "results/diagnostic_poc_transformer_b16.json",
    "results/diagnostic_poc_transformer_seed1.json",
    "results/diagnostic_poc_transformer_seed2.json",
    "results/diagnostic_poc_transformer_seed3.json",
    "results/diagnostic_poc_quick_arch_update.json",
    "results/prefix_ast_degradation_lowcompute.json",
    "results/block_sweep_b4.json",
    "results/block_sweep_b8.json",
    "results/block_sweep_b16.json",
    "results/seed_sweep_1.json",
    "results/seed_sweep_2.json",
    "results/seed_sweep_3.json",
]

EXCLUDE_SUFFIXES = {
    ".aux",
    ".blg",
    ".log",
    ".out",
    ".pyc",
    ".zip",
}

EXCLUDE_DIR_NAMES = {
    ".git",
    "__pycache__",
    "__MACOSX",
}

EXCLUDE_FILE_NAMES = {
    "Styles.zip",
}

EXCLUDE_NAME_MARKERS = {
    "conference",
    "template",
    "style",
}

RENAMED_RELEASE_FILES = [
    ("paper/main.pdf", "CSMT-GNN_arXiv_preprint_v0.1.2.pdf"),
]


def should_copy(path: Path) -> bool:
    if any(part in EXCLUDE_DIR_NAMES for part in path.parts):
        return False
    if path.name in EXCLUDE_FILE_NAMES:
        return False
    lowered_parts = [part.lower() for part in path.parts]
    if any(any(marker in part for marker in EXCLUDE_NAME_MARKERS) for part in lowered_parts):
        return False
    if path.name == "main.bbl":
        return True
    if path.suffix in EXCLUDE_SUFFIXES:
        return False
    return True


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path, package_root: Path) -> List[str]:
    copied: List[str] = []
    for path in src.rglob("*"):
        if not path.is_file() or not should_copy(path.relative_to(src)):
            continue
        rel = path.relative_to(src)
        copy_file(path, dst / rel)
        copied.append(str((dst / rel).relative_to(package_root)).replace("\\", "/"))
    return copied


def zip_dir(source_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in source_dir.rglob("*"):
            if path.is_file() and ".git" not in path.parts:
                zf.write(path, path.relative_to(source_dir))


def clear_output_dir(output_dir: Path) -> None:
    """Clear generated release files while preserving a Git checkout if present."""
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        return
    if not (output_dir / ".git").exists():
        shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)
        return
    for path in output_dir.iterdir():
        if path.name == ".git":
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def build(output_dir: Path, zip_path: Path) -> None:
    clear_output_dir(output_dir)

    copied: List[str] = []
    for rel in FILES:
        src = ROOT / rel
        if src.exists():
            copy_file(src, output_dir / rel)
            copied.append(rel)

    for rel in RESULT_FILES:
        src = ROOT / rel
        if src.exists():
            copy_file(src, output_dir / rel)
            copied.append(rel)

    for rel in DIRS:
        src = ROOT / rel
        if src.exists():
            copied.extend(copy_tree(src, output_dir / rel, output_dir))

    for src_rel, dst_rel in RENAMED_RELEASE_FILES:
        src = ROOT / src_rel
        if src.exists():
            copy_file(src, output_dir / dst_rel)
            copied.append(dst_rel)

    manifest = {
        "package": "csmt-gnn-github-release",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "public release package accompanying the arXiv preprint",
        "target": "arXiv public preprint",
        "files": sorted(copied),
    }
    (output_dir / "RELEASE_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    zip_dir(output_dir, zip_path)
    print(json.dumps({"release_dir": str(output_dir), "zip": str(zip_path), "files": len(copied)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the CSMT-GNN GitHub release package.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build" / "github_release")
    parser.add_argument("--zip-path", type=Path, default=ROOT / "build" / "csmt_gnn_github_release.zip")
    args = parser.parse_args()
    build(args.output_dir, args.zip_path)


if __name__ == "__main__":
    main()
