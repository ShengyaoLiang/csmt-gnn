# CSMT-GNN: Start Here

This folder is now organized around the arXiv-first release:

1. The paper.
2. The arXiv package.
3. The public code/release package.

## Paper

- Main PDF: `paper/main.pdf`
- Main LaTeX source: `paper/main.tex`
- Bibliography: `paper/references.bib`
- Bibliography style: standard `plainnat`.

## Packages

- arXiv source folder: `build/arxiv_source/`
- arXiv source zip: `build/csmt_gnn_arxiv_source.zip`
- Public GitHub release folder: `build/github_release/`
- Public GitHub release zip: `build/csmt_gnn_github_release.zip`
- Archived old drafts: `build/old_thinking_archive.zip`
- Latest local diagnostic summary: `results/lowcompute_validation_summary.md`
- Architecture cost audit: `results/architecture_cost_table.json`
- Structural probe result: `results/structural_probe_eval.json`
- Direct dependency-preservation result: `results/structural_hallucination_eval.json`
- Block-size sensitivity result: `results/block_size_sensitivity_32_64_128.md`
- CVD mask audit: `results/cvd_mask_audit.json`
- Data pipeline preflight: `results/data_pipeline_validation.json`
- Transformer comparison diagnostic: `results/diagnostic_poc_transformer.json`

## Important

- `arxiv_metadata.json` contains the public author metadata for Shengyao Liang.
- The only current publication target in this workspace is arXiv.
- The old conference template files and temporary render images were removed
  from the working tree to keep this folder readable.

## Rebuild

```powershell
python scripts\build_arxiv_package.py
python scripts\build_github_release_package.py
```

Run the Stage-0 AST degradation diagnostic:

```powershell
python scripts\prefix_ast_degradation.py --source-file example.py `
  --vocab-path ast_data\ast_vocab.json --output results\prefix_ast_degradation.json
```

Run the current local mechanism checks:

```powershell
python scripts\architecture_cost_table.py --output results\architecture_cost_table.json
python scripts\structural_probe_eval.py --output results\structural_probe_eval.json `
  --block-size 8 --max-tokens 64
python scripts\structural_hallucination_eval.py `
  --output results\structural_hallucination_eval.json
python scripts\cvd_mask_audit.py --output results\cvd_mask_audit.json `
  --steps 24 --hidden-size 16 --ast-dim 8 --block-size 8 --max-tokens 64
python scripts\validate_data_pipeline.py --data-path tmp\cvd_mask_audit\tokens `
  --ast-path tmp\cvd_mask_audit\ast --vocab-size 128 --block-size 8 `
  --max-tokens 64 --output results\data_pipeline_validation.json
python scripts\diagnostic_poc_train.py --output results\diagnostic_poc_transformer.json `
  --steps 12 --hidden-size 16 --ast-dim 8 --block-size 8 --max-tokens 64
python scripts\block_size_sensitivity.py --block-sizes 32,64,128 `
  --steps 8 --hidden-size 16 --ast-dim 8 --max-tokens 220 `
  --case-set long --seed 7
```

Compile the paper with Docker:

```powershell
docker run --rm -v "C:\Model\mix\paper:/work" -w /work texlive/texlive:latest `
  sh -lc "pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex"
```
