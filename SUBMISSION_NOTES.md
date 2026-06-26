# Submission Notes

## Current Target

- Active target: arXiv public preprint.
- Suggested arXiv primary category: `cs.LG`.
- Possible cross-lists: `cs.CL`, `cs.SE`.

## Public Author Metadata

- Author: Shengyao Liang.
- E-mail: pikeshuaiwe@gmail.com.
- Affiliation: Independent Researcher.
- ORCID: https://orcid.org/0009-0002-3713-8700.

The public metadata is stored in `arxiv_metadata.json` and `CITATION.cff`.

## Scientific Boundary

- I do not claim Pearl-style do-intervention, SCM identification, or causal
  adjustment.
- The mechanism is Contextual Variable Dropout (CVD): structured value-message
  masking over definition-bearing blocks.
- I do not claim frontier-scale or best-known results.
- I claim a corrected architecture, ablation-ready reference implementation,
  theoretical checks, a local diagnostic path, and an auditable multi-LLM
  assisted development workflow.
- The author narrative is intentionally public: I frame code as a production
  material, not as a stage for making AI imitate human consciousness, and I
  describe multi-LLM development as an aid for independent research that still
  requires human audit and responsibility.
- The local diagnostic path includes prefix/full AST degradation measurement before
  any model training claim.
- The current local evidence also includes structural probe coverage and CVD
  mask audits. These are mechanism checks, not performance evidence.
- The current local evidence includes a pure token-only Transformer baseline and
  a rough parameter-neighbor Transformer control. These are comparison smoke
  tests, not evidence that CSMT-GNN beats Transformer.

## arXiv Package

- Paper source: `paper/main.tex`.
- Compiled draft: `paper/main.pdf`.
- Source builder: `python scripts\build_arxiv_package.py`.
- Output directory: `build/arxiv_source/`.
- Output zip: `build/csmt_gnn_arxiv_source.zip`.
- Manifest: `build/arxiv_source/ARXIV_PACKAGE_MANIFEST.json`.

The arXiv package includes `main.tex`, `main.bbl`, `references.bib`, and
`arxiv_metadata.json`.

## Public Code Package

- Public repository: https://github.com/ShengyaoLiang/csmt-gnn.
- Archived releases are indexed under the stable all-version Zenodo concept DOI:
  https://doi.org/10.5281/zenodo.20840624.
- Builder: `python scripts\build_github_release_package.py`.
- Output directory: `build/github_release/`.
- Output zip: `build/csmt_gnn_github_release.zip`.

This package is for the public arXiv/GitHub story. It keeps the author identity
and the multi-LLM development narrative. I release the source package, paper
source, tests, small diagnostic artifacts, and packaging scripts so that the
evidence chain can be inspected without relying on private training data,
hidden scripts, or unpublished checkpoints.

## Current Commands

```powershell
python -m py_compile ast_preprocessor.py csmt_gnn.py transformer_baseline.py train.py inference_ast.py diagnostics.py scripts\architecture_cost_table.py scripts\prefix_ast_degradation.py scripts\structural_probe_eval.py scripts\cvd_mask_audit.py scripts\diagnostic_poc_train.py scripts\build_github_release_package.py scripts\build_arxiv_package.py
python scripts\build_arxiv_package.py
python scripts\build_github_release_package.py
```

When NumPy/tree-sitter dependencies are available:

```powershell
python diagnostics.py --output-dir tmp\diagnostics --block-size 8
Set-Content -LiteralPath tmp\diagnostics\prefix_probe.py -Value "def f(x):`n    y = x + 1`n    return y`n"
python inference_ast.py --source-file tmp\diagnostics\prefix_probe.py --block-size 8 --max-tokens 64 --repeat 32
python scripts\prefix_ast_degradation.py --source-file tmp\diagnostics\prefix_probe.py --block-size 8 --max-tokens 64 --repeat 1
python scripts\architecture_cost_table.py --output results\architecture_cost_table.json
python scripts\structural_probe_eval.py --output results\structural_probe_eval.json --block-size 8 --max-tokens 64
python -m unittest discover -s tests -v
```

When PyTorch is available:

```powershell
python scripts\cvd_mask_audit.py --steps 24 --hidden-size 16 --ast-dim 8 --block-size 8 --max-tokens 64
python scripts\diagnostic_poc_train.py --steps 12 --output results\diagnostic_poc_transformer.json
```

The diagnostic JSON should include `prefix_feature_audit.all_prefix_aligned =
1.0` with an empty `violations` list. This checks the diagnostic path itself:
CSMT variants receive AST ids and definition masks clipped to the same prefix as
the next-token input.
