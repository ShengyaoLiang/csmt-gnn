# CSMT-GNN Research Workspace

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20840625.svg)](https://doi.org/10.5281/zenodo.20840625)

I started CSMT-GNN from a stubborn question: if code is full of explicit
structure, why should a code model have to rediscover all of that structure from
raw tokens at every layer?

This repository is the research workspace for the arXiv version of the
CSMT-GNN paper. It is not only a README and not only a paper. It is meant to be
a complete package: paper source, reference code, diagnostic scripts, tests,
submission notes, arXiv source packaging, and a public GitHub release builder.

## Scientific Position

CSMT-GNN studies a narrow claim: code language models may preserve long-range
program dependencies better when token modeling is paired with compact AST
features and prefix-masked block graph states.

I no longer claim Pearl-style causal intervention. The early drafts used the
phrase "do-intervention"; that was too strong. The current mechanism is
Contextual Variable Dropout (CVD), a structured regularizer that replaces value
messages for definition-bearing blocks during training. This is smaller, but it
is honest enough to test.

The current release separates mechanism evidence from performance claims.  It
contains:

- a corrected reference architecture;
- theoretical checks for prefix visibility, edge counts, gate stability, AST
  cost, boundary causality, prefix AST degradation metrics, CVD smoothing, and
  bounded perturbations;
- a structural edge-count audit for the block-local plus block-graph trade-off;
- small structural diagnostics for shadowing, long-range imports, and guarded
  attributes;
- structural probe coverage for definition-use and cross-block relations;
- a CVD mask audit that separates definition-targeted sampling from random
  valid-block sampling;
- a prefix/full AST degradation diagnostic for `fallback_rate`, `unknown_rate`,
  `prefix_full_divergence`, and parser latency;
- a tiny ablation-oriented diagnostic training script that can run when PyTorch is available;
- an independent token-only Transformer baseline and a rough parameter-neighbor
  Transformer control inside the tiny diagnostic script;
- a local diagnostic record under `results/lowcompute_validation_summary.md`;
- a staged falsification plan before any future 300M--700M parameter study.

## Development Position

This project also argues for a workflow and a stance. I care about code models
because code is becoming a production material for science, engineering, data
analysis, and automation. I am less interested in making AI look human than in
making it a reliable part of human productivity. If generated code is fluent but
structurally wrong, it does not merely fail as text; it damages a tool chain.

I used multiple LLMs during early prototype development because I wanted to see
whether an independent researcher could move from architecture thinking to a
runnable artifact without institutional-scale support. Different systems helped
with different parts of the work: scaffolding, implementation, criticism, and
documentation. That made research feel less locked behind infrastructure: one
person could explore preprocessors, kernels, training code, and documentation
much faster.

But the same workflow produced mistakes that serious science cannot keep:
overconfident causal language, repeated docs, brittle hard-coded kernels, random
offline AST embeddings, and code names that sounded more rigorous than the
implementation. My later work was audit and authorship: narrowing the claims,
rewriting the code, removing the pseudo-causal frame, and making the artifact
inspectable.

For the arXiv and public GitHub versions, I keep the fuller author story: this
is an independent-research artifact that used multi-LLM collaboration early and
then required human audit, deletion, renaming, and rewrite. The important lesson
is not that AI tools make research automatic. It is that more people can start
from a serious question, use AI to reach a testable artifact, and then take
responsibility for the result.

## Publication Plan

- Current target: arXiv public preprint.
- Current LaTeX style: plain `article` with `natbib`, suitable for arXiv source upload.
- arXiv metadata is stored in `arxiv_metadata.json` and is filled for the public
  preprint build.

See `SUBMISSION_NOTES.md` for the current submission boundary.

## Layout

```text
C:\Model\mix
├── paper/                    arXiv preprint paper source and compiled draft
├── scripts/                  Build and local diagnostic scripts
├── tests/                    Unit tests for model shape and leakage checks
├── results/                  Machine-readable diagnostic and package reports
├── build/                    Zip packages and rendered artifacts
├── ast_preprocessor.py       Offline AST id and definition-mask builder
├── inference_ast.py          Prefix AST feature builder for generation
├── diagnostics.py            Structural diagnostic data generator
├── csmt_gnn.py               Reference model implementation
├── transformer_baseline.py   Independent token-only Transformer diagnostic baseline
└── train.py                  Single-process and torchrun training entry point
```

## Quick Start

Preprocess a Python file:

```powershell
python ast_preprocessor.py --source-file example.py --output-dir ast_data `
  --tokenizer-name-or-path /path/to/tokenizer
```

Offline preprocessing defaults to complete-file parsing for cleaner training
artifacts. Prefix parsing is explicit and is used for generation or degradation
diagnostics, not as the default training path.

Build prefix AST features for incomplete generation:

```powershell
python inference_ast.py --source-file partial_generation.py `
  --vocab-path ast_data/ast_vocab.json --block-size 64 --max-tokens 2048 `
  --repeat 32
```

When `--vocab-path` is provided, the AST vocabulary is frozen and unseen node
types map to `<UNKNOWN>`. The command prints `fallback_rate`, `unknown_rate`,
and `avg_ms`, which are the quick checks I use for prefix-feature degradation
and latency.

Measure full-vs-prefix AST drift on a finished file:

```powershell
python scripts\prefix_ast_degradation.py --source-file example.py `
  --vocab-path ast_data/ast_vocab.json --block-size 64 --max-tokens 2048 `
  --repeat 1 --output results\prefix_ast_degradation.json
```

This reports `fallback_rate`, `unknown_rate`, `prefix_full_divergence`, and
`avg_ms`. It is a Stage-0 diagnostic for the inference concern, not a model
result.

Generate the structural edge-count audit:

```powershell
python scripts\architecture_cost_table.py --output results\architecture_cost_table.json
```

This counts causal dense attention edges against block-local token edges plus
causal block-graph edges. It is a mathematical audit of the communication
pattern, not a runtime benchmark.

Generate structural diagnostic arrays:

```powershell
python diagnostics.py --output-dir tmp\diagnostics --block-size 8
```

Measure whether the tiny diagnostics actually contain cross-block structure:

```powershell
python scripts\structural_probe_eval.py --output results\structural_probe_eval.json `
  --block-size 8 --max-tokens 64
```

Audit CVD sampling without claiming model quality:

```powershell
python scripts\cvd_mask_audit.py --output results\cvd_mask_audit.json `
  --steps 24 --hidden-size 16 --ast-dim 8 --block-size 8 --max-tokens 64
```

The model keeps CVD sampling statistics off by default to avoid per-step device
synchronization during training. The audit script enables the same counters
explicitly, so mechanism inspection and the training fast path remain separate.
For curated NumPy pipelines that have already been validated, `train.py` also
offers `--no-input-range-validation` to skip per-forward token/AST id range
scans. Shape, dtype, and sequence-length checks remain active.

Run the tiny diagnostic training sanity check when PyTorch is installed:

```powershell
python scripts\diagnostic_poc_train.py --steps 12 --output results\diagnostic_poc_transformer.json
```

The default diagnostic script trains:
`transformer_baseline`, `transformer_matched`, `token_baseline`,
`boundary_only`, `ast_only`, `graph_only`, `ast_graph`,
`random_dropout_control`, `variable_cvd`, and `full_moe`.

This repository now includes one local diagnostic run:

```powershell
type results\lowcompute_validation_summary.md
```

That run verifies that the reference paths train end to end on CPU, including
AST gate, block graph, boundary mixing, CVD, dense FFN, MoE fallback, and the
two token-only Transformer controls. It also records structural coverage and
CVD sampling audits. It is not a benchmark result and does not establish that
CSMT-GNN beats Transformer or that CVD improves accuracy.

Train on paired token and AST arrays:

```powershell
python train.py --data-path tokenized_data --ast-path ast_data `
  --num-layers 12 --hidden-size 768 --block-size 64 --max-tokens 2048 `
  --boundary-width 2 --micro-batch-size 4
```

Build the public release package:

```powershell
python scripts\build_github_release_package.py
```

Build the arXiv source package:

```powershell
python scripts\build_arxiv_package.py
```

## Current Status

The implementation passes Python syntax checks in this workspace. A local
`.venv-lowcompute` environment using Python 3.10.11, NumPy, tree-sitter, and
PyTorch CPU completed the unit tests and diagnostic runs recorded in
`results/lowcompute_validation_summary.md`.

The next serious step is not a leaderboard claim. It is a stricter
Transformer-vs-CSMT falsification run where depth, width, FFN size, training
tokens, optimizer, wall-clock budget, and dependency-specific metrics are
matched explicitly. Only after that should I consider a small 300M--700M
parameter study.
