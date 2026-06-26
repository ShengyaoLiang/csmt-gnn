# CSMT-GNN Engineering Notes

These notes record the engineering corrections made after archiving the original
four drafts into `build/old_thinking_archive.zip`. I keep the history because it
is part of the paper: multi-LLM assisted prototyping helped me move quickly, but
the final artifact only became defensible after audit, renaming, deletion and
rewrite.

## Target Package

- Public preprint target: arXiv.
- Current style: plain `article` with `natbib`.
- arXiv package: `build/arxiv_source/`, built by `scripts/build_arxiv_package.py`.
- Public release path: `build/github_release/`, built by
  `scripts/build_github_release_package.py`.

## Main Corrections

- Removed the unsupported Pearl/do-intervention framing.
- Renamed the mechanism to Contextual Variable Dropout (CVD), a structured
  regularizer over definition-bearing block value messages.
- AST embeddings are now trainable model parameters; the preprocessor writes
  integer ids only.
- AST artifacts can align to model tokens through HuggingFace fast-tokenizer
  offset mappings.
- The preprocessor writes both block-level and token-level variable-definition
  masks.
- Offline preprocessing now defaults to complete-file parsing for training
  artifacts; prefix parsing is explicit for inference and degradation checks.
- Token-level definition masks prefer Python binding spans when the source is
  parseable, covering function/class names, parameters, imports, assignments,
  attributes, destructuring, loop targets, and use-only negatives.
- Incremental inference uses prefix AST extraction rather than assuming a
  complete parse is available.
- Incremental AST configuration now validates `block_size` and `max_tokens`,
  and the command-line path no longer relies on Python `assert` statements.
- Prefix inference freezes a saved AST vocabulary and reports `unknown_rate`, so
  unseen node types do not create embedding ids outside the training vocabulary.
- The paper now defines `fallback_rate`, `unknown_rate`, and prefix/full AST
  divergence as the measurable degradation boundary for inference-time AST use.
- `scripts/prefix_ast_degradation.py` measures that boundary directly on a
  finished source file by comparing complete-file AST ids with per-token prefix
  AST ids.
- Python token positions are converted to UTF-8 byte spans before tree-sitter
  lookup.
- `--per-token-prefix` fails explicitly if tree-sitter is unavailable.
- CVD samples with the configured Bernoulli probability instead of forcing one
  masked block per eligible sample.
- CVD sampling counters are opt-in through `cvd_audit`; default training avoids
  per-step CPU synchronization for human-readable CVD counts.
- The reference model uses dynamic block shapes and no hard-coded `B=64`
  Triton softmax path.
- The model and trainer support true `[batch, tokens]` micro-batches.
- The model now accepts true sequence lengths, so padding tokens do not become
  real block content.
- Model inputs now fail early on invalid token ids, AST ids, mask dtypes, and
  out-of-range lengths instead of silently clamping or truncating metadata.
- `validate_input_ranges` keeps token/AST id range scans enabled by default, but
  `train.py --no-input-range-validation` can skip those large-tensor scans after
  the data pipeline has been checked; dtype, shape, and length validation remain
  active.
- `scripts/validate_data_pipeline.py` performs that one-time NumPy preflight
  over token ids, AST ids, AST block shapes, masks, and usable sequence lengths.
- Shared two-dimensional AST ids and definition masks can be broadcast across a
  batch, matching the documented single-example feature format.
- AST pooling handles all-padding blocks without producing NaNs.
- Boundary-aware pooling is strictly causal: the first `boundary_width` tokens
  of a current block can read the previous block tail, but previous blocks
  cannot read future block heads.
- Architecture components are now ablation switches:
  `use_ast_gate`, `use_block_graph`, `use_cvd`, `use_moe`, and `use_boundary`.
- The no-MoE ablation uses a dense SwiGLU-style FFN rather than removing the
  feed-forward path.
- CVD supports `cvd_scope=variable` and `cvd_scope=random`, so variable-targeted
  CVD can be compared with ordinary random block value replacement.
- Variable-targeted CVD intersects definition masks with the valid-block mask,
  so padded blocks cannot be sampled as definition-bearing messages.
- Prefix block graph attention now combines the causal block mask with a
  valid-block key mask and zeroes padded block outputs, so mixed-length batches
  do not let padding states participate in graph communication.
- Token-level CVD masks are interpreted against the current forward pass block
  count, so short sequences and next-token inputs do not get mistaken for
  block-level masks just because they are shorter than the configured maximum.
- The trainer computes next-token loss only over real, non-padding positions.
- The trainer trims AST ids and definition masks to the next-token model-input
  prefix before calling the model, rather than passing full-sample side inputs
  and relying on model-internal truncation.
- MoE fallback sorts top-k assignments into contiguous expert batches, skips
  absent experts, and exposes load-balancing/z-loss terms.
- `diagnostics.py` now writes a shared token vocabulary for tiny diagnostic
  training instead of reusing incompatible token ids per sample.
- `scripts/structural_probe_eval.py` measures definition-use and cross-block
  coverage in the tiny diagnostics before model claims are made.
- `scripts/architecture_cost_table.py` records dense causal attention edge
  counts against block-local plus block-graph edge counts for several sequence
  lengths and block sizes.
- `scripts/cvd_mask_audit.py` records CVD eligible and sampled blocks for
  variable-targeted and random-block scopes.
- `scripts/diagnostic_poc_train.py` provides a local falsification sanity
  check with independent Transformer controls, token, AST, graph,
  random-dropout, variable-CVD, and MoE variants when PyTorch is available.
- The diagnostic trainer now applies the same next-token prefix trimming to AST
  ids and definition masks that `train.py` uses, and records a
  `prefix_feature_audit` block in the JSON output.
- `transformer_baseline.py` is a separate token-only causal Transformer, so the
  Transformer control is not just a CSMT configuration with features disabled.
- The diagnostic script includes `transformer_matched`, a rough
  parameter-neighbor Transformer control for the tiny CSMT variants.
- The Transformer controls now share the same fail-fast input contract as the
  CSMT model for token ids, integer lengths, and out-of-range metadata.

## Minimal Checks

```powershell
python -m py_compile ast_preprocessor.py csmt_gnn.py transformer_baseline.py train.py inference_ast.py diagnostics.py scripts\architecture_cost_table.py scripts\prefix_ast_degradation.py scripts\structural_probe_eval.py scripts\cvd_mask_audit.py scripts\validate_data_pipeline.py scripts\diagnostic_poc_train.py scripts\build_github_release_package.py scripts\build_arxiv_package.py
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
python scripts\validate_data_pipeline.py --data-path tmp\diagnostics\tokens --ast-path tmp\diagnostics\ast --vocab-size 128 --block-size 8 --max-tokens 64
python -m unittest discover -s tests -v
```

When PyTorch is available:

```powershell
python scripts\cvd_mask_audit.py --steps 24 --hidden-size 16 --ast-dim 8 --block-size 8 --max-tokens 64
python scripts\diagnostic_poc_train.py --steps 12 --output results\diagnostic_poc_transformer.json
```

## Remaining Risks

- No full model training result is claimed.
- Use `.venv-lowcompute` or another environment with NumPy, tree-sitter, and
  PyTorch installed for diagnostics and training tests.
- Prefix AST inference has an implementation path, but latency and quality must
  be measured with `avg_ms`, `fallback_rate`, and `unknown_rate`.
- AST value must be proven empirically against its preprocessing and runtime
  cost.
- The paper includes a staged falsification plan; Stage 0 and Stage 1 should
  fail fast before any larger training run is attempted.
- The local diagnostic run is a smoke test only. It confirms that paths train
  and masks are observable; it does not establish CVD or AST effectiveness.
- The local Transformer comparison is also a smoke test only. It shows the
  baseline exists and that `transformer_matched` is a strong control; it does
  not establish that CSMT-GNN beats Transformer.
