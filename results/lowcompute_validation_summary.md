# Local Diagnostic Summary

This is a local mechanism-check record, not a benchmark result.

## Environment

- python: C:/Model/mix/.venv-lowcompute/Scripts/python.exe
- torch: 2.12.1+cpu
- numpy: 2.2.6
- tree_sitter_available: True
- docker_hello_world: passed
- GPU: not used

## Stage 0

- `py_compile`: passed
- Unit tests: 41/41 passed with PyTorch CPU after adding fail-fast Transformer
  baseline checks, incremental AST configuration checks, and opt-in CVD audit
  coverage plus validated-pipeline input range controls and a token-level CVD
  mask regression test for short next-token inputs
- Data pipeline preflight: passed on 3 tiny diagnostic samples; token ids, AST
  ids, AST block shapes, masks, and usable lengths were valid
- Definition-mask tests: passed; assignment, parameters, imports, attributes, destructuring, for-targets, and use-only negatives covered
- Offline AST preprocessing now defaults to complete-file parsing; prefix parsing is explicit for inference/degradation checks
- Prefix AST fallback_rate: 0.3846
- Prefix AST unknown_rate: 0.0000
- Prefix/full divergence: 0.4615
- Prefix degradation avg_ms: 4.296

## Architecture Cost Audit

This is a structural edge-count audit, not a runtime benchmark.

| Sequence length | Block size | Dense causal edges | CSMT counted edges | Ratio | Continuous B* |
|---:|---:|---:|---:|---:|---:|
| 1024 | 64 | 524800 | 33416 | 0.0637 | 12.70 |
| 2048 | 64 | 2098176 | 67088 | 0.0320 | 16.00 |
| 4096 | 64 | 8390656 | 135200 | 0.0161 | 20.16 |

The table supports the block-local plus block-graph communication story, while keeping block size as a measured variable rather than a fixed constant.

## Structural Probe Coverage

| Case | Tokens | Blocks | Definition tokens | Definition-use pairs | Cross-block pairs | Max block distance |
|---|---:|---:|---:|---:|---:|---:|
| `shadowing` | 34 | 5 | 6 | 3 | 2 | 3 |
| `long_range_import` | 30 | 4 | 6 | 3 | 3 | 3 |
| `guarded_attribute` | 34 | 5 | 7 | 5 | 3 | 2 |

## Direct Structural Dependency Preservation

This is a static dependency-preservation check for diagnostic or generated code
snippets, not a semantic correctness proof. The current recorded run uses the
reference diagnostic snippets because no model-generated candidates are bundled
with the release.

| Source | Dependencies | Preserved | Dependency preservation rate | Parse success rate |
|---|---:|---:|---:|---:|
| Reference snippets | 6 | 6 | 1.0000 | 1.0000 |

## CVD Mask Audit

Sampling-count synchronization is disabled in the default training path and
enabled explicitly for this audit.

| Scope | Eligible blocks | Sampled blocks | Valid blocks | Sample rate |
|---|---:|---:|---:|---:|
| `variable` | 72 | 16 | 112 | 0.2222 |
| `random` | 112 | 25 | 112 | 0.2232 |

## Stage 1 Tiny Ablation With Transformer Controls

| Variant | Params | Final loss | Eval loss | Definition-use loss | Cross-block use loss |
|---|---:|---:|---:|---:|---:|
| `transformer_baseline` | 4688 | 3.6034 | 3.6342 | 3.6630 | 3.6809 |
| `transformer_matched` | 6992 | 3.6751 | 3.5889 | 3.9101 | 3.7469 |
| `token_baseline` | 4576 | 3.7727 | 3.7131 | 3.7730 | 3.9152 |
| `boundary_only` | 4832 | 3.7944 | 3.6823 | 3.8444 | 3.9949 |
| `ast_only` | 4896 | 3.7194 | 3.6715 | 3.7286 | 3.8034 |
| `graph_only` | 6384 | 3.7367 | 3.6630 | 3.4462 | 3.6176 |
| `ast_graph` | 6960 | 3.9444 | 3.6480 | 3.4645 | 3.5019 |
| `random_dropout_control` | 6976 | 3.6432 | 3.5470 | 3.7569 | 3.8565 |
| `variable_cvd` | 6976 | 3.6431 | 3.5470 | 3.7567 | 3.8563 |
| `full_moe` | 8160 | 3.6942 | 3.6087 | 3.4301 | 3.2707 |

## Transformer Seed Sweep Summary

Seeds 1-5 use block size 8 and the same 12-step CPU diagnostic. Values are means with population standard deviations in parentheses.

| Variant | Params | Final loss | Eval loss | Cross-block use loss |
|---|---:|---:|---:|---:|
| `transformer_baseline` | 4688 | 3.7466 (0.0664) | 3.6541 (0.0439) | 3.7939 (0.1964) |
| `transformer_matched` | 6992 | 3.7029 (0.0673) | 3.6396 (0.0802) | 3.5655 (0.3241) |
| `ast_graph` | 6960 | 3.6608 (0.0836) | 3.6346 (0.0693) | 3.5089 (0.2441) |
| `variable_cvd` | 6976 | 3.7325 (0.0610) | 3.6748 (0.0448) | 3.8846 (0.2214) |
| `full_moe` | 8160 | 3.7526 (0.0994) | 3.6491 (0.0657) | 3.6296 (0.1806) |

## CVD Random-Control Sweep

Seeds 1-5 compare random block replacement with variable-targeted CVD under the same tiny diagnostic configuration.

| Variant | Params | Final loss | Eval loss | Cross-block use loss |
|---|---:|---:|---:|---:|
| `random_dropout_control` | 6976 | 3.7325 (0.0610) | 3.6748 (0.0448) | 3.8846 (0.2214) |
| `variable_cvd` | 6976 | 3.7325 (0.0610) | 3.6748 (0.0448) | 3.8846 (0.2214) |

Mean variable-minus-random final-loss delta: -2.8e-6. Mean variable-minus-random cross-block delta: 1.6e-6.

## Transformer Block-Size Smoke Sweep

| Block size | Variant | Final loss | Eval loss | Cross-block use loss |
|---:|---|---:|---:|---:|
| 4 | `transformer_baseline` | 3.6034 | 3.6342 | 3.5679 |
| 4 | `transformer_matched` | 3.6751 | 3.5889 | 3.8467 |
| 4 | `ast_graph` | 3.9444 | 3.6443 | 3.4398 |
| 4 | `variable_cvd` | 3.6362 | 3.5419 | 3.7726 |
| 4 | `full_moe` | 3.6869 | 3.5951 | 3.4101 |
| 8 | `transformer_baseline` | 3.6034 | 3.6342 | 3.6809 |
| 8 | `transformer_matched` | 3.6751 | 3.5889 | 3.7469 |
| 8 | `ast_graph` | 3.9444 | 3.6480 | 3.5019 |
| 8 | `variable_cvd` | 3.6431 | 3.5470 | 3.8563 |
| 8 | `full_moe` | 3.6942 | 3.6087 | 3.2707 |
| 16 | `transformer_baseline` | 3.6034 | 3.6342 | 3.4863 |
| 16 | `transformer_matched` | 3.6751 | 3.5889 | 3.3827 |
| 16 | `ast_graph` | 3.9548 | 3.6639 | 3.7887 |
| 16 | `variable_cvd` | 3.6405 | 3.5540 | 3.9170 |
| 16 | `full_moe` | 3.6991 | 3.6147 | 3.3882 |

## Block-Size Sensitivity at B=32,64,128

This long-case diagnostic follows the block-size sensitivity concern from the
paper. It is intentionally reported as boundary evidence: cross-block
dependencies are present for every tested block size, but this tiny 8-step CPU
run does not show AST-graph dependency preservation or an advantage over the
rough parameter-neighbor Transformer.

| B | Def-use pairs | Cross-block pairs | AST graph cross-block loss | Transformer matched cross-block loss | AST graph preservation | Transformer preservation |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 9 | 8 | 5.2285 | 4.8751 | 0.0000 | 0.0000 |
| 64 | 9 | 8 | 5.2408 | 4.8751 | 0.0000 | 0.0000 |
| 128 | 9 | 8 | 5.2433 | 4.8751 | 0.0000 | 0.0000 |

## Interpretation

- This confirms forward/backward, AST gate, prefix block graph, boundary mixing, CVD scopes, dense FFN, and MoE fallback can train on the local machine.
- Block sizes 4, 8, and 16 all completed the tiny diagnostic run.
- Seeds 1, 2, 3, 4, and 5 all completed the tiny diagnostic run for transformer_baseline, transformer_matched, ast_graph, variable_cvd, and full_moe.
- This does not show model superiority; the run is too small and stochastic.
- The pure Transformer baseline is now an explicit independent model, not a disabled CSMT configuration.
- A rough parameter-neighbor Transformer control has 6992 parameters, close to ast_graph at 6960 and variable_cvd at 6976 in the tiny setting.
- On the seed-7 block-8 run, CSMT graph variants improve cross-block use loss over token_baseline, but the pure and matched Transformers remain strong controls.
- Across seeds 1-5, ast_graph has the lowest mean cross-block use loss, but transformer_matched is close and has high variance; this is a comparison starting point, not evidence that CSMT-GNN beats Transformer.
- Random CVD and variable CVD remain numerically almost identical across the five-seed control, so CVD benefit is not established.
- The direct structural-dependency evaluator is now available for generated candidates; the bundled 1.0000 preservation result is only a reference-snippet sanity check.
- The B=32,64,128 sensitivity run is a useful negative/neutral result: the long diagnostic keeps cross-block dependencies at every tested block size, but this run does not show AST-graph preservation or a matched-Transformer win.
