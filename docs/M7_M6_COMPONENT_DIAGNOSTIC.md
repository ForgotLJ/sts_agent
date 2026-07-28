# M7 M6 Component Diagnostic

## Scope

This diagnostic re-evaluates the three frozen M6 EMA checkpoints with the previously omitted `learned-heuristic` composition: the recurrent network selects non-combat actions and the deterministic heuristic selects combat actions.

The already revealed M6 final seed range `2000000-2001023` is used only for post-hoc diagnosis. It is not part of the M7 selection or final seed protocol.

## Artifact Verification

- M6 release archive SHA-256: `80c1b752566e318e702cf7336465509b8c54c7101102edd89c122358539411c1`
- All 45 manifest entries were hash-verified locally.
- The three M6 checkpoint hashes match the release manifest.
- The supplemental diagnostic output is stored outside the repository at `D:\Project\STS\m7_component_diagnostics_local`.

## Learned-Heuristic Results

| M6 run seed | Mean floor | Act 1 clear | Act 2 clear | Act 3 clear | Wins | Errors |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 17.507 | 38.477% | 0.977% | 0.000% | 0 | 0 |
| 29 | 17.843 | 39.844% | 1.172% | 0.098% | 1 | 0 |
| 43 | 17.735 | 41.016% | 0.879% | 0.000% | 0 | 0 |

The M6 training-time validation range reported mean floors near 18.6 for the selected checkpoints. The larger 1024-episode post-hoc evaluation is lower and confirms that the small fixed M6 validation batch was not sufficient for robust model selection.

## Paired Component Effects

All differences below are candidate minus reference final floor. Intervals use M7's two-level bootstrap over training run and environment seed.

| Component | Mean delta | 95% interval | Interpretation |
|---|---:|---:|---|
| `heuristic-search - heuristic` | -2.818 | [-3.341, -2.314] | Belief search harms the heuristic full-run policy. |
| `learned-heuristic - heuristic` | -3.561 | [-4.103, -2.997] | The learned non-combat policy is materially below heuristic even without search. |
| `learned-search - learned-heuristic` | -1.421 | [-1.830, -1.017] | Search further harms the learned policy. |
| `learned-search - heuristic-search` | -2.164 | [-2.748, -1.589] | With the same search module, learned non-combat decisions remain weaker. |

The aggregate M6 final report previously counted the deterministic heuristic result three times. M7 reporting marks those records as duplicate episode data and retains the number of unique environment seeds separately.

## Decision

1. The M7 control and balanced pilots use `learned-heuristic` selection only.
2. `learned-search` is not an M7 primary method until the full-distribution combat benchmark passes its frozen noninferiority gate.
3. The first training ablation targets non-combat supervision coverage rather than larger PPO models or more updates of the M6 objective.
4. M7 final seeds remain unused; this diagnostic does not modify M6 artifacts or M7 model-selection state.
