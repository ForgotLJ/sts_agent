# M7-C Persistent DAgger Protocol

## Status

M7-B is frozen as a negative formal result. Its held-out teacher-action score
improved from 83.58% to 91.43%, but its paired end-to-end mean-floor difference
against the M6 initialization was -1.8730 with a 95% interval of
[-2.6094, -1.1523]. M7-C therefore tests closed-loop data aggregation before
changing the network architecture.

M7-C must not modify, reinterpret, or reuse the following M7-B ranges for
model selection or evaluation:

| Range | Prior use | M7-C status |
| --- | --- | --- |
| 1400000-1400511 | M7 pilot gate | permanently locked |
| 1500000-1500511 | M7-B teacher validation | permanently locked |
| 1600000-1600511 | M7-B paired gate | permanently locked |
| 3000000-3002047 | M7 final blind test | permanently locked |

The complete historical and M7-C registry is defined in
`sts_env.training.m7c_protocol`. A run must reject a seed range unless it is
exactly registered for its declared role.

## Hypothesis

The M7-B failure is covariate shift: teacher-forced training evaluates states
visited by the teacher, while autonomous execution visits states induced by
earlier student errors. Persistent DAgger labels those student-induced states
with the heuristic policy and aggregates them across rounds.

## Invariants

- `TraceStep.action` is always the behavior action that advances the environment.
- Each trace header stores a teacher action label for every trace step.
- Replaying a trace follows behavior actions; cross-entropy uses teacher labels.
- Student-induced loops remain supervised DAgger states; M7-B's loop-erasure
  safeguard is intentionally disabled only for the M7-C data path.
- A missing, stale, or illegal teacher label is a hard failure.
- Corpus manifests include every trace hash, phase count, behavior-policy hash,
  teacher-policy identity, mixing probability, and seed range.
- Formal training retains exactly the frozen M7-B teacher corpus: seeds
  `400000-404095`, aggregate SHA-256
  `0dfdd54bccc66b6c16b2f4515fa160ecc46752f21dc1022d1032c57b026fdd14`.
  A substituted corpus, even with the same schema, is rejected at start and on
  resume.
- All checkpoints, corpus manifests, and replay caches are written atomically.

## Pre-registered Data Split

| Role | Range | Count |
| --- | ---: | ---: |
| DAgger round 0 collection | 2200000-2201023 | 1024 |
| DAgger round 1 collection | 2201024-2202047 | 1024 |
| DAgger round 2 collection | 2202048-2203071 | 1024 |
| Teacher-state anchor validation | 2210000-2210511 | 512 |
| On-policy validation, rounds 0/1/2 | 2211000/2212000/2213000 | 512 each |
| Promotion evaluation | 2220000-2220511 | 512 |
| Formal paired gate | 2221000-2221511 | 512 |

The three collection rounds use teacher mixing probabilities 0.50, 0.25, and
0.00. Combat stays delegated to the same heuristic used by M7-B, while all
non-combat behavior is selected by the current student unless a round's
pre-registered teacher mixing decision is taken.

The collector is resumable and only accepts the named pre-registered range:

```bash
PYTHONPATH=src .venv/bin/python scripts/collect-m7c-dagger-corpus.py \
  --checkpoint /scratch/sts_agent/experiments/m7b_formal/seed-17/best-evaluation-checkpoint.pt \
  --seed-range-name on_policy_round_0 \
  --round-index 0 \
  --teacher-mix-probability 0.0 \
  --run-seed 17 \
  --device cuda \
  --workers 1 \
  --output /scratch/sts_agent/experiments/m7c_diagnostic_seed17
```

It writes `collection-summary.json`, a hash-verified corpus `manifest.json`,
and `on-policy-diagnostic.json`. The diagnostic is descriptive and must not be
used as an end-to-end promotion result.

`m7c_smoke` and `m7c_smoke_validation` are separate eight-seed engineering
ranges. They may verify collection, replay, and checkpoint-resume wiring only;
they are never validation, selection, promotion, or formal-evaluation ranges.

## Execution Gates

1. Engineering smoke: collect a small isolated corpus, verify hashes, verify
   replay semantics, resume an interrupted train batch, and require all tests.
2. Diagnostic: measure student-teacher agreement by floor and phase on fresh
   student-induced states. This is descriptive and cannot select a checkpoint.
3. GRU DAgger control: use the M7-B architecture and pure teacher-action
   cross-entropy. Select only on the teacher anchor and the matching round's
   on-policy validation set.
4. Promotion: compare the selected GRU DAgger candidate with both M6 initial
   and the heuristic on the fresh promotion range. Safety counters must all be
   zero. The paired M7-C-minus-M6 final-floor interval must have a positive
   lower bound and Act 1 must not regress before the formal gate is opened.
5. Attention ablation: only after step 4 passes. It reuses the same DAgger
   corpus, training budget, and validation seeds. The only changed variable is
   the candidate-action attention architecture.
6. Formal multi-seed evaluation: only the promoted architecture trains seeds
   17, 29, and 43. The formal paired gate remains unused until checkpoints are
   frozen. The M7 final blind range remains untouched.

## Attention Scope

Attention is not a primary corrective action. The first ablation retains the
GRU as temporal context and adds permutation-invariant current-state tokens
for player, deck, relics, potions, map, and candidate actions. Candidate actions
query state tokens through cross-attention. Full-trajectory Transformers and
map graph encoders are deferred until the GRU DAgger control has an interpretable
result.
