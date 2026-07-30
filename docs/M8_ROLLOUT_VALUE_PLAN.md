# M8 Rollout-Value Policy Improvement Draft

## Status

M7-C is frozen as a negative promotion result. Persistent DAgger improved the
seed-17 GRU candidate by `0.767578125` mean final floors over the M6
initialization, but the paired 95% interval `[-0.044921875, 1.568359375]` did
not clear zero. The candidate remained `3.490234375` mean floors below the
heuristic and failed both heuristic non-inferiority requirements. Safety was
clear and all promotion evaluations completed without runtime errors.

M8 is not authorized to train yet. This document defines the decision process
that follows the read-only M7-C diagnostic. M7-C training, validation,
promotion, formal-gate, and historical blind ranges remain locked and cannot
be used for M8 selection.

## Research Question

M7-C optimized deterministic teacher-action cross-entropy with no value loss.
That objective treats every labeled action mismatch as equally important and
does not distinguish a harmless presentation choice from an irreversible map,
shop, card-reward, or event decision that changes the rest of the run.

M8 asks whether explicit long-horizon candidate outcomes can improve policy
selection beyond action imitation while retaining the audited structured
environment and dynamic legal-action interface.

## Diagnostic Decision Rule

The frozen M7-C diagnostic is descriptive and cannot select a checkpoint. It
chooses which engineering ablations are justified before any M8 seed ranges
are registered.

1. Run `scripts/diagnose-m7c.py` on all six M7-C corpora and three selected
   validation records with trace-hash verification enabled.
2. Use only round-to-round changes, phase/floor agreement, confidence on wrong
   actions, and observed final-floor associations. Do not call these values
   causal regret.
3. The rollout-value control is always the primary M8 candidate because the
   missing long-horizon objective is known from the frozen M7-C configuration.
4. Add a candidate-action attention ablation only if the round-2 disagreement
   rate is concentrated in one or more high-cardinality phases (`card_reward`,
   `shop`, or `map`) by at least five percentage points relative to overall
   student agreement, or if wrong-action margins in those phases remain larger
   than correct-action margins.
5. If disagreement rises across rounds without phase concentration, diagnose
   persistent aggregation, weighting, and optimization before changing the
   architecture.

The five-percentage-point rule is an engineering trigger, not a claim of
statistical significance. It controls whether attention is worth implementing;
it does not determine promotion.

## M8-A: Rollout-Value GRU Control

M8-A retains the M7-C GRU state and candidate encoders. It adds value-sensitive
supervision without changing the observation contract.

### Counterfactual corpus

- Replay frozen M7-C training traces to reconstruct eligible non-combat states.
- Stratify states by phase, act, floor, legal-action count, and whether the
  M7-C student agreed with the teacher.
- Clone the environment at each selected state.
- Advance each legal action once, then roll out the frozen heuristic policy to
  termination or a fixed audited horizon.
- Use common rollout rules and the same horizon for every action at a state.
- Record the visible observation digest, candidate action, suffix outcome,
  clone/runtime identity, source-trace hash, and rollout trace hash.
- Never expose simulator RNG state or hidden draw order as model input.

The first engineering corpus is limited to 256 states. The pilot corpus uses a
pre-registered fixed state count selected before outcome generation. It does
not keep only successful or high-regret states.

### Targets

Each candidate receives structured suffix targets rather than one arbitrary
scalar reward:

- terminal win;
- final act;
- final floor;
- final HP;
- proxy score;
- horizon-truncation marker.

Candidate ranking uses the pre-registered lexicographic outcome order
`win`, `final_act`, `final_floor`, `final_hp`, `proxy_score`. Regression heads
retain the raw component targets. No metric is normalized by the candidate's
own batch maximum.

### Losses

The initial M8-A ablation changes only the objective and output heads:

```text
L = L_DAgger_CE
  + lambda_outcome * L_structured_outcome
  + lambda_rank * L_pairwise_candidate_rank
```

`L_DAgger_CE` remains identical to M7-C. `L_structured_outcome` uses binary
cross-entropy for win/act-clear targets and robust regression for floor, HP,
and proxy score. `L_pairwise_candidate_rank` compares candidates from the same
reconstructed state. Loss coefficients, corpus size, and inference rule must be
registered before pilot labels are inspected.

The first pilot evaluates two inference modes fixed in advance:

1. policy-only, which checks whether auxiliary value learning improves the
   shared representation without using value at action time;
2. policy-plus-rank, which combines policy logits with standardized candidate
   rank scores using a single registered coefficient.

## M8-B: Conditional Candidate Attention

M8-B is built only if the diagnostic decision rule triggers it. It keeps the
same recurrent context, corpora, losses, optimizer budget, and validation
ranges as M8-A. The only change is permutation-invariant state-token attention
queried by candidate actions. A full trajectory Transformer, language model,
map-specific graph network, or larger training budget is outside this ablation.

## Pilot Comparisons

The M8 pilot must include:

- frozen M7-C round-2 GRU checkpoint;
- M8-A policy-only;
- M8-A policy-plus-rank;
- M8-B only if the diagnostic trigger fires;
- frozen M6 initialization;
- frozen heuristic.

All methods use identical environment seeds and policy seeds. The pilot reports
paired final-floor, Act 1 clear, Act 2 clear, win rate, decisions, runtime, and
all safety counters. Raw episode records remain available for paired bootstrap
analysis.

## Seed And Data Isolation

Before collection, a new `m8_protocol` registry must reserve disjoint ranges
for engineering smoke, rollout-state sampling, validation, pilot evaluation,
formal gate, and a future blind range. The registry must reject overlap with
all M6, M7, M7-B, and M7-C ranges, including `3000000-3002047`.

No exact M8 range is valid until it appears in the registry, configuration,
tests, and server runbook in the same immutable commit. M7-C promotion seeds
`2220000-2220511` may appear only in the frozen M7-C report and cannot evaluate
M8.

## Promotion Requirements

M8 does not advance to multi-seed formal training unless one registered
candidate satisfies all of the following on the fresh pilot range:

- every safety counter is zero;
- paired final-floor 95% interval lower bound is greater than zero against the
  frozen M7-C round-2 checkpoint;
- paired Act 1 clear difference is non-negative against M7-C;
- heuristic final-floor non-inferiority lower bound is greater than `-1.0`;
- heuristic Act 1 difference is at least `-0.02`;
- the selected method was fixed before the pilot outcomes were inspected.

A failure freezes M8 as a negative result and returns to diagnosis. It does not
authorize more pilot seeds, hidden coefficient tuning, reuse of the pilot
range, or blind evaluation.

## Required Engineering Before Training

1. Complete and archive the M7-C post-hoc diagnostic.
2. Implement hash-bound counterfactual corpus manifests and replay validation.
3. Add structured outcome and within-state ranking targets without exposing
   hidden simulator state.
4. Add deterministic checkpoint/resume and safe-stop coverage.
5. Register disjoint M8 ranges and machine-readable promotion audit.
6. Pass the complete local and server test suites plus a 256-state smoke.
7. Freeze code, configuration, M7-C initialization, teacher identity, native
   extension identity, and counterfactual corpus before the pilot starts.
