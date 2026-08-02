# Map Counterfactual Rollout Corpus

This corpus is generated only from the local simulator. Each record contains a public map observation, every legal map action, the frozen clone-value baseline action, and terminal outcomes from redeterminized clone rollouts for every candidate action.

It is deliberately separate from the public run-history data. The labels are counterfactual outcomes under a fixed rollout policy, not claims about a globally optimal move. A record is accepted only if every candidate preserves the same public observation after cloning and retains its legal action identity.

## Act 1 Pilot

Run this only on a server with a compatible `sts_lightspeed` extension and the frozen A20 Ironclad value checkpoint:

```bash
python scripts/collect-map-counterfactual-corpus.py \
  --checkpoint /scratch/sts_agent/experiments/a20_online_value_ironclad_v3/a20-online-value-IRONCLAD.pt \
  --output /scratch/sts_agent/experiments/map_counterfactual_act1_pilot \
  --seed-start 2310000 --seed-count 64 \
  --seed-range-name map_act1_pilot \
  --acts 1 --per-act 16 \
  --particles-per-action 2 \
  --max-decisions-per-seed 1 \
  --override-margin 0.016514360904693604 \
  --device cpu

python scripts/validate-map-counterfactual-corpus.py \
  --input /scratch/sts_agent/experiments/map_counterfactual_act1_pilot \
  --require-complete

python scripts/diagnose-map-counterfactual-corpus.py \
  --input /scratch/sts_agent/experiments/map_counterfactual_act1_pilot \
  --output /scratch/sts_agent/experiments/map_counterfactual_act1_pilot/diagnostics.json \
  --min-records 16 --min-contrasting-fraction 0.20
```

The pilot writes only simulator-generated records and a manifest. It does not train a policy. Do not use formal evaluation seeds, and do not mix a failed/incomplete collection with a later corpus.

The first map layer normally exposes only `M` room symbols. The diagnostic reports room-symbol counts but gates candidate diversity on distinct public target coordinates; route topology, not an impossible first-layer room-symbol mixture, is the relevant signal for this pilot.

## Scale Gate

Only after the pilot manifest is `complete=true` and the validator accepts it, collect the fixed training corpus. The first multi-act collection exhausted its range with no Act 3 records under the frozen clone-value baseline, so it remains a failed historical artifact. The successor protocol is deliberately Act 1-only: it uses new seed ranges and never applies its model beyond Act 1. A shortfall is a failed collection, not permission to reuse the range with altered settings.

```bash
python scripts/collect-map-counterfactual-corpus.py \
  --checkpoint /scratch/sts_agent/experiments/a20_online_value_ironclad_v3/a20-online-value-IRONCLAD.pt \
  --output /scratch/sts_agent/experiments/map_counterfactual_act1_v2 \
  --seed-start 2322000 --seed-count 4096 \
  --seed-range-name map_act1_collection_v2 \
  --acts 1 --per-act 300 \
  --particles-per-action 2 --max-decisions-per-seed 1 \
  --rollout-max-steps 5000 \
  --override-margin 0.016514360904693604 --device cpu

python scripts/validate-map-counterfactual-corpus.py \
  --input /scratch/sts_agent/experiments/map_counterfactual_act1_v2 \
  --require-complete
```

## Offline Map Model

Only a complete corpus accepted by the validator may enter training. The model is action-conditioned: it sees the public player/deck/relic state, full public map graph, candidate node, immediate successors, and graph-reachable room statistics. Train/validation/test are split by root simulator seed, never by individual action record.

```bash
python scripts/train-map-action-value.py \
  --input /scratch/sts_agent/experiments/map_counterfactual_act1_v2 \
  --output /scratch/sts_agent/experiments/map_counterfactual_act1_v2/a20-map-action-value-IRONCLAD.pt \
  --frozen-evaluation /scratch/sts_agent/experiments/map_counterfactual_act1_v2/frozen-evaluation.json \
  --epochs 60 --groups-per-batch 32 --learning-rate 0.0003 --seed 17 --device cuda
```

The held-out metrics are counterfactual ranking diagnostics, not proof that the policy improves a full run. In particular, the reported model-minus-behavior final floor is measured on fixed rollout labels and must not be substituted for online evaluation.

## Online Protocol

The candidate policy wraps the already frozen p80 clone-value card policy. It only scores legal map-node actions from the acts declared in its checkpoint and returns the baseline action elsewhere unless the map-model advantage clears a separately calibrated map margin. Map inference adds no simulator clone calls.

First use a development-only, disjoint record-only range to inspect advantages. It must reproduce the reference episode outcomes exactly, because it always returns the baseline map action.

```bash
python scripts/evaluate-a20-map-value-policy.py \
  --checkpoint /scratch/sts_agent/experiments/map_counterfactual_act1_v2/a20-map-action-value-IRONCLAD.pt \
  --card-checkpoint /scratch/sts_agent/experiments/a20_online_value_ironclad_v3/a20-online-value-IRONCLAD.pt \
  --output /scratch/sts_agent/experiments/map_counterfactual_act1_v4/profile-2332000-2332127.json \
  --seed-start 2332000 --seed-count 128 \
  --seed-range-name map_act1_value_profile_v4 \
  --override-margin 0.0 --card-override-margin 0.016514360904693604 \
  --record-only --device cpu
```

Freeze a conservative margin from that profile before running any effect range. Then use new, non-overlapping seed ranges in order: 32-seed smoke, 512-seed formal evaluation, and an independent 512-seed replication. Compare against the p80 clone-value card policy, not the earlier heuristic-only policy. Promote only if both 512-seed runs have zero safety failures, final-floor bootstrap CI lower bounds above zero, and non-negative Act 1 mean differences.

After the two 512-seed results are frozen, audit them without rerunning either evaluation:

```bash
python scripts/audit-map-action-value.py \
  --formal /scratch/sts_agent/experiments/map_counterfactual_act1_v2/formal.json \
  --replication /scratch/sts_agent/experiments/map_counterfactual_act1_v2/replication.json \
  --output /scratch/sts_agent/experiments/map_counterfactual_act1_v2/audit.json \
  --map-checkpoint-sha256 '<frozen-map-checkpoint-sha256>' \
  --card-checkpoint-sha256 8c7f053c64b9bd57ccba6ae64ecba8586a29d37dfaf1842f00d083b07b113a3c \
  --formal-range-name map_act1_value_formal_v4 \
  --replication-range-name map_act1_value_replication_v4 \
  --trained-acts 1 --trained-floor-range 0 0
```

The audit verifies fixed ranges, disjoint episodes, both checkpoint hashes, safety counts, each 512-seed gate, and the pooled comparison. `replicated_improved` is required before the map module can become a new baseline.

## Autonomous Stage

For the server, use one fresh stage directory rather than pausing after every intermediate artifact. The runner re-diagnoses the fixed pilot, validates and reuses the complete 300-decision Act 1 corpus, trains once with a floor-coverage guard, profiles a pre-specified p80 margin, then runs smoke, formal, replication, and audit. It records `stage.json` and automatically stops at the first failed gate while preserving all evidence.

```bash
python scripts/run-a20-map-action-stage.py \
  --pilot /scratch/sts_agent/experiments/map_counterfactual_act1_pilot \
  --reuse-corpus /scratch/sts_agent/experiments/map_action_stage_v2/corpus \
  --output /scratch/sts_agent/experiments/map_action_stage_v4 \
  --card-checkpoint /scratch/sts_agent/experiments/a20_online_value_ironclad_v3/a20-online-value-IRONCLAD.pt \
  --rollout-device cpu \
  --training-device cuda \
  --evaluation-device cpu
```

The output directory must not already exist. A stopped stage is a valid result; inspect `stage.json` and its referenced artifacts rather than rerunning a failed seed range with modified settings.

Use `--dry-run` with the same three required paths to inspect the fixed seed ranges, devices, training settings, and promotion gates without accessing the checkpoint, pilot corpus, or simulator.
