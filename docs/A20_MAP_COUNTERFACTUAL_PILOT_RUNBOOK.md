# A20 Ironclad Map Counterfactual Pilot

## Scope

This runbook permits only the Act 1 counterfactual corpus pilot. It does not permit model training, corpus scaling, map-policy evaluation, threshold selection, or modification of historical source trees and experiment directories.

The source must be the annotated tag `a20-map-counterfactual-pilot-v1`. The frozen card-value checkpoint must have SHA-256 `8c7f053c64b9bd57ccba6ae64ecba8586a29d37dfaf1842f00d083b07b113a3c`.

## Isolated Worktree

```bash
set -euo pipefail

ROOT=/scratch/sts_agent
WORKTREE=$ROOT/.map-counterfactual-pilot-worktree
OUTPUT=$ROOT/experiments/map_counterfactual_act1_pilot
CHECKPOINT=$ROOT/experiments/a20_online_value_ironclad_v3/a20-online-value-IRONCLAD.pt

git -C "$ROOT" fetch --tags origin
git -C "$ROOT" rev-parse --verify 'a20-map-counterfactual-pilot-v1^{commit}'
test ! -e "$WORKTREE"
test ! -e "$OUTPUT"
git -C "$ROOT" worktree add --detach "$WORKTREE" a20-map-counterfactual-pilot-v1
cd "$WORKTREE"
```

Do not reset, clean, commit, or otherwise modify the original `$ROOT` worktree.

## Extension and Preflight

The worktree may reuse the previously verified native extension from the original tree, but it must not rebuild it. Create the link only when the isolated worktree has no extension and the original tree has exactly one compatible Linux extension.

```bash
mapfile -t EXTENSIONS < <(find "$ROOT/build" -type f -name 'slaythespire*.so' -print)
test "${#EXTENSIONS[@]}" -eq 1
EXTENSION=${EXTENSIONS[0]}
mkdir -p build/sts_lightspeed-py311
ln -s "$EXTENSION" "build/sts_lightspeed-py311/$(basename "$EXTENSION")"

export PYTHONPATH="$WORKTREE/src${PYTHONPATH:+:$PYTHONPATH}"
python -m unittest \
  tests.test_map_counterfactual \
  tests.test_map_action_protocol \
  tests.test_map_action_value
python scripts/check-lightspeed.py --imports 10 --seeds 1000
test "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" = \
  8c7f053c64b9bd57ccba6ae64ecba8586a29d37dfaf1842f00d083b07b113a3c
```

If any preflight check fails, stop. Do not build missing dependencies, downgrade to CPU/GPU alternatives, alter the code, or start collection.

## Pilot Collection

```bash
python scripts/collect-map-counterfactual-corpus.py \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --seed-start 2310000 --seed-count 64 \
  --seed-range-name map_act1_pilot \
  --acts 1 --per-act 16 \
  --particles-per-action 2 \
  --max-decisions-per-seed 1 \
  --rollout-max-steps 5000 \
  --override-margin 0.016514360904693604 \
  --device cpu

python scripts/validate-map-counterfactual-corpus.py \
  --input "$OUTPUT" --require-complete
```

## Required Report

Report the tag commit, checkpoint SHA-256, test result, Lightspeed result, collection wall time, and the complete `manifest.json` plus validator JSON. Explicitly report `complete`, `counts`, `errors`, `records.sha256`, `records`, and `act_counts`.

Stop after this report. Keep `$WORKTREE` and `$OUTPUT` intact for audit. Do not start the 300-per-act collection, training, profile, smoke, formal evaluation, or replication.
