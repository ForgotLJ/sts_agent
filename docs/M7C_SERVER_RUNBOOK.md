# M7-C Server Runbook

## Scope And Stop Rules

This runbook executes the pre-registered seed-17 GRU DAgger control only. It
does not start an attention ablation, seeds 29/43, the formal paired gate, or
the M7 final blind range. It stops after the promotion audit regardless of the
verdict.

Never access or reuse these historical evaluation ranges:

- `1400000-1400511` M7 pilot gate;
- `1500000-1500511` M7-B teacher validation;
- `1600000-1600511` M7-B paired gate;
- `2000000-2001023` revealed M6 final data;
- `3000000-3002047` M7 final blind data.

The only M7-C end-to-end evaluation range used here is promotion
`2220000-2220511`. The formal M7-C gate `2221000-2221511` remains untouched.

## Immutable Inputs

The code revision must be the released M7-C revision supplied by the operator.
The frozen input archive contains exactly these artifacts:

| Artifact | Identity |
| --- | --- |
| M7-B teacher corpus | seeds `400000-404095`, SHA-256 `0dfdd54bccc66b6c16b2f4515fa160ecc46752f21dc1022d1032c57b026fdd14` |
| M7-B initialization | `m7b`, seed 17, SHA-256 `ca6b91ac701306c2dca5aa4e1eef217691c41f515df2b6b250a99d1f2b728383` |
| M6 promotion baseline | `m6`, seed 17, SHA-256 `51b2e02e87af9753c9f2b0eba8a731733426d51c78cdc6bcc23114a8bdae83d5` |

The current input release archive is `m7c-frozen-inputs-v1.tar.gz` with
SHA-256 `01268c47cbf63689c4a37ba949d0927f60dd1b32274a6d8f6cf455711173cc8d`.
Do not substitute a corpus, checkpoint, or archive whose identity differs.

## Server Layout

Use only the existing project tree and its writable experiment directory:

```bash
ROOT=/scratch/sts_agent
WORKTREE="$ROOT/.m7c-control-worktree"
EXPERIMENTS="$ROOT/experiments"
PYTHON="$ROOT/.venv/bin/python"
INPUTS="$EXPERIMENTS/m7c-frozen-inputs"
RUN="$EXPERIMENTS/m7c_dagger_seed17"
```

Do not create new top-level `/scratch` directories and do not use `sudo`.
Do not modify the original `$ROOT` worktree or existing M6/M7-B outputs.

## Code And Runtime Preflight

Replace `<M7C_REVISION>` with the immutable Git tag or commit supplied with
the M7-C code release. If the target worktree already exists and is dirty,
stop and report it. Do not reset, clean, or delete it.

```bash
cd "$ROOT"
git fetch --tags origin

if test -e "$WORKTREE"; then
  cd "$WORKTREE"
  test -z "$(git status --porcelain)"
  git switch --detach <M7C_REVISION>
else
  git worktree add --detach "$WORKTREE" <M7C_REVISION>
  cd "$WORKTREE"
fi
git rev-parse HEAD

EXTENSION=$(find "$ROOT/build" -type f -name 'slaythespire*.so' -print -quit)
test -n "$EXTENSION"
EXTENSION_DIR=$(dirname "$EXTENSION")
WORKTREE_EXTENSION="$WORKTREE/build/$(basename "$EXTENSION_DIR")"
mkdir -p "$WORKTREE/build"
if test -e "$WORKTREE_EXTENSION" || test -L "$WORKTREE_EXTENSION"; then
  test -L "$WORKTREE_EXTENSION"
  test "$(readlink -f "$WORKTREE_EXTENSION")" = "$EXTENSION_DIR"
else
  ln -s "$EXTENSION_DIR" "$WORKTREE_EXTENSION"
fi
export PYTHONPATH="$WORKTREE/src:$WORKTREE_EXTENSION"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$PYTHON" - <<'PY'
import slaythespire
import torch
assert torch.cuda.is_available()
print({"torch": torch.__version__, "cuda": torch.cuda.get_device_name(0)})
PY
nvidia-smi

"$PYTHON" -B -m unittest discover -s tests -v
```

All tests must pass. CUDA, the native extension, or test failure is a hard
stop. The verified symlink keeps the shared native build outside the Git
worktree while allowing runtime manifests to hash the exact loaded extension.
Do not fall back to CPU for the formal training phase.

## Import Frozen Inputs

Download the archive and its `.sha256` sidecar from the M7-C input Release to
a temporary download path. Verify the archive before importing it. If
`$INPUTS` already exists, only run the verification command; do not pass
`--force` during a formal run.

```bash
cd "$WORKTREE"
sha256sum -c /path/to/m7c-frozen-inputs-v1.tar.gz.sha256

if test -e "$INPUTS"; then
  "$PYTHON" -B scripts/verify-m7c-frozen-inputs.py --root "$INPUTS"
else
  "$PYTHON" -B scripts/import-m7c-frozen-inputs.py \
    /path/to/m7c-frozen-inputs-v1.tar.gz \
    --destination "$INPUTS"
fi

TEACHER="$INPUTS/teacher-train/manifest.json"
M7B_INITIAL="$INPUTS/m7b-seed-17-best-evaluation-checkpoint.pt"
M6_BASELINE="$INPUTS/m6-seed-17-baseline-checkpoint.pt"
```

The verifier re-hashes 4100 files and validates the M7-B corpus manifest plus
both frozen checkpoint identities. It must report `verified: true` before any
collection begins.

## Resource Controls

Use the following starting values on the 18-core/36-thread server:

```bash
COLLECT_WORKERS=28
TORCH_THREADS=6
TORCH_INTEROP_THREADS=1
```

DAgger collection uses CPU workers so that independent simulator processes can
use the server cores. The model is small; GPU collection is intentionally not
used because its one-worker constraint is slower. Training remains CUDA-only.
These controls adjust throughput only, not seed ranges, labels, model, loss,
or validation criteria.

## Engineering Smoke

This uses the registered development-only range and cannot be used for model
selection. It establishes server collection throughput before a 1024-seed
round. The resulting directory must stay separate from formal outputs.

```bash
SMOKE="$EXPERIMENTS/m7c_server_collection_smoke"
"$PYTHON" -B scripts/collect-m7c-dagger-corpus.py \
  --checkpoint "$M7B_INITIAL" \
  --seed-range-name m7c_smoke \
  --round-index 0 \
  --teacher-mix-probability 0.0 \
  --run-seed 17 \
  --device cpu \
  --workers 8 \
  --progress-interval 8 \
  --output "$SMOKE"
```

Record wall time from `collection-summary.json`. A formal 1024-seed collection
is estimated from this measurement; do not alter the protocol because the
estimate is inconvenient.

## Teacher Anchor

Collect the independent teacher-state anchor once. It is retained unchanged
for all three rounds. The earlier `teacher-anchor` collection used the default
`1000`-decision horizon and failed at seed `2210169`. The subsequent
`teacher-anchor-5000` attempt reached the fixed `5000`-decision limit at the
same seed. Neither directory has a manifest. Retain both incomplete directories
for audit, but do not reuse, overwrite, or remove them. The formal anchor is a
fresh deterministic collection into `teacher-anchor-5000-truncated`; its
manifest and every trace bind the `5000`-decision horizon and explicit
horizon-truncation policy.

```bash
ANCHOR="$RUN/teacher-anchor-5000-truncated"
"$PYTHON" -B scripts/collect-m7b-teacher-corpus.py \
  --seed-start 2210000 \
  --seed-count 512 \
  --max-steps 5000 \
  --allow-horizon-truncation \
  --workers "$COLLECT_WORKERS" \
  --progress-interval 64 \
  --output "$ANCHOR"
```

## Round 0

Round 0 behavior begins at the frozen M7-B checkpoint. Its DAgger collection
uses 50% deterministic teacher mixing; its on-policy validation uses no
mixing. Each collector is resumable: rerun the identical command after an
interruption and it reuses verified completed traces.

```bash
D0="$RUN/dagger-round-0"
V0="$RUN/on-policy-round-0"
T0="$RUN/training"

"$PYTHON" -B scripts/collect-m7c-dagger-corpus.py \
  --checkpoint "$M7B_INITIAL" \
  --seed-range-name dagger_round_0 \
  --round-index 0 \
  --teacher-mix-probability 0.5 \
  --run-seed 17 --device cpu --workers "$COLLECT_WORKERS" \
  --progress-interval 64 --output "$D0"

"$PYTHON" -B scripts/collect-m7c-dagger-corpus.py \
  --checkpoint "$M7B_INITIAL" \
  --seed-range-name on_policy_round_0 \
  --round-index 0 \
  --teacher-mix-probability 0.0 \
  --run-seed 17 --device cpu --workers "$COLLECT_WORKERS" \
  --progress-interval 64 --output "$V0"

STOP0="$T0/STOP"
test ! -e "$STOP0"
nohup "$PYTHON" -B scripts/train-m7c-dagger.py \
  --config config/m7c_dagger_control.json \
  --run-seed 17 --round-index 0 \
  --teacher-corpus "$TEACHER" \
  --train-corpus "dagger_round_0=$D0/manifest.json" \
  --on-policy-validation-corpus "on_policy_round_0=$V0/manifest.json" \
  --teacher-anchor-validation-corpus "teacher_anchor=$ANCHOR/manifest.json" \
  --initialize-from "$M7B_INITIAL" \
  --output "$T0" --stop-file "$STOP0" \
  --progress-interval-batches 4 \
  --torch-threads "$TORCH_THREADS" \
  --torch-interop-threads "$TORCH_INTEROP_THREADS" \
  >"$T0/train-round-0.log" 2>&1 &
echo $! >"$T0/train-round-0.pid"
```

Round 0 completes only when the log reports `state: complete` and
`$T0/seed-17/round-0/best-evaluation-checkpoint.pt` exists. The selection
checkpoint is evaluation-only and is the sole allowed behavior checkpoint for
Round 1.

## Round 1

```bash
R0="$T0/seed-17/round-0/best-evaluation-checkpoint.pt"
D1="$RUN/dagger-round-1"
V1="$RUN/on-policy-round-1"

"$PYTHON" -B scripts/collect-m7c-dagger-corpus.py \
  --checkpoint "$R0" --seed-range-name dagger_round_1 --round-index 1 \
  --teacher-mix-probability 0.25 --run-seed 17 --device cpu \
  --workers "$COLLECT_WORKERS" --progress-interval 64 --output "$D1"
"$PYTHON" -B scripts/collect-m7c-dagger-corpus.py \
  --checkpoint "$R0" --seed-range-name on_policy_round_1 --round-index 1 \
  --teacher-mix-probability 0.0 --run-seed 17 --device cpu \
  --workers "$COLLECT_WORKERS" --progress-interval 64 --output "$V1"

STOP1="$T0/STOP-round-1"
test ! -e "$STOP1"
nohup "$PYTHON" -B scripts/train-m7c-dagger.py \
  --config config/m7c_dagger_control.json --run-seed 17 --round-index 1 \
  --teacher-corpus "$TEACHER" \
  --train-corpus "dagger_round_0=$D0/manifest.json" \
  --train-corpus "dagger_round_1=$D1/manifest.json" \
  --on-policy-validation-corpus "on_policy_round_1=$V1/manifest.json" \
  --teacher-anchor-validation-corpus "teacher_anchor=$ANCHOR/manifest.json" \
  --initialize-from "$R0" --output "$T0" --stop-file "$STOP1" \
  --progress-interval-batches 4 --torch-threads "$TORCH_THREADS" \
  --torch-interop-threads "$TORCH_INTEROP_THREADS" \
  >"$T0/train-round-1.log" 2>&1 &
echo $! >"$T0/train-round-1.pid"
```

## Round 2

```bash
R1="$T0/seed-17/round-1/best-evaluation-checkpoint.pt"
D2="$RUN/dagger-round-2"
V2="$RUN/on-policy-round-2"

"$PYTHON" -B scripts/collect-m7c-dagger-corpus.py \
  --checkpoint "$R1" --seed-range-name dagger_round_2 --round-index 2 \
  --teacher-mix-probability 0.0 --run-seed 17 --device cpu \
  --workers "$COLLECT_WORKERS" --progress-interval 64 --output "$D2"
"$PYTHON" -B scripts/collect-m7c-dagger-corpus.py \
  --checkpoint "$R1" --seed-range-name on_policy_round_2 --round-index 2 \
  --teacher-mix-probability 0.0 --run-seed 17 --device cpu \
  --workers "$COLLECT_WORKERS" --progress-interval 64 --output "$V2"

STOP2="$T0/STOP-round-2"
test ! -e "$STOP2"
nohup "$PYTHON" -B scripts/train-m7c-dagger.py \
  --config config/m7c_dagger_control.json --run-seed 17 --round-index 2 \
  --teacher-corpus "$TEACHER" \
  --train-corpus "dagger_round_0=$D0/manifest.json" \
  --train-corpus "dagger_round_1=$D1/manifest.json" \
  --train-corpus "dagger_round_2=$D2/manifest.json" \
  --on-policy-validation-corpus "on_policy_round_2=$V2/manifest.json" \
  --teacher-anchor-validation-corpus "teacher_anchor=$ANCHOR/manifest.json" \
  --initialize-from "$R1" --output "$T0" --stop-file "$STOP2" \
  --progress-interval-batches 4 --torch-threads "$TORCH_THREADS" \
  --torch-interop-threads "$TORCH_INTEROP_THREADS" \
  >"$T0/train-round-2.log" 2>&1 &
echo $! >"$T0/train-round-2.pid"
```

## Safe Pause And Resume

To pause a running round, create its declared stop file or send `SIGTERM` to
the PID. The trainer finishes the current trace batch and atomically saves a
checkpoint.

```bash
touch "$STOP1"
kill -TERM "$(cat "$T0/train-round-1.pid")"
tail -f "$T0/train-round-1.log"
```

To resume, remove only that round's stop file and rerun its training command,
replacing `--initialize-from <previous-round-checkpoint>` with:

```bash
--resume "$T0/seed-17/round-1/checkpoint.pt"
```

Keep all corpus and validation arguments identical. The checkpoint checks every
aggregate corpus hash, validation hash, and frozen teacher identity before
continuing.

## Promotion Audit

Only run this after Round 2 completes. The candidate uses the selected Round 2
checkpoint, not the resumable completion checkpoint. Evaluate all three methods
on exactly the promotion range, then run the machine-readable audit.

```bash
R2="$T0/seed-17/round-2/best-evaluation-checkpoint.pt"
PROMOTION="$RUN/promotion"
mkdir -p "$PROMOTION"

"$PYTHON" -B scripts/evaluate-m7.py \
  --method learned-heuristic --report-label m7c-dagger --checkpoint "$R2" \
  --seed-start 2220000 --seed-count 512 --m7c-range-name promotion \
  --policy-seed 17 --bootstrap-samples 10000 \
  --output "$PROMOTION/m7c-dagger.json"
"$PYTHON" -B scripts/evaluate-m7.py \
  --method learned-heuristic --report-label m6-initial --checkpoint "$M6_BASELINE" \
  --seed-start 2220000 --seed-count 512 --m7c-range-name promotion \
  --policy-seed 17 --bootstrap-samples 10000 \
  --output "$PROMOTION/m6-initial.json"
"$PYTHON" -B scripts/evaluate-m7.py \
  --method heuristic --report-label heuristic \
  --seed-start 2220000 --seed-count 512 --m7c-range-name promotion \
  --policy-seed 17 --bootstrap-samples 10000 \
  --output "$PROMOTION/heuristic.json"
"$PYTHON" -B scripts/summarize-m7-evaluations.py \
  --evaluation "$PROMOTION/m7c-dagger.json" \
  --evaluation "$PROMOTION/m6-initial.json" \
  --evaluation "$PROMOTION/heuristic.json" \
  --reference-method m6-initial --reference-method heuristic \
  --bootstrap-samples 10000 \
  --output "$PROMOTION/summary.json"
"$PYTHON" -B scripts/audit-m7c.py \
  --end-to-end-summary "$PROMOTION/summary.json" \
  --gate-seed-start 2220000 --gate-seed-count 512 \
  --output "$PROMOTION/audit.json"
```

`audit.json` is authoritative. A failure stops the program before attention,
formal paired gate, seed 29/43 work, or blind evaluation. A pass only opens the
pre-registered attention ablation design; it does not authorize the formal
gate automatically.

## Post-Hoc Diagnostic

After the promotion audit has stopped the formal run, the frozen outputs may be
summarized descriptively without running another policy evaluation. This report
verifies all six corpus manifests and trace hashes, reads each selected
validation checkpoint record, and records behavior disagreement by round,
phase, and floor. It explicitly marks observed outcome differences as
associations rather than counterfactual regret and cannot select a model.

```bash
DIAGNOSTIC="$RUN/diagnostics/m7c-posthoc.json"
test ! -e "$DIAGNOSTIC"
"$PYTHON" -B scripts/diagnose-m7c.py \
  --run-root "$RUN" \
  --run-seed 17 \
  --output "$DIAGNOSTIC"
```

Do not pass `--skip-trace-hashes` for a retained experiment result. The option
exists only for local engineering tests where the trace archive is unavailable.
Running this diagnostic does not reopen the promotion range or authorize any
additional training, attention ablation, paired gate, or blind evaluation.

## Release Artifacts

Keep raw traces and ordinary checkpoints on the server. For a result package,
retain the frozen-input verification output, the seven corpus manifests and
collection summaries, each round's resolved config/manifest/metrics/best
validation/selected checkpoint, promotion method JSON files, summary, audit,
and SHA-256 checksums. Exclude raw trace bodies, virtual environments, build
directories, TensorBoard temporary data, and unselected ordinary checkpoints.
