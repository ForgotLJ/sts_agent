# Observation-Aligned A20 Ironclad Card Cold Start

This component learns only the card-reward decision from normal-mode A20 Ironclad Heart victories. It is not a combat policy and does not claim an end-to-end gameplay result.

## Feature Contract

The deployed model consumes only fields available in a live `Observation`:

- floor, act, ascension, HP, maximum HP, and gold
- current deck identity histogram and card-count scalars
- current relic and potion identity histograms
- the legal card-reward candidates, including an explicit skip candidate

The run-history encoder reconstructs the state before each current card choice from the Ironclad starting deck, strictly earlier card rewards, and strictly earlier event card changes. It excludes final `master_deck`, the current choice, all later records, shop purchases with ambiguous item types, and transformations with unknown result identities. This avoids future-label leakage, at the cost of an explicitly documented approximation to the historical deck.

Card IDs are lowercased, upgrades are removed, and non-alphanumeric characters are stripped. This matches the Lightspeed backend's public card-action identity convention.

## Offline Training

Use the filtered Ironclad Heart wins only:

```powershell
$env:PYTHONPATH='D:\Project\STS\sts_agent\.local_packages;D:\Project\STS\sts_agent\src'
python scripts\train-a20-online-card-ranker.py `
  --input D:\Project\STS\Data\formatted_a20h_2020-07-30\a20_heart_wins_IRONCLAD.jsonl `
  --output-dir D:\Project\STS\Data\a20_online_card_ranking_ironclad_v1 `
  --epochs 8 --batch-size 512 --seed 17 --device cpu
```

Run the frozen held-out evaluation separately:

```powershell
python scripts\evaluate-a20-online-card-ranker.py `
  --input D:\Project\STS\Data\formatted_a20h_2020-07-30\a20_heart_wins_IRONCLAD.jsonl `
  --checkpoint D:\Project\STS\Data\a20_online_card_ranking_ironclad_v1\a20-online-card-ranker-IRONCLAD.pt `
  --output D:\Project\STS\Data\a20_online_card_ranking_ironclad_v1\frozen-evaluation.json
```

## Simulator Evaluation

The model is wrapped by `A20OnlineCardRewardPolicy`. It handles only `Phase.CARD_REWARD`; every other legal action is delegated to `HeuristicPolicy`. The paired evaluation uses identical environment seeds for the wrapper and baseline:

```powershell
python scripts\evaluate-a20-online-card-policy.py `
  --checkpoint D:\Project\STS\Data\a20_online_card_ranking_ironclad_v1\a20-online-card-ranker-IRONCLAD.pt `
  --seed-start 2300000 --seed-count 128 `
  --output D:\Project\STS\Data\a20_online_card_ranking_ironclad_v1\lightspeed-paired.json
```

This command requires a compatible built `sts_lightspeed` extension. Its result is diagnostic evidence, not a promotion gate or a claim of A20 Heart performance. Keep raw run-history data out of source control until its license and redistribution terms are confirmed.

## Clone-Value Candidate

The direct behavior-cloning checkpoint is retained as a negative control. It passed offline action agreement but was rejected by a frozen 512-seed Lightspeed comparison: its paired final-floor difference was `-2.232421875` with 95% CI `[-2.666015625, -1.81640625]`.

The next candidate uses all filtered Ironclad A20 attempts, not only Heart wins. Value examples are encoded after each logged card choice so they match the state produced by a simulator clone after a candidate action. At runtime, `A20CloneValueCardRewardPolicy` clones each legal card-reward action, scores the successor with the value model, and overrides the heuristic only when the predicted normalized final-floor advantage reaches the configured margin.

Train the value model from the A20 Ironclad attempts:

```powershell
python scripts\train-a20-online-value.py `
  --input D:\Project\STS\Data\formatted_a20h_2020-07-30\a20_runs_IRONCLAD.jsonl `
  --output-dir D:\Project\STS\Data\a20_online_value_ironclad_v3 `
  --epochs 8 --batch-size 1024 --seed 17 --device cpu
```

The clone-value simulator command is intentionally separate from the rejected behavior-cloning command:

```powershell
python scripts\evaluate-a20-clone-value-policy.py `
  --checkpoint D:\Project\STS\Data\a20_online_value_ironclad_v3\a20-online-value-IRONCLAD.pt `
  --seed-start 2302000 --seed-count 32 `
  --override-margin 0.05 `
  --output D:\Project\STS\Data\a20_online_value_ironclad_v3\smoke-2302000-2302031.json
```

This smoke must pass all safety fields before any larger diagnostic. The value model is still observational: a positive offline AUC does not establish a causal action advantage, so simulator evaluation remains mandatory.
