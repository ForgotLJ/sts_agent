# M7-B Formal Seed-17 Result

## Protocol

The formal local run was completed on 2026-07-29 with the pre-registered M7-B
protocol. It used:

- teacher training seeds `400000-404095` (4096 runs);
- held-out teacher validation seeds `1500000-1500511` (512 runs);
- paired end-to-end gate seeds `1600000-1600511` (512 runs);
- M6 seed-17 as the frozen initialization and baseline;
- pure non-combat teacher-action cross-entropy;
- no value, PPO, entropy, or uniform-exploration loss.

The final blind range `3000000-3002047` and the previously exposed M7 pilot
range `1400000-1400511` were not accessed.

## Data And Training

The teacher corpus completed without collection errors. The 4096-run training
corpus contained 275,027 non-combat phase visits and the 512-run validation
corpus contained 34,778. Both corpora and their encoded replay caches have
aggregate SHA-256 manifests.

Training ran for seven epochs and stopped normally by early stopping after
three consecutive non-improvements. Epoch 4 was selected as the best
checkpoint because its weakest validation phase, `map`, reached 74.19%.

## Teacher-Action Gate

The teacher-action gate passed on the fixed 512-run validation corpus.

| Metric | M6 initial | M7-B | Difference |
|---|---:|---:|---:|
| Overall accuracy | 83.58% | 91.43% | +7.84 pp |
| Cross-entropy | 0.4755 | 0.2147 | -0.2608 |
| Card reward accuracy | 88.64% | 94.10% | +5.45 pp |
| Event accuracy | 93.59% | 99.97% | +6.38 pp |
| Map accuracy | 62.65% | 74.19% | +11.54 pp |
| Rest-site accuracy | 81.17% | 97.84% | +16.67 pp |
| Shop accuracy | 74.41% | 88.49% | +14.08 pp |

All phase accuracies were non-regressing, overall accuracy increased, and
overall cross-entropy decreased.

## End-To-End Gate

The paired end-to-end gate failed.

| Metric | M6 initial | M7-B | Difference |
|---|---:|---:|---:|
| Mean final floor | 17.2129 | 15.3398 | -1.8730 |
| Act 1 clear rate | 39.06% | 27.34% | -11.72 pp |
| Act 2 clear rate | 0.59% | 0.39% | -0.20 pp |
| Win rate | 0.20% | 0.00% | -0.20 pp |

The paired hierarchical bootstrap 95% interval for the final-floor difference
was `[-2.6094, -1.1523]`. Errors, crashes, illegal actions, recovery failures,
timeouts, and cycles were all zero for both methods.

Formal audit verdict: `FAIL`.

## Completion Verification

The completion audit re-read every source trace and re-hashed every encoded
replay batch on 2026-07-29. The authoritative digests are:

| Artifact | SHA-256 |
|---|---|
| Training corpus | `0dfdd54bccc66b6c16b2f4515fa160ecc46752f21dc1022d1032c57b026fdd14` |
| Validation corpus | `07df86cc9712a883c27627c627a47d2d7e6bbf26551b0faffb1004f748b776d9` |
| Training replay cache | `e8867b98954d076d707b2dc842482e933e73697f18a84559c2952352ed1e09bf` |
| Validation replay cache | `47c3344860cf0830d26b4e27c458b4cacd7b25cbbaedda34e1919bdd8dc5c109` |
| Best M7-B checkpoint | `ca6b91ac701306c2dca5aa4e1eef217691c41f515df2b6b250a99d1f2b728383` |

The completion checkpoint records 448 completed trace batches, seven completed
epochs, `epochs_without_improvement = 3`, and
`completion_reason = early_stopping`. The best checkpoint matches the candidate
used by both gates. Both end-to-end reports contain exactly the ordered seed set
`1600000-1600511` with no duplicates.

The full repository test suite passed `139/139`. No seed-29 or seed-43 run was
created. The protected final range appears only as a declared range in runtime
manifests; no trace, checkpoint, or evaluation was produced for it. No M7 pilot
gate seed was reused.

## Interpretation

M7-B answered its main diagnostic question: the recurrent policy can fit the
heuristic teacher substantially better on held-out teacher-forced states.
However, that improvement does not survive closed-loop execution. Small action
errors alter the route, deck, rewards, and shop inventory, after which the
student visits states that are absent or rare in the fixed teacher corpus. The
result is consistent with compounding covariate shift, not insufficient epochs.

The next experiment should therefore use persistent DAgger-style aggregation:
collect states from the current student in closed loop, label those exact states
with the heuristic teacher, append them to a versioned corpus, and distill again
under the same two-stage gate. More epochs on the current fixed corpus, seed
29/43 replication, PPO restoration, or final-blind evaluation are not justified
by this result.
