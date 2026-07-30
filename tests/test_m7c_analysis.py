from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from sts_env import Action, ActionKind, Phase
from sts_env.trace import EpisodeTrace, TraceStep
from sts_env.training.m7c_analysis import (
    build_m7c_diagnostic_report,
    summarize_m7c_behavior_associations,
)
from sts_env.training.m7c_dagger import (
    M7CDaggerLabel,
    M7C_DAGGER_TRACE_PROTOCOL,
    build_m7c_corpus_manifest,
    summarize_m7c_on_policy_labels,
)


def trace(
    seed: int,
    *,
    round_index: int,
    correct: bool,
    final_floor: int,
    teacher_mixed: bool = False,
) -> EpisodeTrace:
    teacher = Action(ActionKind.CHOOSE_CARD, source_id="teacher")
    behavior = teacher if correct else Action(ActionKind.CHOOSE_CARD, source_id="student")
    labels = (
        M7CDaggerLabel(
            step_index=0,
            teacher_action=teacher,
            behavior_action_index=0,
            student_action_index=0,
            phase=Phase.CARD_REWARD,
            teacher_mixed=teacher_mixed,
            floor=2,
            act=1,
            legal_action_count=2,
            policy_entropy=0.4 if correct else 0.7,
            policy_margin=1.2 if correct else 0.3,
        ),
    )
    return EpisodeTrace(
        seed=seed,
        initial_observation_digest="fixture",
        steps=(
            TraceStep(
                action=behavior,
                observation_digest="next",
                reward=0.0,
                terminated=True,
                truncated=False,
                info={},
            ),
        ),
        backend="m7c-analysis-test",
        metadata={
            "protocol": M7C_DAGGER_TRACE_PROTOCOL,
            "round_index": round_index,
            "teacher_mix_probability": 0.0,
            "mixing_seed": seed,
            "behavior_policy": {"checkpoint_sha256": f"round-{round_index}"},
            "teacher_identity": "fixture-teacher",
            "phase_supervision_counts": {
                "card_reward": 1,
                "event": 0,
                "map": 0,
                "rest_site": 0,
                "shop": 0,
            },
            "student_noncombat_steps": 1,
            "mixed_noncombat_steps": 0,
            "horizon_truncated": False,
            "final_act": 1,
            "final_floor": final_floor,
            "won": False,
            "environment_return": 0.0,
            "dagger_labels": [label.to_dict() for label in labels],
        },
    )


def write_corpus(root: Path, *, round_index: int, seed_start: int) -> None:
    traces_root = root / "traces"
    traces_root.mkdir(parents=True)
    traces = (
        trace(seed_start, round_index=round_index, correct=True, final_floor=12),
        trace(seed_start + 1, round_index=round_index, correct=False, final_floor=5),
    )
    for episode in traces:
        episode.write_jsonl(traces_root / f"seed-{episode.seed:08d}.jsonl")
    manifest = build_m7c_corpus_manifest(
        root,
        seed_start=seed_start,
        seed_count=2,
        round_index=round_index,
    )
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    diagnostic = summarize_m7c_on_policy_labels(traces)
    diagnostic.update(
        {
            "seed_range_name": f"fixture-{round_index}",
            "seed_range": manifest["seed_range"],
            "round_index": round_index,
            "corpus_sha256": manifest["aggregate_sha256"],
            "checkpoint": manifest["behavior_policy"],
        }
    )
    (root / "on-policy-diagnostic.json").write_text(
        json.dumps(diagnostic),
        encoding="utf-8",
    )


class M7CAnalysisTests(unittest.TestCase):
    def test_behavior_summary_marks_observational_association(self) -> None:
        summary = summarize_m7c_behavior_associations(
            (
                trace(1, round_index=0, correct=True, final_floor=12),
                trace(2, round_index=0, correct=False, final_floor=5),
                trace(
                    3,
                    round_index=0,
                    correct=True,
                    final_floor=50,
                    teacher_mixed=True,
                ),
            )
        )

        self.assertEqual(summary["student_decisions"]["agreement"], 0.5)
        self.assertEqual(summary["episodes"]["observed_floor_association"], -7.0)
        self.assertEqual(summary["episodes"]["without_student_decisions"], 1)
        self.assertFalse(summary["causal_regret_estimated"])

    def test_report_verifies_rounds_and_promotion_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for round_index in range(3):
                write_corpus(
                    root / f"dagger-round-{round_index}",
                    round_index=round_index,
                    seed_start=100 + round_index * 10,
                )
                write_corpus(
                    root / f"on-policy-round-{round_index}",
                    round_index=round_index,
                    seed_start=200 + round_index * 10,
                )
                training = root / "training" / "seed-17" / f"round-{round_index}"
                training.mkdir(parents=True)
                validation = {
                    "type": "validation",
                    "epoch": 1,
                    "selection_key": [0.5, 0.5],
                    "on_policy": {"accuracy": 0.5},
                    "teacher_anchor": {"accuracy": 0.6},
                }
                (training / "manifest.json").write_text(
                    json.dumps(
                        {
                            "protocol": "m7c-dagger",
                            "run_seed": 17,
                            "round_index": round_index,
                        }
                    ),
                    encoding="utf-8",
                )
                (training / "best-validation.json").write_text(
                    json.dumps(validation),
                    encoding="utf-8",
                )
                (training / "metrics.jsonl").write_text(
                    json.dumps(validation) + "\n",
                    encoding="utf-8",
                )
                (training / "best-evaluation-checkpoint.pt").write_bytes(b"fixture")

            promotion = root / "promotion"
            promotion.mkdir()
            for name in ("m7c-dagger.json", "m6-initial.json", "heuristic.json"):
                (promotion / name).write_text(
                    json.dumps({"method": name.removesuffix(".json")}),
                    encoding="utf-8",
                )
            (promotion / "summary.json").write_text(
                json.dumps(
                    {
                        "paired_comparisons": {
                            "m7c-dagger_minus_m6-initial": {},
                            "m7c-dagger_minus_heuristic": {},
                        }
                    }
                ),
                encoding="utf-8",
            )
            (promotion / "audit.json").write_text(
                json.dumps(
                    {
                        "verdict": "FAIL",
                        "complete": False,
                        "safety_clear": True,
                        "m7c_minus_m6": {"final_floor_requirement_met": False},
                        "m7c_minus_heuristic": {
                            "final_floor_requirement_met": False
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = build_m7c_diagnostic_report(root)

        self.assertEqual(report["promotion"]["audit_verdict"], "FAIL")
        self.assertEqual(len(report["rounds"]), 3)
        self.assertTrue(report["trace_hashes_verified"])
        self.assertFalse(report["model_selection_allowed"])


if __name__ == "__main__":
    unittest.main()
