from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch

from sts_env import (
    Action,
    ActionKind,
    EpisodeTrace,
    LightspeedBackend,
    MapNodeView,
    Observation,
    Phase,
    PlayerView,
    StsEnv,
    ToyCombatBackend,
    TraceStep,
    observation_digest,
    record_episode,
    replay_trace,
)
from sts_env.training import (
    CandidateQConfig,
    CandidateQTrainer,
    CurriculumEnvironment,
    CurriculumScheduler,
    CurriculumSpec,
    DaggerConfig,
    HeuristicPolicy,
    LightspeedEnvironmentFactory,
    M6TrainingConfig,
    MultiprocessRecurrentRolloutCollector,
    ObjectFeatureEncoder,
    ParameterEMA,
    ParameterEMAConfig,
    RandomPolicy,
    RecurrentPPOConfig,
    RecurrentPPOTrainer,
    RecurrentRolloutCollector,
    ReplayBuffer,
    PrefixCorpus,
    RunFeatureEncoder,
    SeedSplit,
    SynchronousVectorCollector,
    SubprocessVectorEnvironment,
    build_imitation_chunks,
    collect_dagger_chunks,
    dagger_training_seeds,
    evaluate_policy,
    evaluate_full_runs,
    imitation_trace_progress,
    is_self_imitation_candidate,
    load_m6_checkpoint,
    m6_validation_selection_key,
    rank_imitation_traces,
    save_m6_checkpoint,
    select_weighted_frontier_traces,
    summarize_m6_evaluations,
    train_self_imitation,
    validate_m6_evaluation_seed_range,
)
from sts_env.training.experiment import build_runtime_manifest


class LoopingEnvironment:
    loop_action = Action(ActionKind.CHOOSE_OPTION, source_id="loop", option_type="loop")
    finish_action = Action(
        ActionKind.CHOOSE_OPTION,
        source_id="finish",
        option_type="finish",
    )

    def __init__(self) -> None:
        self._observation = self._active_observation()

    @property
    def observation(self) -> Observation:
        return self._observation

    def reset(self, seed: int | None = None) -> tuple[Observation, dict[str, object]]:
        self._observation = self._active_observation()
        return self._observation, {"backend": "loop", "seed": seed}

    def step(
        self,
        action: int | Action,
    ) -> tuple[Observation, float, bool, bool, dict[str, object]]:
        resolved = self._observation.legal_actions[action] if isinstance(action, int) else action
        if resolved == self.loop_action:
            return self._observation, 0.0, False, False, {"backend": "loop"}
        if resolved != self.finish_action:
            raise ValueError("unexpected loop test action")
        self._observation = replace(
            self._observation,
            phase=Phase.TERMINAL,
            legal_actions=(),
        )
        return self._observation, 1.0, True, False, {"backend": "loop"}

    def clone(self) -> LoopingEnvironment:
        cloned = LoopingEnvironment()
        cloned._observation = self._observation
        return cloned

    @classmethod
    def trace(cls, loop_count: int) -> EpisodeTrace:
        environment = cls()
        observation, _ = environment.reset(seed=7)
        initial_digest = observation_digest(observation)
        steps: list[TraceStep] = []
        for action in (*((cls.loop_action,) * loop_count), cls.finish_action):
            observation, reward, terminated, truncated, info = environment.step(action)
            steps.append(
                TraceStep(
                    action=action,
                    observation_digest=observation_digest(observation),
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                )
            )
        return EpisodeTrace(
            seed=7,
            initial_observation_digest=initial_digest,
            steps=tuple(steps),
            backend="loop",
        )

    @classmethod
    def _active_observation(cls) -> Observation:
        return Observation(
            phase=Phase.EVENT,
            turn=0,
            player=PlayerView(hp=80, max_hp=80, block=0, energy=0),
            hand=(),
            enemies=(),
            draw_pile=(),
            discard_pile=(),
            exhaust_pile=(),
            legal_actions=(cls.loop_action, cls.finish_action),
            act=1,
            floor=1,
        )


class ForcedChoiceEnvironment(LoopingEnvironment):
    @classmethod
    def _active_observation(cls) -> Observation:
        return replace(
            LoopingEnvironment._active_observation(),
            legal_actions=(cls.finish_action,),
        )


class RedeterminizedPrefixEnvironment:
    advance_action = Action(
        ActionKind.CHOOSE_OPTION,
        source_id="advance",
        option_type="advance",
    )
    finish_action = Action(
        ActionKind.CHOOSE_OPTION,
        source_id="finish",
        option_type="finish",
    )
    wait_action = Action(
        ActionKind.CHOOSE_OPTION,
        source_id="wait",
        option_type="wait",
    )

    def __init__(self) -> None:
        self._future_seed = -1
        self._observation = self._act_one_observation()

    @property
    def observation(self) -> Observation:
        return self._observation

    def reset(self, seed: int | None = None) -> tuple[Observation, dict[str, object]]:
        self._future_seed = -1
        self._observation = self._act_one_observation()
        return self._observation, {"backend": "redeterminized", "seed": seed}

    def step(
        self,
        action: int | Action,
    ) -> tuple[Observation, float, bool, bool, dict[str, object]]:
        resolved = self._observation.legal_actions[action] if isinstance(action, int) else action
        if resolved == self.advance_action:
            self._observation = self._act_two_observation()
            return self._observation, 0.0, False, False, {
                "backend": "redeterminized",
                "future_seed": self._future_seed,
            }
        if resolved != self.finish_action:
            raise ValueError("unexpected redeterminized-prefix test action")
        self._observation = replace(
            self._observation,
            phase=Phase.TERMINAL,
            legal_actions=(),
            act=3,
        )
        return self._observation, float(self._future_seed), True, False, {
            "backend": "redeterminized",
            "future_seed": self._future_seed,
        }

    def clone(self) -> RedeterminizedPrefixEnvironment:
        cloned = RedeterminizedPrefixEnvironment()
        cloned._future_seed = self._future_seed
        cloned._observation = self._observation
        return cloned

    def redeterminized_clone(
        self,
        search_seed: int,
        known_top: tuple[str, ...] = (),
        known_bottom: tuple[str, ...] = (),
    ) -> RedeterminizedPrefixEnvironment:
        cloned = self.clone()
        cloned._future_seed = search_seed
        return cloned

    @classmethod
    def prefix_trace(cls) -> EpisodeTrace:
        environment = cls()
        observation, _ = environment.reset(seed=7)
        initial_digest = observation_digest(observation)
        observation, reward, terminated, truncated, info = environment.step(
            cls.advance_action
        )
        return EpisodeTrace(
            seed=7,
            initial_observation_digest=initial_digest,
            steps=(
                TraceStep(
                    action=cls.advance_action,
                    observation_digest=observation_digest(observation),
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                ),
            ),
            backend="redeterminized",
        )

    @classmethod
    def _act_one_observation(cls) -> Observation:
        return Observation(
            phase=Phase.EVENT,
            turn=0,
            player=PlayerView(hp=80, max_hp=80, block=0, energy=0),
            hand=(),
            enemies=(),
            draw_pile=(),
            discard_pile=(),
            exhaust_pile=(),
            legal_actions=(cls.advance_action,),
            act=1,
            floor=16,
        )

    @classmethod
    def _act_two_observation(cls) -> Observation:
        return replace(
            cls._act_one_observation(),
            legal_actions=(cls.finish_action, cls.wait_action),
            act=2,
            floor=17,
        )


class TrainingInfrastructureTests(unittest.TestCase):
    def test_runtime_manifest_records_source_dependencies_and_hardware(self) -> None:
        project_root = Path(__file__).resolve().parents[1]

        manifest = build_runtime_manifest(project_root)

        self.assertEqual(len(manifest["source_sha256"]), 64)
        self.assertGreater(manifest["source_file_count"], 0)
        self.assertTrue(manifest["runtime_artifacts"])
        self.assertTrue(
            any(
                path.endswith((".pyd", ".so"))
                for path in manifest["runtime_artifacts"]
            )
        )
        self.assertIn("torch", manifest["dependencies"])
        self.assertIn("cuda_available", manifest["hardware"])

    def test_seed_split_is_disjoint_and_explicit(self) -> None:
        split = SeedSplit(0, 100, 1000, 25)

        self.assertEqual(split.training_seeds[0], 0)
        self.assertEqual(split.training_seeds[-1], 99)
        self.assertEqual(split.evaluation_seeds[0], 1000)
        self.assertFalse(set(split.training_seeds).intersection(split.evaluation_seeds))

        with self.assertRaises(ValueError):
            SeedSplit(0, 100, 50, 100)

    def test_object_encoder_scores_exact_dynamic_candidates(self) -> None:
        environment = StsEnv(ToyCombatBackend())
        observation, _ = environment.reset(seed=3)
        encoder = ObjectFeatureEncoder()

        features = encoder.encode_candidates(observation)

        self.assertEqual(features.shape[0], len(observation.legal_actions))
        self.assertEqual(features.shape[1], encoder.dimension)
        self.assertTrue(torch.isfinite(features).all())

    def test_run_encoder_uses_public_map_and_action_costs(self) -> None:
        environment = StsEnv(ToyCombatBackend())
        observation, _ = environment.reset(seed=3)
        observation = replace(
            observation,
            legal_actions=(
                Action(
                    ActionKind.BUY,
                    source_id="carnage",
                    option_type="card",
                    gold_cost=70,
                    label="Carnage",
                ),
            ),
            map_nodes=(
                MapNodeView(1, 0, "M", ((2, 1),)),
                MapNodeView(2, 1, "E", ((3, 2),), burning_elite=True),
            ),
            act_boss="theguardian",
            potion_capacity=3,
        )
        encoder = RunFeatureEncoder()

        state = encoder.encode_state(observation)
        actions = encoder.encode_actions(observation)

        self.assertEqual(state.shape, (encoder.state_dimension,))
        self.assertEqual(actions.shape, (1, encoder.action_dimension))
        self.assertTrue(torch.isfinite(state).all())
        self.assertTrue(torch.isfinite(actions).all())

    def test_recurrent_ppo_collects_and_updates_sequences(self) -> None:
        config = RecurrentPPOConfig(
            recurrent_size=32,
            state_embedding_size=32,
            action_embedding_size=32,
            update_epochs=2,
            minibatch_environments=2,
        )
        trainer = RecurrentPPOTrainer(config=config, seed=17)
        collector = RecurrentRolloutCollector(
            environment_factory=lambda: StsEnv(ToyCombatBackend()),
            trainer=trainer,
            num_environments=4,
            seeds=tuple(range(1000, 1100)),
        )

        rollout = collector.collect(steps_per_environment=8)
        before = {
            name: parameter.detach().clone()
            for name, parameter in trainer.network.named_parameters()
        }
        metrics = trainer.update(rollout)

        self.assertEqual(rollout.states.shape[:2], (8, 4))
        self.assertEqual(rollout.actions.shape[:2], (8, 4))
        self.assertTrue(torch.isfinite(torch.tensor(list(metrics.values()))).all())
        self.assertTrue(
            any(
                not torch.equal(before[name], parameter)
                for name, parameter in trainer.network.named_parameters()
            )
        )

    def test_combat_delegation_masks_policy_updates(self) -> None:
        trainer = RecurrentPPOTrainer(
            config=RecurrentPPOConfig(
                recurrent_size=16,
                state_embedding_size=16,
                action_embedding_size=16,
                update_epochs=1,
                minibatch_environments=2,
            ),
            seed=19,
        )
        heuristic = HeuristicPolicy()
        collector = RecurrentRolloutCollector(
            environment_factory=lambda: StsEnv(ToyCombatBackend()),
            trainer=trainer,
            num_environments=2,
            seeds=tuple(range(2000, 2100)),
            combat_selector=lambda environment: heuristic(environment.observation),
        )

        rollout = collector.collect(steps_per_environment=4)
        metrics = trainer.update(rollout)

        self.assertTrue(torch.equal(rollout.policy_weights, torch.zeros_like(rollout.policy_weights)))
        self.assertTrue(torch.isfinite(torch.tensor(list(metrics.values()))).all())

    def test_curriculum_reward_is_kept_outside_raw_environment(self) -> None:
        curriculum = CurriculumEnvironment(
            environment_factory=lambda: StsEnv(
                LightspeedBackend(neow_history="skipped")
            ),
            spec=CurriculumSpec(
                "first_floor",
                target_floor=1,
                completion_reward=1.0,
                potential_scale=0.0,
                progress_reward_per_floor=0.0,
            ),
        )
        observation, _ = curriculum.reset(seed=1)

        observation, reward, terminated, truncated, info = curriculum.step(
            observation.legal_actions[0]
        )

        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(observation.floor, 1)
        self.assertEqual(info["raw_reward"], 0.0)
        self.assertEqual(reward, 1.0)
        self.assertTrue(info["curriculum_completed"])

        trainer = RecurrentPPOTrainer(
            config=RecurrentPPOConfig(
                recurrent_size=16,
                state_embedding_size=16,
                action_embedding_size=16,
                update_epochs=1,
                minibatch_environments=1,
            ),
            seed=31,
        )
        collector = RecurrentRolloutCollector(
            environment_factory=lambda: CurriculumEnvironment(
                environment_factory=lambda: StsEnv(
                    LightspeedBackend(neow_history="skipped")
                ),
                spec=CurriculumSpec(
                    "first_floor",
                    target_floor=1,
                    completion_reward=1.0,
                    potential_scale=0.0,
                    progress_reward_per_floor=0.0,
                ),
            ),
            trainer=trainer,
            num_environments=1,
            seeds=(1, 2),
        )
        rollout = collector.collect(steps_per_environment=1)
        self.assertEqual(len(rollout.completed_traces), 1)
        replay_trace(
            StsEnv(LightspeedBackend(neow_history="skipped")),
            rollout.completed_traces[0],
        )

    def test_curriculum_loop_limit_survives_trace_recovery(self) -> None:
        spec = CurriculumSpec(
            "loop",
            completion_reward=0.0,
            potential_scale=0.0,
            progress_reward_per_floor=0.0,
            max_repeated_decisions=2,
        )
        trace = LoopingEnvironment.trace(loop_count=2).prefix(2)
        curriculum = CurriculumEnvironment(LoopingEnvironment, spec)

        curriculum.replay_recovery_trace(trace)
        _, _, terminated, truncated, info = curriculum.step(LoopingEnvironment.loop_action)

        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertTrue(info["curriculum_loop_detected"])
        self.assertEqual(info["curriculum_repeat_count"], 3)

    def test_self_imitation_erases_noop_cycles(self) -> None:
        trainer = RecurrentPPOTrainer(
            config=RecurrentPPOConfig(
                recurrent_size=16,
                state_embedding_size=16,
                action_embedding_size=16,
                update_epochs=1,
                minibatch_environments=1,
            ),
            seed=41,
        )
        chunks = build_imitation_chunks(
            LoopingEnvironment,
            trainer,
            (LoopingEnvironment.trace(loop_count=6),),
            chunk_length=64,
            burn_in_steps=0,
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(float(chunks[0].supervision_weights.sum()), 1.0)
        supervised_index = int(torch.argmax(chunks[0].supervision_weights))
        self.assertEqual(int(chunks[0].chosen_actions[supervised_index]), 1)

        metrics = train_self_imitation(trainer, chunks, epochs=1, seed=41)
        self.assertEqual(metrics["supervised_steps"], 1.0)
        self.assertEqual(metrics["supervision_weight"], 1.0)
        self.assertTrue(torch.isfinite(torch.tensor(tuple(metrics.values()))).all())

    def test_dagger_collects_teacher_labels_on_student_states(self) -> None:
        trainer = RecurrentPPOTrainer(
            config=RecurrentPPOConfig(
                recurrent_size=16,
                state_embedding_size=16,
                action_embedding_size=16,
                update_epochs=1,
                minibatch_environments=1,
            ),
            seed=43,
        )

        chunks = collect_dagger_chunks(
            LoopingEnvironment,
            trainer,
            lambda observation: LoopingEnvironment.finish_action,
            seeds=(11, 12),
            max_steps=4,
            chunk_length=4,
            burn_in_steps=0,
            phase_weights={Phase.EVENT: 3.0},
        )

        self.assertTrue(chunks)
        for chunk in chunks:
            supervised = chunk.supervision_weights.bool()
            self.assertTrue(supervised.any())
            self.assertTrue(
                torch.equal(
                    chunk.supervision_weights[supervised],
                    torch.full_like(chunk.supervision_weights[supervised], 3.0),
                )
            )
            self.assertTrue(
                torch.equal(
                    chunk.chosen_actions[supervised],
                    torch.ones_like(chunk.chosen_actions[supervised]),
                )
            )

    def test_imitation_excludes_forced_actions(self) -> None:
        trainer = RecurrentPPOTrainer(
            config=RecurrentPPOConfig(
                recurrent_size=16,
                state_embedding_size=16,
                action_embedding_size=16,
                update_epochs=1,
                minibatch_environments=1,
            ),
            seed=47,
        )

        chunks = collect_dagger_chunks(
            ForcedChoiceEnvironment,
            trainer,
            lambda observation: ForcedChoiceEnvironment.finish_action,
            seeds=(13,),
            max_steps=2,
            chunk_length=2,
            burn_in_steps=0,
        )

        self.assertEqual(chunks, ())

    def test_dagger_training_seeds_rotate_inside_training_split(self) -> None:
        config = DaggerConfig(
            interval_updates=25,
            episodes=4,
            training_seed_offset=48,
            round_seed_stride=10,
            stage_rounds={"full_run": 1},
        )

        first = dagger_training_seeds(
            config,
            training_seed_start=100,
            training_seed_count=50,
            update_index=25,
        )
        second = dagger_training_seeds(
            config,
            training_seed_start=100,
            training_seed_count=50,
            update_index=50,
        )
        second_round = dagger_training_seeds(
            config,
            training_seed_start=100,
            training_seed_count=50,
            update_index=25,
            round_index=1,
        )

        self.assertEqual(first, (148, 149, 100, 101))
        self.assertEqual(second, (102, 103, 104, 105))
        self.assertEqual(second_round, (108, 109, 110, 111))
        self.assertTrue(set(first + second).issubset(set(range(100, 150))))
        self.assertEqual(config.rounds_for_stage("act2_clear"), 2)
        self.assertEqual(config.rounds_for_stage("full_run"), 1)

    def test_full_run_frontier_traces_are_ranked_by_public_progress(self) -> None:
        shallow = LoopingEnvironment.trace(loop_count=1)
        shallow = EpisodeTrace(
            seed=shallow.seed,
            initial_observation_digest=shallow.initial_observation_digest,
            steps=tuple(
                replace(step, info={**step.info, "floor": 17}) for step in shallow.steps
            ),
            backend=shallow.backend,
            metadata=shallow.metadata,
        )
        deep = LoopingEnvironment.trace(loop_count=2)
        deep = EpisodeTrace(
            seed=deep.seed,
            initial_observation_digest=deep.initial_observation_digest,
            steps=tuple(
                replace(step, info={**step.info, "floor": 28}) for step in deep.steps
            ),
            backend=deep.backend,
            metadata=deep.metadata,
        )

        self.assertTrue(
            is_self_imitation_candidate(
                "full_run", target_act=None, final_act=2, won=False
            )
        )
        self.assertFalse(
            is_self_imitation_candidate(
                "full_run", target_act=None, final_act=1, won=False
            )
        )
        self.assertGreater(imitation_trace_progress(deep), imitation_trace_progress(shallow))
        self.assertEqual(rank_imitation_traces((shallow, deep), 1), (deep,))

    def test_frontier_weighting_repeats_only_deepest_trace(self) -> None:
        shallow = LoopingEnvironment.trace(loop_count=1)
        shallow = EpisodeTrace(
            seed=shallow.seed,
            initial_observation_digest=shallow.initial_observation_digest,
            steps=tuple(
                replace(step, info={**step.info, "floor": 17}) for step in shallow.steps
            ),
            backend=shallow.backend,
            metadata=shallow.metadata,
        )
        deep = LoopingEnvironment.trace(loop_count=2)
        deep = EpisodeTrace(
            seed=deep.seed,
            initial_observation_digest=deep.initial_observation_digest,
            steps=tuple(
                replace(step, info={**step.info, "floor": 46}) for step in deep.steps
            ),
            backend=deep.backend,
            metadata=deep.metadata,
        )

        selected = select_weighted_frontier_traces(
            (shallow, deep),
            limit=2,
            frontier_trace_repeats=4,
        )

        self.assertEqual(selected, (deep, deep, deep, deep, shallow))
        self.assertEqual(selected.count(deep), 4)
        self.assertEqual(selected.count(shallow), 1)

    def test_curriculum_scheduler_and_prefix_corpus_are_recoverable(self) -> None:
        scheduler = CurriculumScheduler(
            stages=(
                CurriculumSpec("floor", target_floor=1),
                CurriculumSpec("full", completion_reward=0.0, potential_scale=0.0),
            ),
            promotion_threshold=0.5,
        )
        self.assertFalse(scheduler.observe_validation(0.49))
        self.assertTrue(scheduler.observe_validation(0.5))
        restored_scheduler = CurriculumScheduler.from_state_dict(scheduler.state_dict())
        self.assertEqual(restored_scheduler.current.name, "full")

        trace = record_episode(
            StsEnv(ToyCombatBackend()),
            seed=23,
            policy=HeuristicPolicy(),
            max_steps=200,
        )
        corpus = PrefixCorpus((trace.prefix(2),), target_act=2)
        with tempfile.TemporaryDirectory() as directory:
            corpus.write(directory)
            restored_corpus = PrefixCorpus.read(directory)
        self.assertEqual(restored_corpus, corpus)

    def test_prefix_curriculum_redeterminizes_and_recovers_future_rng(self) -> None:
        prefix = RedeterminizedPrefixEnvironment.prefix_trace()
        factory = lambda: CurriculumEnvironment(
            RedeterminizedPrefixEnvironment,
            CurriculumSpec("act2", start_act=2, target_act=3, use_prefix_starts=True),
            PrefixCorpus((prefix,), target_act=2),
        )
        environment = factory()

        observation, reset_info = environment.reset(seed=11)
        self.assertEqual(observation.act, 2)
        self.assertEqual(reset_info["curriculum_redeterminization_seed"], 11)
        next_observation, reward, terminated, truncated, step_info = environment.step(
            RedeterminizedPrefixEnvironment.finish_action
        )
        self.assertEqual(step_info["future_seed"], 11)

        trace = EpisodeTrace(
            seed=int(reset_info["seed"]),
            initial_observation_digest=observation_digest(observation),
            steps=(
                TraceStep(
                    action=RedeterminizedPrefixEnvironment.finish_action,
                    observation_digest=observation_digest(next_observation),
                    reward=float(step_info["raw_reward"]),
                    terminated=terminated,
                    truncated=truncated,
                    info={
                        "backend": step_info["backend"],
                        "future_seed": step_info["future_seed"],
                    },
                ),
            ),
            backend=str(reset_info["backend"]),
            metadata=dict(reset_info),
        )
        restored = factory()

        restored_observation = restored.replay_recovery_trace(trace)

        self.assertEqual(restored_observation, next_observation)

        nested = EpisodeTrace(
            seed=trace.seed,
            initial_observation_digest=observation_digest(next_observation),
            steps=(),
            backend=trace.backend,
            metadata={
                "curriculum_selection_seed": 19,
                "curriculum_redeterminization_seed": 19,
                "curriculum_source_trace": trace.to_dict(),
            },
        )
        nested_environment = factory()
        nested_observation = nested_environment.replay_recovery_trace(nested)
        self.assertEqual(nested_observation, next_observation)

        trainer = RecurrentPPOTrainer(
            config=RecurrentPPOConfig(
                recurrent_size=16,
                state_embedding_size=16,
                action_embedding_size=16,
                update_epochs=1,
                minibatch_environments=1,
            ),
            seed=53,
        )
        chunks = build_imitation_chunks(
            RedeterminizedPrefixEnvironment,
            trainer,
            (trace,),
            recovery_environment_factory=factory,
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(float(chunks[0].supervision_weights.sum()), 1.0)

        independent = factory()
        independent.reset(seed=12)
        _, _, _, _, independent_info = independent.step(
            RedeterminizedPrefixEnvironment.finish_action
        )
        self.assertEqual(independent_info["future_seed"], 12)

    def test_nonempty_curriculum_worker_prefix_restores_raw_trace(self) -> None:
        trainer = RecurrentPPOTrainer(
            config=RecurrentPPOConfig(
                recurrent_size=16,
                state_embedding_size=16,
                action_embedding_size=16,
                update_epochs=1,
                minibatch_environments=1,
            ),
            seed=37,
        )
        factory = lambda: CurriculumEnvironment(
            environment_factory=lambda: StsEnv(
                LightspeedBackend(neow_history="skipped")
            ),
            spec=CurriculumSpec("floor6", target_floor=6, max_episode_steps=100),
        )
        collector = RecurrentRolloutCollector(
            environment_factory=factory,
            trainer=trainer,
            num_environments=1,
            seeds=(10, 11),
        )
        collector.collect(steps_per_environment=1)

        restored = RecurrentRolloutCollector.from_state_dict(
            factory,
            trainer,
            collector.state_dict(),
        )

        self.assertEqual(restored.observations, collector.observations)

    def test_spawn_workers_collect_recurrent_rollouts(self) -> None:
        trainer = RecurrentPPOTrainer(
            config=RecurrentPPOConfig(
                recurrent_size=16,
                state_embedding_size=16,
                action_embedding_size=16,
                update_epochs=1,
                minibatch_environments=2,
            ),
            seed=23,
        )
        pool = SubprocessVectorEnvironment(
            LightspeedEnvironmentFactory(neow_history="skipped"),
            num_environments=2,
        )
        collector = MultiprocessRecurrentRolloutCollector(
            pool,
            trainer,
            seeds=tuple(range(3000, 3100)),
        )
        try:
            rollout = collector.collect(steps_per_environment=3)
        finally:
            collector.close()

        self.assertEqual(rollout.states.shape[:2], (3, 2))
        self.assertTrue(rollout.action_masks.any(dim=-1).all())

    def test_spawn_workers_are_reproducible_for_same_seed_stream(self) -> None:
        def collect_signature() -> tuple[torch.Tensor, ...]:
            trainer = RecurrentPPOTrainer(
                config=RecurrentPPOConfig(
                    recurrent_size=16,
                    state_embedding_size=16,
                    action_embedding_size=16,
                    update_epochs=1,
                    minibatch_environments=2,
                ),
                seed=67,
            )
            pool = SubprocessVectorEnvironment(
                LightspeedEnvironmentFactory(neow_history="skipped"),
                num_environments=2,
            )
            collector = MultiprocessRecurrentRolloutCollector(
                pool,
                trainer,
                seeds=tuple(range(4000, 4100)),
            )
            try:
                rollout = collector.collect(steps_per_environment=4)
                return (
                    rollout.chosen_actions.clone(),
                    rollout.rewards.clone(),
                    rollout.dones.clone(),
                )
            finally:
                collector.close()

        first = collect_signature()
        second = collect_signature()
        for first_tensor, second_tensor in zip(first, second, strict=True):
            self.assertTrue(torch.equal(first_tensor, second_tensor))

    def test_m6_checkpoint_restores_worker_prefixes_and_recurrent_state(self) -> None:
        trainer = RecurrentPPOTrainer(
            config=RecurrentPPOConfig(
                recurrent_size=16,
                state_embedding_size=16,
                action_embedding_size=16,
                update_epochs=1,
                minibatch_environments=2,
            ),
            seed=29,
        )
        collector = RecurrentRolloutCollector(
            environment_factory=lambda: StsEnv(ToyCombatBackend()),
            trainer=trainer,
            num_environments=2,
            seeds=tuple(range(4000, 4100)),
        )
        rollout = collector.collect(steps_per_environment=5)
        trainer.update(rollout)
        state = collector.state_dict()
        scheduler = CurriculumScheduler(
            stages=(CurriculumSpec("full", completion_reward=0.0, potential_scale=0.0),)
        )
        config = M6TrainingConfig(
            run_seed=29,
            num_environments=2,
            rollout_steps=5,
            total_updates=1,
            checkpoint_interval=1,
            validation_interval=1,
            training_seed_count=100,
            validation_seed_count=10,
            device="cpu",
        )
        parameter_ema = ParameterEMA(trainer.network, decay=0.9)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "m6.pt"
            save_m6_checkpoint(
                path,
                trainer=trainer,
                collector_state=state,
                scheduler=scheduler,
                config=config,
                update_index=1,
                metrics=({"loss": 1.0},),
                manifest={"fixture": True},
                parameter_ema_state=parameter_ema.state_dict(),
            )
            loaded = load_m6_checkpoint(path)

        restored_collector = RecurrentRolloutCollector.from_state_dict(
            lambda: StsEnv(ToyCombatBackend()),
            loaded.trainer,
            loaded.collector_state,
        )
        self.assertEqual(restored_collector.observations, collector.observations)
        original_policy = trainer.sample_actions(
            collector.observations,
            state["hidden"],
            deterministic=True,
        )
        restored_policy = loaded.trainer.sample_actions(
            restored_collector.observations,
            loaded.collector_state["hidden"],
            deterministic=True,
        )
        self.assertTrue(torch.equal(original_policy.logits, restored_policy.logits))
        self.assertTrue(torch.equal(original_policy.values, restored_policy.values))
        self.assertTrue(torch.equal(original_policy.next_hidden, restored_policy.next_hidden))
        self.assertEqual(loaded.metrics, ({"loss": 1.0},))
        self.assertEqual(loaded.manifest, {"fixture": True})
        self.assertEqual(loaded.parameter_ema_state["decay"], 0.9)
        self.assertEqual(
            loaded.parameter_ema_state["averaged"].keys(),
            trainer.network.state_dict().keys(),
        )

    def test_parameter_ema_updates_and_restores_online_parameters(self) -> None:
        module = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            module.weight.fill_(1.0)
        parameter_ema = ParameterEMA(module, decay=0.5)
        with torch.no_grad():
            module.weight.fill_(3.0)

        parameter_ema.update(module)
        with parameter_ema.use_averaged_parameters(module):
            self.assertEqual(float(module.weight.item()), 2.0)

        self.assertEqual(float(module.weight.item()), 3.0)
        restored = ParameterEMA(module, decay=0.5)
        restored.load_state_dict(module, parameter_ema.state_dict())
        with restored.use_averaged_parameters(module):
            self.assertEqual(float(module.weight.item()), 2.0)
        self.assertEqual(
            ParameterEMAConfig(stages=["full_run"]).stages,
            ("full_run",),
        )

    def test_m6_final_seed_range_cannot_be_crossed_implicitly(self) -> None:
        with self.assertRaises(ValueError):
            validate_m6_evaluation_seed_range(1_999_999, 2, final=False)
        self.assertEqual(
            validate_m6_evaluation_seed_range(2_000_000, 1_024, final=True),
            (2_000_000, 2_001_023),
        )
        with self.assertRaises(ValueError):
            validate_m6_evaluation_seed_range(2_000_000, 1_023, final=True)
        with self.assertRaises(ValueError):
            M6TrainingConfig(
                run_seed=17,
                training_seed_start=1_999_999,
                training_seed_count=2,
                validation_seed_start=1_100_000,
                validation_seed_count=64,
            )
        self.assertGreater(
            m6_validation_selection_key(
                "full_run",
                {"win_rate": 1 / 64, "completion_rate": 1 / 64, "mean_floor": 10.0},
            ),
            m6_validation_selection_key(
                "full_run",
                {"win_rate": 0.0, "completion_rate": 0.0, "mean_floor": 50.0},
            ),
        )

    def test_final_freeze_manifest_verifies_all_ema_checkpoints(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        source_sha256 = build_runtime_manifest(project_root)["source_sha256"]
        script_path = project_root / "scripts" / "evaluate-m6.py"
        specification = importlib.util.spec_from_file_location(
            "evaluate_m6_for_test",
            script_path,
        )
        assert specification is not None and specification.loader is not None
        evaluate_m6 = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(evaluate_m6)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = {}
            paths = {}
            for run_seed in (17, 29, 43):
                trainer = RecurrentPPOTrainer(
                    config=RecurrentPPOConfig(
                        recurrent_size=16,
                        state_embedding_size=16,
                        action_embedding_size=16,
                        update_epochs=1,
                        minibatch_environments=1,
                    ),
                    seed=run_seed,
                )
                scheduler = CurriculumScheduler(
                    stages=(
                        CurriculumSpec(
                            "full_run",
                            completion_reward=0.0,
                            potential_scale=0.0,
                        ),
                    )
                )
                checkpoint_path = root / f"seed-{run_seed}.pt"
                save_m6_checkpoint(
                    checkpoint_path,
                    trainer=trainer,
                    collector_state={},
                    scheduler=scheduler,
                    config=M6TrainingConfig(
                        run_seed=run_seed,
                        total_updates=1,
                        checkpoint_interval=1,
                        validation_interval=1,
                        training_seed_count=100,
                        validation_seed_count=64,
                        device="cpu",
                    ),
                    update_index=1,
                    manifest={
                        "source_sha256": source_sha256,
                        "evaluation_only": True,
                        "parameter_source": "ema",
                    },
                )
                paths[run_seed] = checkpoint_path
                entries[str(run_seed)] = {
                    "path": str(checkpoint_path.resolve()),
                    "sha256": evaluate_m6.sha256_file(checkpoint_path),
                    "update": 1,
                    "run_seed": run_seed,
                    "source_sha256": source_sha256,
                    "stage": "full_run",
                    "evaluation_only": True,
                    "parameter_source": "ema",
                }
            manifest_path = root / "freeze.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "final_seed_range": [2_000_000, 2_001_023],
                        "source_sha256": source_sha256,
                        "checkpoints": entries,
                    }
                ),
                encoding="utf-8",
            )

            verification = evaluate_m6.verify_freeze_manifest(
                manifest_path,
                paths[17],
            )

        self.assertEqual(verification["checkpoint_run_seed"], 17)
        self.assertEqual(verification["source_sha256"], source_sha256)

    def test_replay_jsonl_round_trip_is_stable(self) -> None:
        collector = SynchronousVectorCollector(
            environment_factory=lambda: StsEnv(ToyCombatBackend()),
            num_environments=2,
            seeds=tuple(range(20)),
        )
        batch = collector.collect(12, RandomPolicy(seed=5))
        replay = ReplayBuffer(capacity=32, seed=7)
        for transition in batch.transitions:
            replay.add(transition)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.jsonl"
            replay.write_jsonl(path)
            restored = ReplayBuffer.read_jsonl(path, seed=7)

        self.assertEqual(len(restored), len(replay))
        self.assertEqual(restored.state_dict()["items"], replay.state_dict()["items"])

    def test_vector_collector_uses_explicit_seed_stream(self) -> None:
        collector = SynchronousVectorCollector(
            environment_factory=lambda: StsEnv(ToyCombatBackend()),
            num_environments=4,
            seeds=tuple(range(100, 200)),
        )

        batch = collector.collect(400, RandomPolicy(seed=11))

        self.assertEqual(len(batch.transitions), 400)
        self.assertTrue(batch.completed_episodes)
        self.assertTrue(all(100 <= episode.seed < 200 for episode in batch.completed_episodes))

    def test_checkpoint_restores_q_values_actions_optimizer_and_replay(self) -> None:
        config = CandidateQConfig(
            hidden_sizes=(32, 16),
            batch_size=8,
            replay_capacity=128,
            warmup_steps=8,
            target_update_interval=2,
            epsilon_decay_steps=100,
        )
        trainer = CandidateQTrainer(config=config, seed=11)
        collector = SynchronousVectorCollector(
            environment_factory=lambda: StsEnv(ToyCombatBackend()),
            num_environments=2,
            seeds=tuple(range(200)),
        )
        batch = collector.collect(
            48,
            lambda observation, _: trainer.select_action(observation, explore=True),
        )
        metrics = None
        for transition in batch.transitions:
            trainer.observe(transition)
            metrics = trainer.train_step()

        self.assertIsNotNone(metrics)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))
        self.assertLessEqual(metrics["gradient_norm"], config.gradient_clip_norm * 10)
        probe, _ = StsEnv(ToyCombatBackend()).reset(seed=999)
        before_values = trainer.q_values(probe).cpu()
        before_action = trainer.greedy_action(probe)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            trainer.save_checkpoint(path, metadata={"fixture": True})
            restored, metadata = CandidateQTrainer.load_checkpoint(path)

        self.assertEqual(metadata, {"fixture": True})
        self.assertEqual(restored.environment_steps, trainer.environment_steps)
        self.assertEqual(restored.gradient_steps, trainer.gradient_steps)
        self.assertEqual(len(restored.replay), len(trainer.replay))
        self.assertTrue(torch.equal(restored.q_values(probe).cpu(), before_values))
        self.assertEqual(restored.greedy_action(probe), before_action)

    def test_heuristic_efficiency_score_exceeds_random_on_unseen_seeds(self) -> None:
        seeds = tuple(range(10_000, 10_128))
        factory = lambda: StsEnv(ToyCombatBackend())

        random_summary = evaluate_policy(factory, RandomPolicy(seed=13), seeds, max_steps=500)
        heuristic_summary = evaluate_policy(factory, HeuristicPolicy(), seeds, max_steps=500)

        self.assertGreater(heuristic_summary.mean_score, random_summary.mean_score)
        self.assertLess(heuristic_summary.mean_length, random_summary.mean_length)

    def test_full_run_evaluation_reports_all_seeds_and_intervals(self) -> None:
        seeds = tuple(range(5000, 5016))
        summary = evaluate_full_runs(
            lambda: StsEnv(ToyCombatBackend()),
            lambda policy_seed, _: RandomPolicy(policy_seed),
            seeds,
            policy_seed=41,
            max_steps=500,
            bootstrap_samples=200,
        )

        self.assertEqual(len(summary.episodes), len(seeds))
        self.assertEqual({episode.seed for episode in summary.episodes}, set(seeds))
        self.assertLessEqual(summary.win_rate_ci95[0], summary.win_rate)
        self.assertGreaterEqual(summary.win_rate_ci95[1], summary.win_rate)
        self.assertLessEqual(summary.mean_floor_ci95[0], summary.mean_floor)
        self.assertGreaterEqual(summary.mean_floor_ci95[1], summary.mean_floor)
        self.assertEqual(summary.errors, 0)
        self.assertEqual(summary.crashes, 0)
        self.assertEqual(summary.illegal_actions, 0)
        self.assertEqual(summary.recovery_failures, 0)
        self.assertEqual(summary.truncations, 0)

    def test_m6_reporting_aggregates_runs_and_paired_differences(self) -> None:
        def evaluation(run_seed: int, method: str, floor_delta: int) -> dict[str, object]:
            episodes = [
                {
                    "seed": seed,
                    "won": False,
                    "final_floor": 10 + seed - 7000 + floor_delta,
                    "final_hp": 20 + floor_delta,
                    "proxy_score": 1.0 + floor_delta,
                    "decisions": 50 + floor_delta,
                    "simulator_calls": 0,
                    "wall_seconds": 1.0,
                }
                for seed in range(7000, 7004)
            ]
            return {
                "method": method,
                "run_seed": run_seed,
                "policy_seed": run_seed,
                "summary": {
                    "episodes": episodes,
                    "win_rate": 0.0,
                    "act1_clear_rate": 0.5,
                    "act2_clear_rate": 0.0,
                    "act3_clear_rate": 0.0,
                    "mean_floor": 11.5 + floor_delta,
                    "median_floor": 11.5 + floor_delta,
                    "mean_final_hp": 20.0 + floor_delta,
                    "mean_proxy_score": 1.0 + floor_delta,
                    "mean_decisions": 50.0 + floor_delta,
                    "total_simulator_calls": 0,
                    "total_wall_seconds": 4.0,
                    "errors": 0,
                    "crashes": 0,
                    "illegal_actions": 0,
                    "recovery_failures": 0,
                    "truncations": 0,
                    "timeouts": 0,
                    "cycles": 0,
                },
            }

        report = summarize_m6_evaluations(
            (
                evaluation(17, "heuristic", 0),
                evaluation(17, "learned", 2),
                evaluation(29, "heuristic", 0),
                evaluation(29, "learned", 4),
            ),
            bootstrap_samples=100,
        )

        paired = report["runs"]["17"]["paired_comparisons"][
            "learned_minus_heuristic"
        ]
        self.assertEqual(paired["sample_count"], 4)
        self.assertEqual(paired["metrics"]["final_floor"]["mean_difference"], 2.0)
        learned_floor = report["aggregate"]["learned"]["metrics"]["mean_floor"]
        self.assertEqual(learned_floor["values"], [13.5, 15.5])
        self.assertGreater(learned_floor["standard_deviation"], 0.0)


if __name__ == "__main__":
    unittest.main()
