# Experiment Audit Report

**Date**: 2026-07-17  
**Auditor**: Independent external agent, xhigh reasoning, read-only  
**Project**: STS Agent M6

## Overall Verdict: WARN

## Integrity Status: warn

未发现伪造真值、基于模型自身输出的分数归一化、幽灵结果文件或已报告数字不匹配。当前风险来自协议完整性：所有现有整局结果都属于验证集诊断，不具备最终 M6 结论资格；最终种子和 checkpoint 冻结保护仍存在可绕过边界。

## Checks

### A. Ground Truth Provenance: PASS

- 整局结果来自 `StsEnv(LightspeedBackend())` 的真实模拟终局；胜利由环境回报和终止状态共同判定。
- `proxy_score` 明确标记为代理指标，由环境回报、楼层和 HP 构成，不使用模型输出作为真值。
- 配对比较中的 reference 是另一策略在相同环境 seed 上的结果，不被描述为 ground truth。
- 证据：`scripts/evaluate-m6.py:144`、`src/sts_env/training/full_run_evaluation.py:108`、`src/sts_env/training/full_run_evaluation.py:125`、`src/sts_env/training/full_run_evaluation.py:130`。

### B. Score Normalization: PASS

- 未发现用模型预测的最大值、最小值或均值归一化指标。
- 比率仅除以 episode 数；HP 比例使用环境给出的最大 HP。
- 审计员重新计算了 7 组共 7168 条 episode，汇总数字全部一致。
- 证据：`src/sts_env/training/full_run_evaluation.py:130`、`src/sts_env/training/full_run_evaluation.py:152`、`src/sts_env/training/m6_reporting.py:138`。

### C. Result Existence: WARN

- 所列结果文件均存在且可解析；DAgger、update 900 对 750、25% 插值的数字和置信区间均与源文件一致。
- `docs/M6_TRAINING_AUDIT_LOG.md` 中旧的 update 700 `1/64` Act 2 clear 叙述无法由当时 metrics 的字段直接复核，应删除或降级为不可审计诊断。
- DAgger 轮数选择来自已因 Runic Dome 泄漏失去正式资格的旧 pilot，必须在干净 checkpoint 上重做。
- 当前 lightspeed 结果的 `game_score` 为 `null`；协议应明确该后端不可用，而不能暗示已报告。
- 证据：`docs/M6_TRAINING_AUDIT_LOG.md:36`、`experiments/m6_dagger_round_comparison_u1100.json:2`、`docs/M6_IMPLEMENTATION_AND_EVALUATION_PLAN.md:107`。

### D. Dead Code Detection: PASS

- full-run 汇总、Wilson 区间、bootstrap、配对差值和多 run 汇总均被实际调用并进入结果文件。
- 未发现死指标函数；game-score 缺失属于实现/后端能力缺口。
- 证据：`src/sts_env/training/full_run_evaluation.py:157`、`src/sts_env/training/m6_reporting.py:86`、`tests/test_training.py:1066`。

### E. Scope Assessment: WARN

- 当前范围仅为 Ironclad、Ascension 0、本地 `sts_lightspeed`，且正式整局诊断只覆盖 run seed 17。
- 尚缺 run seed 29/43、最终未见 seed，以及 random、heuristic、heuristic-search、pure learned、learned-search 的完整基线套件。
- 现有文档总体正确地把这些结果称为诊断，没有发现“全面”“鲁棒”等超范围结论。

### F. Evaluation Type: simulation_only

- `evaluate-m6.py`、DAgger 验证和配对汇总均属于 `simulation_only`。
- DAgger 对启发式教师标签的 imitation accuracy 属于明确标注的 `synthetic_proxy` 训练指标。
- 当前没有 human evaluation 或 dataset-provided real ground truth。

## Integrity Controls

- **Final-test protection — WARN**：旧 CLI 只在 `seed_start >= 2000000` 时要求 `--final`，起点低于该值但跨入最终区间的范围可绕过保护。
- **Checkpoint freeze — WARN**：旧冻结流程校验 checkpoint 哈希，但未强制 EMA evaluation checkpoint、统一源码哈希、清单条目语义或在评估结果中记录冻结清单摘要。
- **Method labeling — WARN**：新 `learned-heuristic` 结果正确记录 `combat_policy: heuristic`；旧 `learned` 诊断缺少该字段且不能代表训练策略。
- **Validation reuse — WARN**：DAgger、checkpoint、插值与 EMA 设计复用了验证 seed，允许用于模型选择，但选中结果存在选择偏差，必须由未触碰最终集确认。

## Action Items

- 修复所有与最终 seed 区间相交的跨界保护，并让训练/验证配置拒绝最终保留区间。
- 强化冻结脚本：只接受 EMA evaluation checkpoint，要求三个 run 使用同一非空源码哈希，并在评估产物记录冻结清单 SHA-256。
- 用干净 checkpoint 重做 full-run DAgger 一轮/两轮对照。
- 将 update 700 Act 2 clear 叙述改为不可审计诊断，或补充独立逐局分类结果。
- 明确本地 lightspeed 不提供 game score；只在后端给出时汇总该字段。
- 完成三个正式 run、冻结、最终未见 seed 和五类基线后才能形成最终 M6 结论。

## Claim Impact

- 当前诊断中“前沿加权产生首个验证胜利”：**supported，但仅限验证诊断**。
- “25% 插值避免显著主体退化并保留胜利尾部”：**supported，但存在验证集选择偏差**。
- “M6 已完成或具备最终性能结论”：**unsupported**。

