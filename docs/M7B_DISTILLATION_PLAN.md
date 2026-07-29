# M7-B 非战斗策略蒸馏计划

## 1. 为什么不直接扩大 M7

M7 pilot 已证明动态 phase 平衡在机制上生效，但没有证明策略质量提升：

| 比较 | 平均终止楼层差 | 95% CI | Act 1 差 |
|---|---:|---:|---:|
| control - M6 initial | +0.1172 | [-0.5859, 0.8281] | +0.0156 |
| balanced - M6 initial | +0.3555 | [-0.3281, 1.0293] | +0.0137 |
| control - heuristic | -3.8613 | [-4.6523, -3.0605] | -0.1250 |
| balanced - heuristic | -3.6230 | [-4.4414, -2.7715] | -0.1270 |

因此不能把 balanced pilot 的微小点估计提升解释成有效进展，也不应直接投入三种子正式训练。当前更基础的问题是：网络是否能稳定学会已有启发式教师的非战斗决策。

M7-B 只检验这个问题。它不是 M7 正式训练，也不访问最终盲测 seed。

## 2. 可证伪假设

M7-B 同时检验三个假设：

1. M7 中每轮临时生成、随后丢弃的监督数据不够稳定；持久语料和多轮复用可提高样本效率。
2. PPO、价值损失和探索正则与稀疏的非战斗监督相互干扰；移除这些目标后，教师动作拟合应明显改善。
3. phase 平衡应作用于 replay 抽样，而不是只缩放单批次权重；稀有 phase 需要有界重复采样。

若 held-out teacher-action gate 仍失败，优先怀疑状态表示、循环上下文或模型容量，而不是继续增加端到端训练量。

## 3. 固定协议

| 用途 | seed 范围 | 数量 | 是否可选模 |
|---|---:|---:|---|
| 启发式教师训练语料 | 400000-404095 | 4096 | 是，训练数据 |
| 教师动作验证语料 | 1500000-1500511 | 512 | 是，早停与最佳 checkpoint |
| 端到端 promotion gate | 1600000-1600511 | 512 | 只用于最终晋级判断 |
| 最终盲测 | 3000000-3002047 | 2048 | M7-B 禁止访问 |

已经揭示的 M7 pilot gate `1400000-1400511` 不得用于 M7-B 选模或晋级。

本地 `--smoke` 允许使用独立开发 seed，但所有报告必须标记为工程验证，不能作为性能证据。

## 4. 训练方法

1. 使用确定性 `HeuristicPolicy` 完整运行游戏并保存逐局 JSONL 轨迹。
2. 只监督 `card_reward`、`event`、`map`、`rest_site` 和 `shop` 中合法动作数大于 1 的状态。
3. 战斗动作由同一启发式教师执行以保持轨迹可复现，但不进入 M7-B loss。
4. 从 M6 seed-17 最佳评测 checkpoint 初始化 128/128/128 recurrent policy。
5. 每个 epoch 复用全部 4096 条持久轨迹；每个 trace batch 按 phase 拆分监督 mask，并最多重复 4 倍。
6. 优化目标只有 teacher action cross-entropy。价值损失、PPO、entropy 和 uniform exploration 均为 0。
7. 每个 epoch 在固定的 512 条 held-out 教师轨迹上评估；按“最差 phase accuracy、平均 phase accuracy、平均 phase cross-entropy”字典序选最佳模型。
8. 最多 20 epochs；连续 3 epochs 不提升则早停。

训练 checkpoint 每个 trace batch 原子保存一次。`SIGINT`、`SIGTERM` 或 stop file 会等待当前 trace batch 完成并保存后退出；最多重算一个尚未完成的 batch。

## 5. 两级门槛

### Teacher-action gate

候选和 M6 initial 必须在同一 512 条 validation corpus 上比较，并同时满足：

- 五个 phase 的动作准确率均不低于 M6 initial；
- 总体动作准确率严格高于 M6 initial；
- 总体 cross-entropy 严格低于 M6 initial；
- corpus 聚合哈希和各 phase 样本数完全一致。

失败时停止，不运行 512 局端到端 gate。

### End-to-end gate

通过 teacher-action gate 后，固定使用 `learned-heuristic`：网络负责非战斗，启发式负责战斗。候选和 M6 initial 在相同的 `1600000-1600511` 上配对比较，并同时满足：

- errors、crashes、illegal actions、recovery failures、timeouts 和 cycles 全为 0；
- 候选减基线的平均终止楼层 95% 分层 bootstrap CI 下界大于 0；
- Act 1 通过率平均差不小于 0。

只有两个门槛都通过，才讨论扩展到 seed 29/43。M7-B pilot 无论结果如何都不得直接运行最终 2048 盲测。

## 6. 结果解释

| Teacher gate | End-to-end gate | 结论与下一步 |
|---|---|---|
| FAIL | 不运行 | 当前网络没有可靠学会教师；检查表示、上下文和容量 |
| PASS | FAIL | 行为克隆有效但闭环分布偏移仍严重；下一步考虑持久化 DAgger，不恢复 PPO |
| PASS | PASS | M7-B 机制成立；再预注册 seed 29/43 复现实验 |

局部准确率提升但端到端失败不是矛盾：少数路径、商店或奖励选择会改变后续整局状态分布，teacher-forced accuracy 不能替代闭环评测。

## 7. 工程验证记录

2026-07-29 的本地 smoke 使用独立开发 seed，仅验证工程链路：

- 16 条训练轨迹、8 条 validation 轨迹，五个 phase 均有覆盖；
- 在 trace batch 1 安全停止，随后从 checkpoint 恢复并完成 2 epochs；
- teacher-action accuracy 从 83.72% 提升到 85.78%，五 phase 无回退，cross-entropy 从 0.4741 降到 0.4071；
- 8 局端到端 gate 正确给出 FAIL，且所有安全错误计数为 0。

样本量过小，因此这些数值不构成 M7-B 有效性的证据。
