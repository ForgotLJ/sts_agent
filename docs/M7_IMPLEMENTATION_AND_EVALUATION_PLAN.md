# M7 组件校正与稳健泛化协议

## 1. 目标

M7 不把“偶然出现一次完整胜利”继续作为主要优化目标。M7 的目标是分别修复非战斗策略和战斗搜索，并在新的未见 seed 上证明完整分层智能体至少达到、随后超过启发式基线。

M6 已完成工程门槛，但其正式结果同时表明：

- `heuristic` 平均终止楼层为 `21.255859`；
- `heuristic-search` 为 `18.438151`，搜索平均损失约 `2.82` 层；
- `learned-search` 为 `16.274089`，相同搜索下的学习非战斗策略仍弱于启发式；
- 三个 run 实际获得的 full-run 更新数为 `100 / 2225 / 3825`；
- 模型选择只重复使用了验证区间前 64 个 seed；
- M6 的三份确定性 heuristic 结果完全相同，不能当作 3072 个独立环境样本。

因此，未经组件诊断直接扩大 M6 训练量不属于 M7 的有效实验。

M7-A 已完成的补充诊断进一步量化了该结论：`learned-heuristic - heuristic` 的平均楼层差为 `-3.561`，两层 bootstrap 95% 区间为 `[-4.103, -2.997]`；`learned-search - learned-heuristic` 为 `-1.421 [-1.830, -1.017]`。完整记录见 `docs/M7_M6_COMPONENT_DIAGNOSTIC.md`。

## 2. 信息边界

M7 延续 M6 的公开信息边界。智能体不得读取环境 seed、未公开牌序、未来遭遇、未来奖励或内部 RNG 状态。

战斗搜索只能通过 `redeterminized_clone()` 重采样隐藏战斗世界。非战斗阶段当前没有经过验证的公开状态重随机化，因此 M7 不实现非战斗精确 clone 或单一隐藏未来上的反事实 Q 教师。非战斗改进使用学生闭环 DAgger、公开轨迹自模仿和阶段平衡监督。

## 3. Seed 协议

所有范围均为闭区间，且由 `M7TrainingConfig` 强制不重叠。

| 用途 | Seed |
|---|---:|
| 训练 | `0-999999` |
| 课程晋级 | `1200000-1200127` |
| 轮转快速筛选 | `1210000-1211023` |
| 大样本 checkpoint 选择 | `1300000-1300511` |
| Pilot 独立门槛 | `1400000-1400511` |
| 组件开发诊断 | `2100000` 起的显式开发区间 |
| M7 最终盲测 | `3000000-3002047` |

M6 的 `2000000-2001023` 已经公开，只能用于补齐 M6 事后诊断。它不得用于 M7 调参、选模或正式结论。

## 4. 固定训练预算

M7 将课程训练和最终阶段训练分开计数：

1. 每个 run 最多使用 `max_curriculum_updates` 次更新进入 `full_run`；
2. 进入 `full_run` 后重新从零计数；
3. 每个 run 必须完成完全相同的 `full_run_updates`；
4. 未在课程预算内进入 `full_run` 的 run 明确失败，不用较少 full-run 训练量继续凑齐总更新数；
5. checkpoint 保存 `M7TrainingProgress`，恢复后继续相同阶段预算。

Control 和 balanced pilot 都从同一份 M6 EMA 权重直接进入 `full_run`，使用 1000 次 full-run 更新，从而只比较 M7 训练改动。正式候选从完整课程开始并暂定 2000 次 full-run 更新；该值只能在 pilot 完成后冻结，不得查看 M7 最终 seed 后修改。

## 5. 验证与选模

课程阶段只使用 128 个晋级 seed。完整运行阶段使用两级验证：

- 每 25 次更新在 1024 个筛选 seed 中轮转取 128 个，监控训练趋势；
- 每 250 次 full-run 更新在固定 512 个 selection seed 上做一次大样本选模；
- screening 不保存最佳模型；只有 selection 可以更新 `best-evaluation-checkpoint.pt`；
- 选模顺序固定为完整胜率、Act 3、Act 2、Act 1 通过率、平均楼层、proxy score；
- selection 的战斗模块由配置显式指定。当前 pilot 使用 `heuristic`，避免尚未通过门槛的搜索污染非战斗策略选模。

最终冻结同时验证最佳 EMA checkpoint 和完成固定预算的可恢复训练 checkpoint。最佳模型可以来自 full-run 中途，但对应 run 必须完整训练到预算终点。

## 6. 非战斗训练

M7 保留 PPO、DAgger、前沿自模仿和参数 EMA，不先扩大网络规模。主要变化是监督覆盖：

- `ImitationChunk` 记录每个监督步骤的界面类型；
- M7 对当前批次中的监督权重做有界逆频率校正；
- 默认最大校正倍数为 4，防止极少数样本产生不稳定梯度；
- 指标同时记录原始和重加权后的 phase coverage；
- control 配置保持 M6 的静态 `1/1/3/2/4` 权重，balanced 配置使用统一初始权重和动态平衡。

第一轮单种子 ablation 只比较 control 与 balanced DAgger。checkpoint 仍只由 selection seed 选择；选择完成后，二者在从未参与选模的 pilot-gate seed 上比较。只有 balanced 相对 control 的平均楼层配对 bootstrap 下界大于零，且 Act 1 通过率不退化，才进入三种子确认。

## 7. 战斗搜索门槛

搜索不再凭首场战斗结果直接进入完整智能体。正式接入前必须：

1. 从训练或开发 seed 的完整启发式运行中抽取按 Act 平衡的可重放战斗起点；
2. 在相同起点上比较 heuristic 与 16/64/256 调用搜索；
3. 报告战斗胜率、HP 变化、动作数、真实模拟调用和墙钟时间；
4. 所有错误必须为零；
5. 64 调用搜索相对 heuristic 的战斗胜率和 HP 变化必须通过预先冻结的非劣门槛。

若搜索未通过，M7 正式智能体使用启发式战斗。搜索失败不会阻止非战斗学习继续，但不得以 `learned-search` 作为主方法。

## 8. 统计协议

正式汇总以环境 seed 和训练 run seed 为两个独立层级：

- 确定性 heuristic 只需要评测一次，不复制成三个独立样本；
- 学习方法保留三个训练 run 的所有结果；
- 配对差值使用相同环境 seed；
- 置信区间同时重采样训练 run 和环境 seed；
- 报告 episode record 数、独立环境 seed 数、记录胜利数和独立胜利 seed 数；
- 完全重复的评测文件产生显式 warning。

## 9. 晋级与正式成功条件

Pilot 晋级条件：

- 全部工程错误为零；
- balanced 相对 control 的配对平均楼层 95% 区间下界大于零；
- Act 1 通过率不退化；
- phase coverage 中主要非战斗界面均有有效监督，不用总体准确率掩盖稀有界面缺失。

M7 正式成功条件：

- 三个 run 均完成冻结的 full-run 更新预算；
- 最终 2048 个盲测 seed 在 checkpoint 冻结前未被访问；
- 主方法相对 heuristic 的平均楼层分层 bootstrap 95% 区间下界大于零；
- Act 1 通过率非劣；
- 至少两个训练 run 产生完整 A0 胜利；
- 所有评测无崩溃、非法动作、恢复失败、超时或循环。

## 10. 主要入口

- `scripts/train-m7.py`：训练、暂停恢复、筛选和选模主程序；
- `scripts/evaluate-m7.py`：M6 事后诊断和 M7 正式评测；
- `scripts/run-m7-component-diagnostics.py`：补跑 `learned-heuristic` 并分解组件损失；
- `scripts/collect-m7-combat-corpus.py`：抽取完整分布战斗起点；
- `scripts/benchmark-m7-combat.py`：单方法战斗基准；
- `scripts/summarize-m7-combat.py`：配对统计和搜索非劣门槛；
- `scripts/freeze-m7-source.py` 与 `scripts/freeze-m7.py`：源码和 checkpoint 冻结。
