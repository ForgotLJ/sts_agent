# M5 随机性感知搜索与策略价值蒸馏报告

## 结论

M5 的战斗内目标已经完成，并在两层证据上通过验证：

1. 可穷举随机战斗夹具证明搜索与精确期望最优动作一致；
2. `sts_lightspeed` 的 Ironclad A0 首场战斗中，信念搜索在 64 和 256 次严格模拟调用预算下超过启发式与纯网络。

该结论仅覆盖 **Ironclad、A0、首场战斗**。它不表示智能体已经学会完整爬塔、奖励选择、路线、商店、事件或 Boss 决策。

## 信息边界

- `seed` 已从 `Observation` 中移除，只保留在 reset/transition 日志元数据中；
- 正式搜索只接收公开观察、公开历史和动态合法动作；
- `LightspeedBackend.redeterminized_clone()` 只允许在战斗内调用；
- 未公开的抽牌顺序被重新洗牌，未来战斗随机流使用独立 `search_seed` 重建；
- 显式 `known_top` / `known_bottom` 约束在重采样时保持；
- 精确隐藏状态 clone 被隔离在 `search/diagnostics/oracle.py`，必须显式传入 `allow_hidden_state=True`；
- 搜索进入奖励界面时立即截断，不继续读取战斗后的隐藏奖励、路线或遭遇信息。

精确 clone Oracle 仅用于测量隐藏未来带来的虚假收益，不作为正式智能体。

## 搜索算法

实现采用 root-sampling POMCP/ISMCTS 风格搜索：

1. 每次模拟从当前公开状态重新采样一个隐藏世界；
2. 决策节点使用 PUCT 选择动作；
3. 随机结果按公开观察键聚合；
4. chance outcome 使用渐进扩展限制分支数量；
5. 叶节点可执行启发式 rollout，所有 rollout step 都计入模拟调用预算；
6. 返回根节点访问分布、动作均值、搜索价值、展开节点数和真实 simulator calls。

搜索预算是硬上限，不是名义 simulation 数。16、64、256 调用方法的实际 `calls_per_decision` 分别严格为 16、64、256。

## 策略价值蒸馏

蒸馏数据完全由模拟器搜索生成，不使用人类牌局数据：

- 策略头对当前每个动态合法动作独立评分；
- 价值头对候选动作特征做置换不变池化后预测搜索价值；
- 教师策略使用搜索最终改进行动作为监督目标；
- 组合搜索使用网络先验和价值，但与均匀先验、规则叶值混合，防止欠训练网络支配搜索；
- checkpoint 保存模型、优化器、RNG、编码器和训练配置，并重新加载逐 episode 复现评估结果。

## 可穷举夹具

随机战斗夹具包含两个可能隐藏牌序。精确 belief solver 穷举隐藏世界并在每个公开观察后重新优化动作。

正式夹具实验：

- 运行 seed：`17 / 29 / 43`；
- 评估 seed：`1000000–1000255`；
- 搜索预算：`16 / 64 / 256`；
- 所有验收门槛通过；
- 网络与 Candidate-Q 均学会精确最优动作；
- 低预算网络+搜索达到穷举最优分数；
- 256 调用纯搜索达到穷举最优分数。

产物：`experiments/m5_fixture/summary.json`。

## Lightspeed 首场战斗实验

实验设置：

- 角色：Ironclad；
- 难度：A0；
- 范围：每个 seed 的首场战斗；
- 训练 seed：`0–1999`；
- 评估 seed：`1000000–1000063`；
- 独立 run seed：`17 / 29 / 43`；
- 每个方法保留所有失败、超步数和低分 episode；
- 评估分数：`战斗胜负 + 最终 HP 比例 - 0.01 × 动作数`。

### 正式结果

| 方法 | 平均分 | run 95% CI | 平均最终 HP | 调用/决策 |
|---|---:|---:|---:|---:|
| Random | 1.3780 | — | — | 0 |
| Heuristic | 1.8660 | episode CI 0.0135 | 76.59 | 0 |
| Candidate-Q | 1.2571 | 0.3834 | 53.32 | 0 |
| Distilled network | 1.8486 | 0.0080 | 76.10 | 0 |
| Belief search 16 | 1.8435 | 0.0055 | 77.66 | 16 |
| Belief search 64 | 1.8761 | 0.0024 | 78.43 | 64 |
| Belief search 256 | 1.8850 | 0.0057 | 78.76 | 256 |
| Network + search 16 | 1.8637 | 0.0033 | 77.82 | 16 |
| Exact-clone Oracle 64 | 1.8930 | 0.0000 | 78.97 | 64 |
| Exact-clone Oracle 256 | 1.8915 | 0.0000 | 79.03 | 256 |

配对到相同评估 seed 的结果：

- 64 调用搜索相对启发式：`+0.0101 ± 0.0024`；
- 256 调用搜索相对启发式：`+0.0190 ± 0.0057`；
- 搜索质量随 16 → 64 → 256 调用总体单调提升；
- 蒸馏网络保留约 92.8% 的“随机基线到 256 调用搜索”分数增益；
- 16 调用组合搜索优于同预算纯搜索，但 64/256 调用下纯搜索更好，说明当前网络仍会干扰高预算搜索；
- Oracle 与信念搜索之间仍有正差距，证明直接复制精确 RNG 会高估算法能力。

产物：`experiments/m5_lightspeed_combat/summary.json`。

## 负面结果

- Candidate-Q 在真实首场战斗训练中跨 run 方差很大，没有稳定超过随机或启发式；
- 蒸馏网络超过随机，但没有超过人工启发式；
- 高预算网络+搜索弱于纯信念搜索；
- 因此当前证据支持“搜索有效”和“蒸馏保留大部分搜索增益”，不支持“网络已替代搜索”或“组合方法在所有预算下最优”。

## 验证

- Python 测试：76 项通过；
- 扩展独立导入：100 次通过；
- 固定 seed 确定性：10,000 个 seed 通过；
- 夹具隐藏牌序采样接近均匀分布；
- Lightspeed 重采样前后根公开观察完全一致；
- 不同 `search_seed` 产生不同未公开未来手牌；
- checkpoint 重新加载后逐 episode 结果完全一致。

## 复现命令

```powershell
cd D:\Project\STS\sts_agent
$env:PYTHONPATH='src'

& "$env:USERPROFILE\.conda\envs\pytorch_env\python.exe" `
  scripts\run-m5-fixture-experiment.py `
  --output experiments\m5_fixture

& "$env:USERPROFILE\.conda\envs\pytorch_env\python.exe" `
  scripts\run-m5-lightspeed-combat.py `
  --output experiments\m5_lightspeed_combat
```

## 剩余限制

- 自动公开历史跟踪尚未覆盖所有置顶、置底、预知和牌堆搜索卡牌；当前 API 支持约束，但首场战斗实验使用初始卡组，不触发这些机制；
- 重采样目前仅允许战斗内使用，不能搜索隐藏奖励、未来遭遇或完整路线；
- 当前网络是候选动作 MLP，没有循环记忆或长程 run-level 表征；
- M6 必须建立房间级决策、完整 A0 课程与整局终局评价，不能把首场战斗结果外推到完整游戏。
