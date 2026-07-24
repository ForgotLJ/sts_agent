# M4 训练基础设施与 Candidate-Q 基线报告

## 结论

M4 已完成。动态候选动作 Candidate-Q 基线在 3 个独立训练 seed 上，均在完全未见的 256 个评估 seed 上稳定超过随机策略。

本结论只适用于 `ToyCombatBackend` 的训练管线验收，不代表已经学会完整《杀戮尖塔》运行策略。

## 实现范围

- 同步多环境 actor/collector：`src/sts_env/training/collector.py`
- 对象化状态与动作编码：`src/sts_env/training/encoding.py`
- 动态候选动作 Q 网络与 Double-DQN 目标：`src/sts_env/training/candidate_q.py`
- 轨迹、回放缓冲区和 JSONL：`src/sts_env/training/replay.py`
- 随机、启发式和一步 clone 搜索：`src/sts_env/training/policies.py`
- 未见 seed 评估和 95% 区间：`src/sts_env/training/evaluation.py`
- 三 seed 实验、TensorBoard、checkpoint 和运行清单：`src/sts_env/training/experiment.py`
- 可复现配置：`config/candidate_q_toy.json`

Candidate-Q 不使用固定动作编号输出层。网络对当前 `Observation` 中每个合法 `Action` 独立编码和评分，因此候选动作数量可以动态变化，非法动作不会进入 argmax。

## 训练信号

环境终局奖励保持不变。训练器显式使用以下 shaping：

```text
r_train = r_environment
          + 0.03 * damage_dealt
          - 0.04 * damage_taken
          - 0.002
```

评估不使用训练 shaping，而使用：

```text
score = environment_return + final_hp / max_hp - 0.01 * episode_length
```

Toy 环境的随机策略胜率接近 100%，因此只比较胜率无法检验是否学到了更有效率的策略；该分数同时评价终局、剩余生命和动作效率。

## Seed 隔离

- 训练 seed：`0–19999`
- 评估 seed：`1000000–1000255`
- 训练 run seed：`17`、`29`、`43`
- 训练与评估 seed 集合严格不相交。

## 正式结果

实验产物：`experiments/candidate_q_toy_m4/summary.json`

| 策略 | 平均分 | 95% 区间半径 | 胜率 | 平均步数 | 平均终局 HP |
|---|---:|---:|---:|---:|---:|
| Random | 1.4851 | 0.0315 | 99.61% | 17.98 | 53.82 |
| Heuristic | 1.8078 | 0.0032 | 100% | 8.72 | 71.60 |
| One-step search | 1.8078 | 0.0032 | 100% | 8.72 | 71.60 |
| Candidate-Q，seed 17 | 1.8019 | 0.0038 | 100% | 9.33 | 71.62 |
| Candidate-Q，seed 29 | 1.8035 | 0.0035 | 100% | 9.14 | 71.60 |
| Candidate-Q，seed 43 | 1.8099 | 0.0030 | 100% | 8.51 | 71.60 |

三次 Candidate-Q run 的平均分为 `1.8051 ± 0.0048`，比随机策略高 `0.3200`。Candidate-Q 的跨 run 下界仍高于随机策略的单 episode 统计上界。

## 数值与恢复验证

- 全量 Python 测试：64 项通过。
- loss、Q 值、目标值和梯度范数均检查有限性。
- 梯度使用全局范数裁剪，正式 run 最后梯度范数为 `0.0404–0.0721`。
- 三个正式 checkpoint 均重新加载并在全部 256 个评估 seed 上复评，分数与原报告逐位一致。
- checkpoint 保存网络、目标网络、优化器、回放缓冲区、训练计数、Python RNG、Torch RNG、编码器配置和实验元数据。
- 运行清单记录 Torch/TensorBoard 版本、CUDA/GPU、源码 SHA-256。当前目录没有有效 Git 元数据，因此 `git_commit` 明确记录为 `unavailable`，不伪造提交号。

## 产物

- 聚合报告：`experiments/candidate_q_toy_m4/summary.json`
- seed 17：`experiments/candidate_q_toy_m4/seed-17/checkpoint.pt`
- seed 29：`experiments/candidate_q_toy_m4/seed-29/checkpoint.pt`
- seed 43：`experiments/candidate_q_toy_m4/seed-43/checkpoint.pt`
- TensorBoard：各 seed 目录下的 `tensorboard/events.out.tfevents.*`
- 快速冒烟实验：`experiments/candidate_q_toy_quick/summary.json`

## 复现命令

```powershell
cd D:\Project\STS\sts_agent
$env:PYTHONPATH='src'
& "$env:USERPROFILE\.conda\envs\pytorch_env\python.exe" `
  scripts\train-candidate-q.py `
  --output experiments\candidate_q_toy_m4
```

快速管线检查：

```powershell
& "$env:USERPROFILE\.conda\envs\pytorch_env\python.exe" `
  scripts\train-candidate-q.py `
  --quick `
  --output experiments\candidate_q_toy_quick
```

## 限制与下一步

- 当前训练结论仅证明动态动作训练管线能够学习，不证明完整游戏泛化。
- Toy 环境的启发式和一步搜索上限相同，不能用于研究长程信用分配。
- M5 应在相同模拟调用预算下加入随机性感知搜索，并让网络蒸馏搜索策略与价值。
