# Slay the Spire Learning Agent

这是一个面向《杀戮尖塔》的可复现学习与搜索环境。公共接口只暴露结构化 `Observation`、动态合法 `Action` 和 `Transition`，不暴露隐藏 RNG、抽牌顺序或模拟器内部状态。

## 当前进度

- M0–M2：`sts_lightspeed` 构建、统一环境、clone、确定性、100k fuzz 和性能验证完成。
- M3：CommunicationMod 实机差分完成，最终连续轨迹与终局均零差异。
- M4：动态候选动作 Candidate-Q 训练基线完成，3 个训练 seed 在 256 个未见 seed 上稳定超过随机策略。
- M5：战斗内随机性感知搜索与策略价值蒸馏完成；在 64 个未见 seed 的 Ironclad A0 首场战斗中，64/256 调用搜索显著超过启发式。完整爬塔尚未开始。

## 后端

- `ToyCombatBackend`：小型确定性训练与基础设施测试环境。
- `LightspeedBackend`：高速完整规则模拟器绑定，支持独立 clone。
- `CommunicationBackend`：真实游戏真值后端，用于固定 seed 差分验证。

## 环境准备

- Python：`C:\Users\ForgotLJ\.conda\envs\pytorch_env\python.exe`（3.11）
- C++：Visual Studio Build Tools 2022、MSVC v143、CMake、Ninja
- 游戏：Steam Slay the Spire、ModTheSpire、BaseMod、CommunicationMod 1.2.1

训练依赖：

```powershell
& "$env:USERPROFILE\.conda\envs\pytorch_env\python.exe" -m pip install -e ".[training]"
```

## 构建与测试

```powershell
cd D:\Project\STS\sts_agent
.\scripts\configure-lightspeed.cmd
.\scripts\build-lightspeed.cmd 4
.\scripts\check-lightspeed.cmd

$env:PYTHONPATH='src'
& "$env:USERPROFILE\.conda\envs\pytorch_env\python.exe" `
  -m unittest discover -s tests -v

.\scripts\fuzz-lightspeed.cmd --steps 100000
```

## Candidate-Q 基线

```powershell
$env:PYTHONPATH='src'
& "$env:USERPROFILE\.conda\envs\pytorch_env\python.exe" `
  scripts\train-candidate-q.py `
  --output experiments\candidate_q_toy_m4
```

配置：`config\candidate_q_toy.json`

报告：`docs\M4_TRAINING_BASELINE_REPORT.md`

## M5 搜索与蒸馏

```powershell
$env:PYTHONPATH='src'
& "$env:USERPROFILE\.conda\envs\pytorch_env\python.exe" `
  scripts\run-m5-lightspeed-combat.py `
  --output experiments\m5_lightspeed_combat
```

配置：`config\m5_lightspeed_combat.json`

报告：`docs\M5_SEARCH_AND_DISTILLATION_REPORT.md`

## 实机差分恢复

```powershell
& "$env:USERPROFILE\.conda\envs\pytorch_env\python.exe" `
  scripts\real-game-differential.py `
  --seed 1 `
  --resume `
  --resume-trace real_game_traces\communication.jsonl `
  --resume-prefix-trace real_game_traces\communication-pre-clean-20260716.jsonl `
  --steps 30
```

详细说明：

- `docs\IMPLEMENTATION_AND_TEST_PLAN.md`
- `docs\M3_REAL_GAME_RUNBOOK.md`
- `docs\M4_TRAINING_BASELINE_REPORT.md`
- `docs\M5_SEARCH_AND_DISTILLATION_REPORT.md`
- `docs\STS_LIGHTSPEED_PATCHES.md`

## Ubuntu M6 正式训练

服务器构建、迁移资产、性能档位、暂停恢复和最终审计见：

- `docs\M6_SERVER_RUNBOOK.md`

可先查看服务器资源解析结果：

```bash
.venv/bin/python scripts/run-m6-server.py plan --profile balanced
```
