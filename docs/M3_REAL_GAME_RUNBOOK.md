# M3 真实游戏差分运行手册

## 已安装组件

- CommunicationMod：`D:\SteamLibrary\steamapps\common\SlayTheSpire\mods\CommunicationMod.jar`
- 版本：`1.2.1`
- SHA-256：`12B99249ECCFD5245DDB0B6B7575E1A1DFBA0CE0ED6FAD5BD5BBD4A303897EF5`
- 配置：`C:\Users\ForgotLJ\AppData\Local\ModTheSpire\CommunicationMod\config.properties`
- Relay：`D:\Project\STS\sts_agent\scripts\communication-relay.py`
- TCP：`127.0.0.1:51234`
- Python：`C:\Users\ForgotLJ\.conda\envs\pytorch_env\python.exe`（3.11）

## 启动流程

1. 使用 Steam 启动 ModTheSpire。
2. 只启用 `BaseMod` 与 `Communication Mod`；其他不影响规则的模组也建议在差分验证时关闭。
3. 点击 `Play`，进入游戏主菜单或需要续跑的存档界面。
4. 检查 relay：

```powershell
Get-NetTCPConnection -LocalPort 51234 -State Listen
```

## 从主菜单开始固定 seed 差分

数值 seed `0` 会被游戏视为未指定，因此实机验证使用 `seed=1`：

```powershell
cd D:\Project\STS\sts_agent
$env:PYTHONPATH='src'
& "$env:USERPROFILE\.conda\envs\pytorch_env\python.exe" `
  scripts\real-game-differential.py `
  --seed 1 `
  --steps 50 `
  --output real_game_traces\differential.jsonl
```

## 从中断存档恢复

`--resume-trace` 重放当前 run 的真实命令；若 relay 轨迹在首战后轮换，使用 `--resume-prefix-trace` 提供轮换前的地图前缀：

```powershell
& "$env:USERPROFILE\.conda\envs\pytorch_env\python.exe" `
  scripts\real-game-differential.py `
  --seed 1 `
  --resume `
  --resume-trace real_game_traces\communication.jsonl `
  --resume-prefix-trace real_game_traces\communication-pre-clean-20260716.jsonl `
  --steps 30 `
  --output real_game_traces\differential-resume.jsonl
```

恢复过程会：

- 根据真实 Neow 历史选择 `full`、`limited` 或 `skipped`；
- 用地图节点 `x` 坐标约束同形战斗前缀；
- 重放 `PLAY`、`END`、`POTION`、`CHOOSE`、`PROCEED`、`RETURN`；
- 校正卡牌奖励和商店的两阶段合成状态；
- 在发送新动作前要求当前公共状态零差异。

## 同步规则

- 状态变化按公共 `Observation` 判断，而不是原始 JSON 文本变化。
- 带 `wait` 的状态需要连续两次相同公共观察才视为稳定。
- FTUE、设置页、事件结果页和休息结果页作为展示层处理，不推进模拟器规则状态。
- 卡牌奖励拆为“打开奖励”和“选择具体卡牌”。
- 商店拆为“进入商店”和“购买/返回”；`RETURN` 只关闭商店内部页，`PROCEED` 才回地图。
- 地图动作按 `x` 坐标匹配，商品和卡牌按规范化 ID 匹配。
- 死亡敌人的状态与意图、非战斗玩家状态不进入规则差分。
- `config\differential_allowlist.json` 当前为空；不得用白名单隐藏规则差异。

## M3 验收证据

- 原始协议轨迹：`real_game_traces\communication.jsonl`
- 最终连续轨迹：`real_game_traces\differential-seed1-aitest-final-clean.jsonl`
  - step 0–29：全部 `differences=0`
  - 覆盖地图、普通战斗、奖励、休息点及精英战
- 终局复核：`real_game_traces\differential-seed1-aitest-final-terminal-check.jsonl`
  - `reference=terminal`
  - `candidate=terminal`
  - `differences=0`
- 额外覆盖：商店购买、商店 `RETURN`、`PROCEED`、事件、宝箱、遗物、药水丢弃、多敌人和精英战。
- Python 全量测试：57 项通过。
- 扩展 smoke check：100 次独立导入、10000 个固定 seed 确定性检查通过。
- 随机 fuzz：100000 step 通过，17788.9 step/s，稳定内存增长 0.64 MiB。

## 当前结论

M3 已达到完成门槛：支持范围内的关键公共字段可逐步一致，已发现的问题均通过同步、动作语义、公共编码或模拟器规则修复解决，没有加入差异白名单。M4 训练基础设施尚未开始。
