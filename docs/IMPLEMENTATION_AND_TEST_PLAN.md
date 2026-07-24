# Slay the Spire Learning Agent：实现与测试计划

## 1. 固定目标

第一阶段目标是建立一个**可复现、可高速采样、不可读取隐藏 RNG、可被搜索克隆**的《杀戮尖塔》训练环境。

首个正式范围：

- 角色：Ironclad；
- 难度：Ascension 0；
- 内容：先战斗，后完整三幕；
- 输入：结构化公开状态，而非截图；
- 输出：当前状态下的动态合法动作；
- 奖励：环境只提供真实终局和事件结果，奖励塑形放在独立 wrapper；
- 真实游戏：只作为规则真值、轨迹记录和最终部署环境，不用于大规模训练；
- 模拟器：`sts_lightspeed` 的本地审计版本；
- 训练 Python：`pytorch_env`，Python 3.11；
- GPU：RTX 5070 Ti Laptop，CUDA PyTorch 已验证可用。

明确不做：

- 不在第一阶段训练完整通关策略；
- 不从像素端到端学习；
- 不让智能体或搜索读取未来牌序、未来随机数或游戏 RNG 内部状态；
- 不把未经真实游戏校验的模拟结果作为最终性能结论；
- 不同时支持四个角色和全部 Ascension。

## 2. 系统边界

```text
StsEnv
  ├─ ToyCombatBackend          # 接口和算法快速测试
  ├─ LightspeedBackend         # 高速正式训练
  └─ CommunicationBackend      # 真实游戏真值与部署

Observation -> legal Action candidates -> Backend.step(Action)
                                      -> reward / termination / info

Backend.clone() -> independent branch for search
```

所有后端必须通过同一套 contract tests。训练代码不得依赖任何后端专属对象。

## 3. 里程碑

### M0：工具链与上游模拟器加固

实现内容：

1. 完成 `sts_lightspeed` 的 MSVC 兼容补丁；
2. 将内置 pybind11 固定到兼容 Python 3.11 的版本；
3. 生成可被 `pytorch_env` 导入的 `slaythespire.pyd`；
4. 编写一键 configure、build、import-check 脚本；
5. 将所有上游修改记录为独立补丁说明，避免与本项目逻辑混杂；
6. 关闭或修复已发现的未定义行为警告，尤其是位移越界；
7. 只保留实际需要的构建目标，减少重复编译。

测试：

- CMake configure 在干净构建目录成功；
- Release 编译成功；
- Python 3.11 可导入扩展；
- 独立 Python 进程连续导入 100 次无失败；
- 创建 10,000 个固定种子局面无崩溃；
- 同一 seed 的公开状态哈希完全一致；
- C++ 原生测试目标运行成功或记录明确的不兼容项。

完成门槛：

> Python 扩展稳定导入，固定种子初始化确定，构建过程不依赖手工打开 Developer Command Prompt。

### M1：正式模拟器适配层

实现内容：

1. 新增 `LightspeedBackend`；
2. 将 C++ 游戏状态转换为公共 `Observation`；
3. 将 C++ 可执行操作转换为公共 `Action`；
4. 支持 reset、step、clone、终止状态和错误报告；
5. 明确公开信息与隐藏信息边界；
6. 建立稳定的 card、relic、enemy、status ID 映射；
7. 为不同 phase 建立动作候选：战斗、奖励、地图、商店、火堆和事件。

测试：

- 所有合法动作都能被执行；
- 非法动作在进入 C++ 前被拒绝；
- clone 后执行相同动作得到相同结果；
- clone 的一个分支不会污染另一个分支；
- 公开观测中不存在 RNG 状态和未来牌序；
- 观测可 JSON 序列化并稳定恢复；
- Toy 与 Lightspeed 后端通过同一 contract suite。

完成门槛：

> 随机合法策略可以在高速后端连续运行 100,000 步，无崩溃、死锁、非法状态或内存持续增长。

### M2：规则性质测试与性能基准

实现内容：

1. 添加随机合法动作 fuzz runner；
2. 添加状态不变量检查器；
3. 添加牌堆守恒与临时牌例外检查；
4. 添加固定 seed 的轨迹记录与回放；
5. 添加单线程、多线程和 Python 调用吞吐基准；
6. 记录初始化、step、clone 和序列化的延迟分布。

核心不变量：

- HP、能量、格挡和层数不出现无解释的非法值；
- 未发生生成、消耗、变形时，牌实例总数守恒；
- 死亡后不存在可执行动作；
- 当前动作列表不包含无法支付费用或无有效目标的动作；
- 相同公开历史下不能通过观测推断内部 RNG 指针；
- 搜索 clone 不共享可变容器。

性能门槛：

- 训练后端至少比真实游戏快 100 倍；
- Python 单进程达到足以持续喂满 GPU 的状态生成速度；
- 若 Python 绑定成为瓶颈，批量 step/rollout API 在本阶段实现；
- 所有性能数据同时报告 CPU 线程数和模拟调用数。

### M3：真实游戏通信与差分验证

实现内容：

1. 接入 CommunicationMod JSON 协议；
2. 实现 `CommunicationBackend`；
3. 记录真实游戏的 observation/action JSONL 轨迹；
4. 建立真实状态到公共 `Observation` 的解析器；
5. 对相同 seed 和动作脚本进行模拟器差分；
6. 建立差异白名单，禁止静默忽略不一致；
7. 区分显示差异、可接受抽象差异和规则错误。

测试集：

- Starter deck 对 Act 1 普通敌人；
- 多敌人目标选择；
- 易伤、虚弱、力量、格挡和多段攻击；
- 抽牌、弃牌、消耗、洗牌；
- 药水使用和丢弃；
- 卡牌奖励、跳过、路线和火堆；
- 固定 seed 的地图和战斗初始状态。

完成门槛：

> 支持范围内的关键字段逐步一致；任何不一致都必须有最小复现、分类和处理决定。

验证状态（2026-07-16）：**已完成**。

- `AITEST`、seed 1 实机轨迹已覆盖 START、PLAY、END、POTION、CHOOSE、PROCEED、RETURN；
- 已覆盖多敌人、普通战斗、精英战、卡牌/金币/遗物/药水奖励、地图、事件、商店、休息点和宝箱；
- `real_game_traces/differential-seed1-aitest-final-clean.jsonl` 的 step 0–29 全部零差异；
- `real_game_traces/differential-seed1-aitest-final-terminal-check.jsonl` 的终局状态零差异；
- `config/differential_allowlist.json` 保持为空；
- 全量 Python 测试 57 项通过，扩展独立导入与 10000 seed 确定性 smoke check 通过；
- 100000 step 随机 fuzz 通过，稳定内存增长 0.64 MiB；
- M4 尚未开始，未启动神经网络训练。

### M4：训练基础设施与基线

实现内容：

1. 建立 vectorized actor/collector；
2. 建立轨迹、回放缓冲区和 checkpoint 格式；
3. 建立随机策略、规则启发式和搜索策略基线；
4. 建立对象化状态编码与动态候选动作评分；
5. 实现一个简单可复现的 RL 基线，优先 recurrent PPO 或 candidate-Q；
6. 添加 TensorBoard 指标和机器可读实验摘要。

测试：

- Toy 环境上能学习超过随机策略；
- 固定 seed 下恢复 checkpoint 后动作一致；
- 训练和评估 seed 完全隔离；
- 无 NaN、非法概率、全动作被 mask 或梯度爆炸；
- 同配置至少 3 个 seed，并报告均值和区间。

完成门槛：

> 基线在未见 seed 上稳定超过随机策略，且整套训练可从配置文件完全复现。

验证状态（2026-07-16）：**已完成**。

- 已实现同步 vector collector、JSONL replay、完整 checkpoint 和 TensorBoard/JSON 输出；
- 已实现随机、启发式、一步 clone 搜索和动态候选动作 Candidate-Q；
- 训练 seed `0–19999` 与评估 seed `1000000–1000255` 完全隔离；
- 正式 run seed 为 `17/29/43`，每个 run 训练 8000 step；
- Candidate-Q 三 run 平均分 `1.8051 ± 0.0048`，随机策略为 `1.4851 ± 0.0315`；
- 三个 checkpoint 重新加载后在全部 256 个评估 seed 上逐位复现原分数；
- 全量 Python 测试 64 项通过；
- 详细证据见 `docs/M4_TRAINING_BASELINE_REPORT.md` 和 `experiments/candidate_q_toy_m4/summary.json`；
- 结论仅适用于 Toy 训练管线，M5 搜索改进尚未开始。

### M5：搜索改进与策略蒸馏

实现内容：

1. 实现随机性可感知的树搜索；
2. 使用 observation history 或粒子表示隐藏状态；
3. 限制 chance branching，并实现 progressive widening；
4. 以策略网络作为先验、价值网络作为截断评估；
5. 保存根节点改进策略和搜索回报；
6. 训练网络拟合搜索策略与价值；
7. 比较固定模拟预算下搜索、网络和二者结合的效果。

测试：

- 小型确定性局面与穷举最优解一致；
- 搜索不得读取实际未来牌序；
- 增加模拟预算时动作质量总体不下降；
- clone 数量和内存随预算受控增长；
- 蒸馏网络在未搜索时逼近搜索策略。

完成门槛：

> 在相同环境调用预算下，搜索改进策略显著超过纯网络和启发式基线。

验证状态（2026-07-16）：**战斗内范围已完成**。

- `seed` 已从策略可见 `Observation` 中移除；
- 已实现公开历史、置顶/置底约束和战斗内隐藏状态重采样；
- 已实现 root-sampling 粒子信念搜索、PUCT、chance progressive widening 和受预算计数的 rollout；
- 小型随机战斗夹具与穷举最优动作一致；
- 已隔离精确 clone Oracle，用于量化隐藏未来泄漏；
- 已实现动态策略头、状态价值头、搜索目标 JSONL、checkpoint 和 TensorBoard；
- Ironclad A0 首场战斗正式评估使用 3 个 run seed 和 64 个未见 seed；
- 64 调用搜索相对启发式配对提升 `+0.0101 ± 0.0024`；
- 256 调用搜索相对启发式配对提升 `+0.0190 ± 0.0057`；
- 蒸馏网络保留约 92.8% 的随机基线到 256 调用搜索增益；
- 结论仅适用于首场战斗，完整 A0 run 仍属于 M6；
- 详细证据见 `docs/M5_SEARCH_AND_DISTILLATION_REPORT.md`。

### M6：完整 A0 运行环境

实现内容：

1. 扩展奖励、地图、商店、事件、火堆和 Boss relic 决策；
2. 将战斗决策和房间级决策建模为不同 phase；
3. 实现完整运行轨迹、崩溃恢复和断点续训；
4. 建立 Act 1、Act 2、Act 3 课程；
5. 最终只以整局目标评价，辅助奖励逐步退火。

完成门槛：

> 智能体可以独立完成 A0 全流程，所有性能结论基于未见 seed 和置信区间，而非挑选录像。

## 4. 测试矩阵

| 层级 | 目的 | 运行频率 |
|---|---|---|
| 单元测试 | 状态、动作、编码、奖励和工具函数 | 每次修改 |
| Contract tests | 保证所有后端具有相同语义 | 每次后端修改 |
| Property/fuzz tests | 发现稀有组合和非法状态 | 每日或长任务 |
| Determinism tests | 固定 seed 可复现 | 每次构建 |
| Clone tests | 保证搜索分支隔离 | 每次 C++ 修改 |
| Differential tests | 模拟器对齐真实游戏 | 每次规则修改 |
| Performance tests | 防止吞吐退化 | 里程碑与发布前 |
| Learning sanity tests | 防止训练管线表面运行但无法学习 | 每次算法修改 |
| Statistical evaluation | 支撑真实性能结论 | 正式实验 |

## 5. 可复现性规则

- 每次 episode 保存环境 seed、策略 seed 和搜索 seed；
- 训练 seed 与评估 seed 使用不重叠集合；
- 模型配置、Git 状态、依赖版本和 GPU 信息写入 checkpoint；
- 性能比较使用相同模拟调用数和墙钟时间两套口径；
- 正式胜率报告样本数和 Wilson 置信区间；
- 失败运行不得从统计中静默删除；
- 任何读取隐藏 RNG 的调试接口必须与训练接口物理隔离。

## 6. 依赖策略

已满足：

- Visual Studio Build Tools 2022；
- MSVC v143、Windows SDK、CMake、Ninja；
- Git 与 Conda；
- `pytorch_env`：Python 3.11、CUDA PyTorch；
- NVIDIA 驱动与 RTX 5070 Ti；
- Steam 版 Slay the Spire。

由项目自动管理，不要求人工安装：

- pybind11；
- nlohmann/json；
- Gymnasium；
- pytest/Hypothesis；
- TensorBoard；
- 其他纯 Python 工具包。

只有真实游戏验证需要人工参与：

1. Steam Workshop 安装 ModTheSpire；
2. Steam Workshop 安装 BaseMod；
3. 在 ModTheSpire 中启用上述两项并成功启动游戏一次；
4. 后续将 CommunicationMod.jar 放入游戏模组目录并启用；
5. CommunicationMod 的命令配置由项目脚本生成，但首次启动和模组勾选需要人工确认。

暂不安装：

- WSL；
- CUDA Toolkit 或 cuDNN；
- 独立 Java JDK/Maven；
- MSYS2/MinGW。

MSYS2 仅作为 MSVC 兼容补丁成本过高时的备用路径，未收到明确通知前不要安装。

## 7. 长任务执行顺序

M0–M5 已完成。下一项长期 goal 建议描述为：

> 完成 M6：将已验证的战斗内信念搜索与策略价值网络扩展到完整 Ironclad A0 run，覆盖奖励、地图、商店、事件、火堆和 Boss relic，并以未见 seed 的整局胜率、层数和置信区间作为最终验收。

执行时先建立 run-level 公开历史和非战斗随机性模型，再扩展课程与训练；不得让搜索直接复制未来奖励、遭遇或路线 RNG。
