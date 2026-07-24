# M6 训练审计日志

## 2026-07-17：存档解锁状态与实机边界复核

- 新的 AITEST 实机追踪显示 seed `1` 的 Act 1 Boss 为 The Guardian，而默认全解锁 `sts_lightspeed` 为 Slime Boss；原版字节码复核确认 Boss 选择还依赖 `STSSeenBosses`，并非 seed 或地图生成回归。
- AITEST 的 `1_STSSeenBosses` 尚未记录 `GUARDIAN/GHOST/SLIME`，但 `1_STSPlayer` 已记录三名基础角色胜利，因此该存档应使用 `guardian_unseen` Boss 历史并保留已解锁的燃烧精英。
- `GameContext`、`SimulatorBridge` 和 `LightspeedBackend` 现显式建模 `act1_boss_history` 与 `final_act_unlocked`；正式训练默认仍为 `all_seen + final_act_unlocked=true`，实机差分则按 AITEST 的真实存档状态运行。
- 上游 CommunicationMod 1.2.1 未导出 `MapRoomNode.hasEmeraldKey`；本地审计版增加 `burning_elite`，补丁 jar SHA-256 为 `CA07236F3A22E0B051772DFCB1E70ECD0F74FA35B7620F7C90EA8E6ABA2F4317`，原 jar 已单独备份。
- patched CommunicationMod 会话位于 `real_game_traces/communication.jsonl` 的 record `909–918`；对 Talk、四项 Neow 选项和颜色卡奖励共 5 个可行动状态离线重放，字段差异为 `0`，结果位于 `real_game_traces/differential-m6-public-boundary-replayed-20260717.jsonl` 与 `experiments/m6_formal_gates/communication-differential.json`。
- 运行时冻结哈希现覆盖 Python 源码、协议文档、C++/Java 审计源码、构建配置、实际加载的 `.pyd` 和 patched CommunicationMod jar；不再只依赖 Python 文件哈希。
- `teacher-v4` 不再依赖三个 run 目录中的人工复制；新增 `teacher-corpus` 必需门禁，对共享语料逐文件聚合哈希、解析全部 trace、核验训练 seed 范围和语义化 Neow 首动作，正式训练启动时再次校验同一聚合哈希。

## 2026-07-17：Runic Dome 信息边界修复

- 审计发现旧 `SimulatorBridge` 在玩家持有 Runic Dome 时仍导出真实敌人伤害与攻击次数。
- `m6_recurrent_ppo_pilot_v2/seed-17` 的训练轨迹中确实存在 Runic Dome 选择，因此该 checkpoint 仅保留作诊断，不具备最终评估资格。
- C++ 桥接层、Lightspeed Python 后端和 CommunicationMod Python 后端均已增加意图屏蔽；CommunicationMod 上游 Java 转换器原本已正确隐藏这些字段。
- 修复后重新构建扩展，99 项单元、契约、恢复与多进程测试全部通过。

## DAgger 轮数选择

- 旧 pilot 的 update 1100 对照因 checkpoint 存在 Runic Dome 信息泄漏而失去正式配置依据，仅保留在 `experiments/m6_dagger_round_comparison_u1100.json` 作为历史诊断。
- 从清洁的 `m6_frontier_weighted_v1/seed-17` update 900 checkpoint 出发，在此前未用于模型选择的验证 seed `1101024–1101087` 上重做逐轮对照。
- 零轮、第一轮和第二轮后的平均终止楼层分别为 `16.109375`、`15.796875` 和 `17.5`；第二轮在该清洁对照中最好。
- 正式完整运行阶段与课程阶段均执行两轮 DAgger；机器可读结果位于 `experiments/m6_dagger_round_comparison_clean_u925.json`。

## 干净教师语料

- 使用修复后的公开信息环境和训练 seed `60000–61023` 重新收集 teacher-v4，共 `1024` 条轨迹。
- Act 1 通过 `577/1024`，Act 2 通过 `14/1024`，完整胜利 `0/1024`，无收集错误。
- 14 条 Act 2 成功轨迹全部通过 `runicdome` 动作审计，未发现禁用信息轨迹。
- 语料位于 `experiments/m6_recurrent_ppo_formal_v1/seed-17/curriculum/teacher-v4`。

## 工程门槛

- 修复后随机前缀恢复：`1000/1000` 通过，结果位于 `experiments/m6_recurrent_ppo_formal_v1/gates/prefix-recovery.json`。
- 全量 Python 测试：`99/99` 通过。
- `m6_recurrent_ppo_formal_v1` 在 update 75 后因训练期间仍补充了纯评估源码而主动停止；其源码哈希不作为最终训练哈希。
- 后续正式 run 必须在源码冻结后从零启动，并使用 run seed `17`、`29`、`43`。

## 前沿自模仿诊断

- 干净的 `m6_recurrent_ppo_formal_v2/seed-17` 在 update 425 前未产生完整胜利；后续诊断运行保留其公开信息边界，但不作为最终冻结模型。
- `m6_frontier_diagnostic_v1/seed-17` 首次在完整开局训练中收集到 Act 3、floor 46 的候选轨迹，证明将失败前最后动作裁掉的前沿自模仿机制能够推进探索边界。
- 未加权候选语料达到 64 条，其中仅 1 条到达 floor 46；在 14 条教师轨迹存在时，每轮会选入约 50 条候选，最佳前沿轨迹因而被明显稀释。
- update 750 验证仍无完整胜利；当时保存的 64 局摘要只包含完成率、平均楼层、长度和胜率，没有逐局 Act 2 clear 分类，因此不再保留无法由原始指标直接复核的 `1/64` 叙述。

## 前沿加权诊断

- 新配置将候选来源上限设为 16，并让排名第一的公开进度轨迹获得 4 次训练曝光；其余候选和 14 条教师轨迹继续保留，以平衡利用与多样性。
- 新增单元测试证明只重复最高进度轨迹；完整 Python 测试为 `101/101` 通过，AST 语法检查通过。当前环境未安装 `ruff`，因此未执行该可选检查。
- `m6_frontier_weighted_v1/seed-17` 从干净的 update 750 checkpoint 恢复，仅复制截至 update 750 的 62 条候选，排除未保存的 update 756 与 759 轨迹。
- 实际每轮自模仿使用 33 次轨迹曝光：floor 46 前沿 4 次、其余候选 15 次、教师轨迹 14 次；诊断计划在 update 900 自动保存并停止。

## 前沿加权结果与评估协议修复

- `m6_frontier_weighted_v1/seed-17` 按计划在 update 900 保存停止；64 局固定验证的平均终止楼层在 update 850/875 达到 `19.453125/19.34375`，update 900 为 `18.140625`，均无验证集内胜利。
- 审计发现旧 `evaluate-m6.py --method learned` 让循环网络直接执行战斗，但训练收集器对战斗步骤设置 PPO policy weight 为 0，并委托启发式战斗策略执行。因此旧的纯网络诊断结果不能代表训练期策略。
- 新增显式 `learned-heuristic` 模式，表示循环网络负责非战斗、启发式负责战斗；`learned` 继续保留为纯循环网络弱基线，`learned-search` 表示分层搜索智能体。
- 在相同验证 seed `1100000–1101023` 上，update 750 的正确组合策略平均 floor 为 `18.34765625`、Act 2 clear 为 `11/1024`、胜利 `0/1024`；update 900 平均 floor 为 `17.62109375`、Act 2 clear 为 `8/1024`、胜利 `1/1024`。
- update 900 相对 750 的配对 floor 差为 `-0.7265625`，bootstrap 95% 区间 `[-1.2890625, -0.1669921875]`。因此加权在线网络产生了首次完整胜利，但主体性能存在显著遗忘，不能直接宣称整体提升。

## EMA 稳定化证据

- 对同一路径上的 update 750 与 900 网络参数进行验证专用线性插值；候选权重 `0.25` 的 checkpoint 在 1024 个相同验证 seed 上达到平均 floor `18.4638671875`、Act 1 clear `463/1024`、Act 2 clear `16/1024`、完整胜利 `1/1024`，且无超时或循环。
- 25% 插值相对 update 750 的配对 floor 差为 `+0.1162109375`，95% 区间 `[-0.3154296875, 0.5595703125]`：未证明均值显著提高，但排除了 update 900 的显著退化，同时保留了胜利尾部。
- 正式训练实现采用 full-run 阶段参数 EMA，衰减率 `0.998`；150 个外层更新的累计新参数质量约为 `1 - 0.998^150 = 25.9%`，与诊断中表现最好的 25% 插值相对应。在线网络继续探索，EMA 网络用于验证和最终冻结，课程晋级时重置。
- checkpoint 现同时保存在线参数与 EMA 状态，并额外生成不可恢复训练的 `evaluation-checkpoint.pt`；CPU 两更新冒烟训练成功，完整 Python 测试为 `102/102` 通过。

## 2026-07-20：正式门槛与源码冻结

- 正式 Python 门槛为 `112/112` 通过；随机前缀恢复为 `1000/1000` 通过；CommunicationMod 差分覆盖 5 个可行动状态且字段差异为 `0`。
- 长时稳定性门槛完成 `10001` 个 episode，错误数为 `0`；运行设备为 CUDA，8 个 worker 共执行 `1887` 次采样更新。
- teacher-v4 聚合门槛验证 `1038` 条有效轨迹，其中 Act 1 轨迹 `1024` 条、Act 2 轨迹 `14` 条；聚合 SHA-256 为 `2a51c71573770f849c5377a52141aace363103e3ecda7f1208049730f481ad46`。
- 正式源码冻结 SHA-256 为 `2b7a4f86937aa41a269ba2915169ed156f9b5d357dc3572a7fdf90f23d92ecf5`；机器可读清单位于 `experiments/m6_formal_gates/source-freeze.json`。
- 正式训练目录固定为 `experiments/m6_recurrent_ppo_formal_final_v1`，run seed 固定为 `17`、`29`、`43`，每个 run 目标为 `5000` 次更新。

## 2026-07-20：训练暂停与恢复审计

- Seed `17` 已完成 `5000/5000` 次更新；最佳 EMA 位于 update `4275`，64 局验证平均终止楼层为 `19.328125`，完整胜率为 `0`。
- Seed `29` 首次暂停前最新指标为 update `1674`，最近有效 checkpoint 为 update `1650`。紧急暂停丢弃 checkpoint 后尚未保存的 24 次更新，未修改任何已冻结参数。
- `pause-pipeline.ps1` 在终止进程前使用正式加载器验证 checkpoint 可恢复；`-WaitForNextCheckpoint` 支持零更新损失暂停，默认即时暂停最多丢弃 24 次未保存更新。
- `resume-pipeline.ps1` 会跳过已完成的 Seed `17`，从 Seed `29` 的 checkpoint 恢复优化器、EMA、采样数据流和随机数状态，并将流水线限制在 12 个逻辑 CPU、`BelowNormal` 优先级运行。
- 恢复后的指标文件会截断到 checkpoint 对应位置再继续追加，因此被丢弃的 update `1651–1674` 不会作为有效恢复历史保留；训练已重新越过该区间。
- 暂停恢复说明位于 `experiments/m6_formal_pipeline/PAUSE_AND_RESUME.md`，机器状态分别记录于 `status.json`、`pause-state.json` 和 `events.jsonl`。

## 2026-07-20：最终完成条件审计

- 正式流水线的 learned 胜利数量现从逐局 `summary.episodes[].won` 统计，不再读取不存在的 `summary.wins` 字段。
- 若 `learned` 与 `learned-search` 的全部最终未见测试均没有完整 A0 胜利，流水线必须失败，不能写入 `pipeline_complete`。
- 新增 `experiments/m6_formal_pipeline/verify-completion.ps1`，机器化检查正式门槛、三次训练、EMA checkpoint 冻结、15 份最终评估、1024 个精确未见 seed、Wilson/bootstrap 区间、配对统计、至少一次 learned A0 胜利和最终流水线状态。
- 当前完成审计为 `7/29` 项通过；其余项目明确等待 Seed `29/43`、checkpoint 冻结和最终未见评估产生，不将训练中状态误报为 M6 完成。
- 最终评估调度改为最多 3 个独立进程并行；每个进程仍使用固定方法、policy seed、冻结 checkpoint 和 `2000000–2001023` seed 区间，不改变统计协议。开发 seed 并发冒烟 `3/3` 通过且错误数为 `0`，结果位于 `experiments/m6_formal_pipeline/parallel-evaluation-smoke`。

## 2026-07-20：checkpoint 文件竞争恢复

- 在等待 Seed `29` 的 update `1925` checkpoint 时，旧暂停控制器每 5 秒完整加载一次 `checkpoint.pt`；Windows 上该读取与训练进程的 `checkpoint.pt.tmp -> checkpoint.pt` 原子替换发生竞争，导致训练在临时文件完整写出后替换失败。
- 故障现场同时保留了可加载的正式 checkpoint update `1900` 和临时 checkpoint update `1925`。二者均通过正式 `load_m6_checkpoint` 验证，run seed、课程阶段和源码哈希一致。
- 原 update `1900` checkpoint 保存在 `checkpoint.pt.backup-1900.pt`；已验证的临时 checkpoint 被提升为正式 `checkpoint.pt`，训练从 update `1925` 恢复，不丢失已完成的 DAgger、验证、优化器、EMA、采样器或 RNG 状态。
- `pause-pipeline.ps1 -WaitForNextCheckpoint` 现只轮询文件长度、修改时间和 `.tmp` 是否存在；仅在正式 checkpoint 元数据稳定变化后加载一次内容，避免再次锁住正在被替换的文件。
- 恢复后的流水线已加载三进程最终评估调度和 29 项自动完成审计；Seed `29` 已越过恢复点继续训练。

## 2026-07-20：运行时性能调整

- 20 秒实时剖析显示 12 核限制下系统 CPU 平均占用 `16.7%`、训练进程约使用 `1.37` 个等效核心、GPU 平均占用 `9.85%`；显存、系统内存、功耗和温度均有充足余量。
- 性能瓶颈主要位于单进程验证和 DAgger 路径，环境 worker 在该阶段大多等待。显著进一步提速需要修改冻结源码或正式训练协议，因此不采用。
- 在不改变训练语义的范围内，流水线及全部训练进程已从 12 核 `BelowNormal` 调整为全部 20 核 `Normal`；调整后采样中主训练进程从约 `1.37` 提高到 `1.51` 个等效核心，GPU 平均占用约 `14.25%`。
- 后续恢复脚本默认沿用 20 核与 `Normal` 优先级；性能采样和完整说明位于 `profile_output/M6_LIVE_PROFILE_20260720.md`。

## 2026-07-20：CUDA 优化器失速恢复

- Seed `29` 在 update `2456` 后超过 100 分钟没有写入新指标；训练主进程仍持续占用约一个 CPU 核，worker 空闲，GPU 仅约 `3%`，因此不是正常验证、系统空闲或进程退出。
- `py-spy --native` 调用栈显示主线程停在 PyTorch Adam multi-tensor 更新的 CUDA `item()/stream synchronize` 路径，位置为 `recurrent_ppo.py:_update_minibatch -> optimizer.step`；调用栈保存在 `profile_output/m6-stall-update2457-pyspy.txt`。
- 流水线从有效 checkpoint update `2450` 重启 CUDA 上下文，丢弃 6 次未保存更新；恢复后 update `2451–2455` 重新正常完成，并成功越过原失速位置。
- 新增独立 `watchdog-pipeline.ps1`：仅当训练阶段连续 45 分钟没有新指标时触发，先保存 `py-spy` 调用栈，再验证 checkpoint、重启 CUDA 上下文并原位恢复；正常 DAgger/验证窗口低于该阈值。
- watchdog PID、stdout、stderr 和事件均保存在 `experiments/m6_formal_pipeline`，用户主动暂停时状态不再是训练阶段，因此不会被自动恢复。

## 2026-07-20：Seed 29 首个完整训练胜利

- 正式 Seed `29` 在 update `2582` 的完整运行训练中首次产生 A0 胜利；环境 seed 为 `10866`，属于预先冻结的训练区间 `0–999999`。
- 轨迹从 Neow 开始，共执行 `765` 个动作，在 floor `51` 以 Bludgeon 击败 Time Eater；终局为 `player_victory`，终止奖励 `+1`，剩余生命 `15`，非超时或截断。
- 轨迹位于 `experiments/m6_recurrent_ppo_formal_final_v1/seed-29/curriculum/candidates/full_run/act-3-floor-51-u002582-e13-s10866.jsonl`，SHA-256 为 `6e0b2204d861183e3aa43412a8f5a9a53425a4d4284fb99eb32409334d21be2f`。
- 使用冻结环境从 seed 与动作前缀完整重放，所有逐步 observation digest 均通过，最终公开状态再次到达 Act 3 floor `51`、HP `15`；机器可读证据为 `experiments/m6_formal_pipeline/seed-29-first-training-win.json`。
- 该结果证明正式训练策略已在训练分布中发现完整胜利，但不替代最终未见 seed 评估；M6 完成仍要求冻结 checkpoint 后在 `2000000–2001023` 上至少出现一次 learned 完整胜利。
