# `sts_lightspeed` 本地审计补丁

## 基线

- 上游提交：`7476a81954020087da31d41d16fddf475746ec2d`
- 原始 `pybind11` 提交：`787d2c88cafa4d07fb38c9519c485a86323cfcf4`
- 本地 `pybind11`：`v2.13.6`（`a2e59f0e7065404b44dfe92a28aca47ba1378dc4`）
- 目标工具链：MSVC v143、CMake、Ninja、Python 3.11

## 构建兼容

- 为 MSVC 使用 `/O2 /permissive-`，并按编译器区分原有 GCC 参数。
- 用 `std::vector` 替代 `Map.cpp` 中的变长数组。
- 修正 `fixed_list` 将 `std::array` 迭代器误当作裸指针的写法。
- 补充 `SimpleAgent.cpp` 缺失的 `<chrono>`。
- 将实际定义为 `struct` 的类型统一使用 `struct` 前向声明，消除 MSVC 的 class-key 符号修饰不一致。
- 修正领取与蓝钥匙绑定的宝箱遗物后未清除 `sapphireKey` 的顺序错误；先记录绑定关系，再移除遗物奖励。
- 修正中文 MSVC `/showIncludes` 前缀被 CMake 错误转码的问题，使 Ninja 能记录头文件依赖。
- 将核心源码编译为单一 `sts_lightspeed_core` 静态库，避免 Python 模块和测试驱动重复编译全部源码。

## 未定义行为与确定性

- 将遗物模板位图改为 `if constexpr`，并对 128 位容量增加编译期范围检查。
- 修正玩家 `justApplied` 位图的越界实例化。
- 修正怪物 `justApplied` 未初始化掩码。
- 为运行时遗物查询增加越界保护。
- 对 `ScreenStateInfo` 使用值初始化，消除同 seed 初始 `encounter` 随栈内容变化的问题。
- 修正药水丢弃动作的有符号编码判断和描述打印。
- 为复制牌建立“内部 ID + 手牌槽位”的公开复合实例 ID，保证单个观察内动作引用唯一。

## 训练桥接

- 新增 `SimulatorBridge`，封装 `GameContext`、可选 `BattleContext`、合法动作执行和深拷贝克隆。
- 克隆时深拷贝 `Map`，不共享可变地图对象。
- 新增战斗状态完整合法动作枚举，包括逐牌实例、逐目标、药水使用、药水丢弃、选牌和结束回合。
- 公开观察只包含当前可见状态、牌堆计数和动态动作，不导出 RNG、RNG counter 或抽牌顺序。
- 修正 `NNInterface.observation_space_size` 的 pybind getter 参数。
- 新增战斗内 `redeterminized_clone`：重新洗牌未公开 draw pile，并用独立 `search_seed` 重建未来战斗 RNG；
- 重采样支持显式已知置顶/置底卡牌约束，且强制重采样前后公开观察完全一致；
- 重采样拒绝非战斗状态，防止搜索继续读取未来奖励、遭遇和路线隐藏状态。

## 构建与验证

```powershell
.\scripts\configure-lightspeed.cmd
.\scripts\build-lightspeed.cmd 2
.\scripts\check-lightspeed.cmd
.\scripts\fuzz-lightspeed.cmd
.\scripts\benchmark-lightspeed.cmd
```

2026-07-16 的本机验证结果：

- MSVC Release 构建成功，日志无 `warning Cxxxx`、编译错误或链接错误。
- 100 个独立 Python 进程连续导入成功。
- 10,000 个固定 seed 双实例公开状态完全一致。
- 原生 `test.exe simple_agent_mt 1 0 1` 成功结束。
- Python 单元与契约测试共 14 项通过。
- 100,000 个随机合法动作、1,471 局、每 1,000 步克隆复放通过。
- fuzz 吞吐约 `19,470 steps/s`，稳定工作集增长约 `0.62 MiB`。
- 正式基准的 step 中位延迟约 `38.5 us`，clone 中位延迟约 `1.7 us`。

四线程 Python 采样受 GIL 影响，吞吐低于单线程。当前单进程约 16k–21k steps/s，M4 collector 应优先采用多进程 actor，而不是共享解释器线程。

## M6 信息边界补丁

- `SimulatorBridge` 在玩家持有 Runic Dome 时将所有敌人的 `intent_damage` 与 `intent_hits` 置零，与真实游戏界面和 CommunicationMod 的公开信息保持一致。
- Python 的 Lightspeed 与 CommunicationMod 后端增加同样的防御性屏蔽，避免旧扩展或异常桥接数据绕过公开信息边界。
- 对应回归测试覆盖存活敌人的隐藏意图，2026-07-17 全量测试共 99 项通过。
