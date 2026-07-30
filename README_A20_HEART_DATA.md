# A20 Heart 数据格式化

`scripts/format_sts_runs.py` 将 `Data` 目录中的 Slay the Spire run-history JSON 分片转换为可流式读取的 JSONL，并生成 A20 与 A20 Heart 子集。

## 运行

```powershell
python scripts/format_sts_runs.py `
  --input-dir D:\Project\STS\Data `
  --output-dir D:\Project\STS\Data\formatted_a20h
```

只保留正常 A20 及 A20 Heart 输出，避免生成 A0-A19 和其他模式的废数据：

```powershell
python scripts/format_sts_runs.py `
  --input-dir D:\Project\STS\Data `
  --output-dir D:\Project\STS\Data\formatted_a20h_a20_only `
  --a20-only
```

处理过程按文件和记录流式写出，不会把所有分片累积到内存中。

默认 Heart 判定阈值为 `floor_reached >= 57`。可以通过 `--heart-min-floor` 覆盖，但应在报告中记录对应数据版本的楼层语义。

## 输出文件

- `runs_normalized.jsonl`：所有去重后的标准化记录
- `normal_runs.jsonl`：排除 trial/daily/beta/endless 后的正常模式记录
- `a20_runs.jsonl`：正常 A20 记录
- `a20_heart_wins.jsonl`：正常 A20 Heart 胜利记录
- `normal_runs_<CHARACTER>.jsonl`：按角色拆分的正常模式记录
- `a20_runs_<CHARACTER>.jsonl`：按角色拆分的 A20 记录
- `a20_heart_wins_<CHARACTER>.jsonl`：按角色拆分的 A20 Heart 胜利记录
- `summary.json`：计数、角色分布和输入文件清单
- `REPORT.md`：人类可读报告

## 当前十个分片结果

- 输入记录：10,421
- 正常模式记录：6,813
- 正常 A20：1,110
- 正常 A20 胜利：49
- A20 Heart 胜利：34

Heart 判定使用 `victory == true`、`killed_by` 为空、`floor_reached >= 57`。源数据没有专门的 `heart_defeated` 字段；`killed_by == "The Heart"` 表示输给心脏，永远不会被标记为胜利。

这些文件是 run-history 汇总数据，包含牌组、遗物、路径、奖励、事件、伤害和结果，但不包含逐步战斗 observation/action。因此它们适合结果条件分析、策略统计和样本筛选，不能单独作为底层动作模仿数据集。
