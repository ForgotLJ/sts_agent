# M7 本地工程与服务器前检查

## 1. 本地测试

PowerShell：

```powershell
cd D:\Project\STS\sts_agent
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = 'src'
$python = 'C:\Users\ForgotLJ\.conda\envs\pytorch_env\python.exe'

& $python -B -m unittest discover -s tests -v
```

## 2. CPU Smoke

Smoke 直接从 `full_run` 开始，使用 2 个环境、2 次 full-run 更新和极小验证集。它只验证训练、EMA、selection、checkpoint 和恢复链路，不产生性能结论。

```powershell
& $python -B scripts\train-m7.py `
  --config config\m7_recurrent_ppo_pilot.json `
  --run-seed 17 `
  --output experiments\m7_cpu_smoke `
  --smoke
```

暂停/恢复 smoke：

```powershell
& $python -B scripts\train-m7.py `
  --config config\m7_recurrent_ppo_pilot.json `
  --run-seed 17 `
  --output experiments\m7_resume_smoke `
  --stop-after-update 1 `
  --smoke

& $python -B scripts\train-m7.py `
  --config config\m7_recurrent_ppo_pilot.json `
  --run-seed 17 `
  --output experiments\m7_resume_smoke `
  --resume experiments\m7_resume_smoke\seed-17\checkpoint.pt `
  --smoke
```

## 3. M6 组件诊断

在服务器上对三份 M6 EMA checkpoint 补跑 `learned-heuristic`：

```bash
.venv/bin/python scripts/run-m7-component-diagnostics.py \
  --checkpoint 17=/scratch/sts_agent/experiments/m6r_server_training/seed-17/best-evaluation-checkpoint.pt \
  --checkpoint 29=/scratch/sts_agent/experiments/m6r_server_training/seed-29/best-evaluation-checkpoint.pt \
  --checkpoint 43=/scratch/sts_agent/experiments/m6r_server_training/seed-43/best-evaluation-checkpoint.pt \
  --existing-evaluation-root /scratch/sts_agent/experiments/m6r_server_evaluations \
  --output /scratch/sts_agent/experiments/m7_component_diagnostics
```

该命令只补齐已冻结模型的诊断，不训练、不改写 M6 产物。

## 4. 搜索基准

先在开发 seed 上抽取战斗起点，再独立运行各方法：

```bash
.venv/bin/python scripts/collect-m7-combat-corpus.py \
  --seed-start 2100000 \
  --seed-count 20000 \
  --per-act 1000 \
  --output /scratch/sts_agent/experiments/m7_combat_corpus_dev

for method in heuristic search-16 search-64 search-256; do
  .venv/bin/python scripts/benchmark-m7-combat.py \
    --corpus /scratch/sts_agent/experiments/m7_combat_corpus_dev \
    --method "$method" \
    --output "/scratch/sts_agent/experiments/m7_combat_benchmark/${method}.json"
done

.venv/bin/python scripts/summarize-m7-combat.py \
  --baseline /scratch/sts_agent/experiments/m7_combat_benchmark/heuristic.json \
  --candidate /scratch/sts_agent/experiments/m7_combat_benchmark/search-16.json \
  --candidate /scratch/sts_agent/experiments/m7_combat_benchmark/search-64.json \
  --candidate /scratch/sts_agent/experiments/m7_combat_benchmark/search-256.json \
  --output /scratch/sts_agent/experiments/m7_combat_benchmark/summary.json
```

## 5. Pilot 训练

Control 与 balanced 使用相同 seed、初始化和训练预算。建议先只运行 seed 17：

```bash
.venv/bin/python scripts/train-m7.py \
  --config config/m7_recurrent_ppo_control.json \
  --run-seed 17 \
  --initialize-from /scratch/sts_agent/experiments/m6r_server_training/seed-17/best-evaluation-checkpoint.pt \
  --output /scratch/sts_agent/experiments/m7_control_pilot

.venv/bin/python scripts/train-m7.py \
  --config config/m7_recurrent_ppo_pilot.json \
  --run-seed 17 \
  --initialize-from /scratch/sts_agent/experiments/m6r_server_training/seed-17/best-evaluation-checkpoint.pt \
  --output /scratch/sts_agent/experiments/m7_balanced_pilot
```

正式配置 `config/m7_recurrent_ppo.json` 目前只是候选，不得在 pilot 结果审计前冻结或启动三种子训练。

## 6. Pilot 独立评测

两个 pilot 都结束后，只评测各自的最佳 EMA checkpoint。`1400000-1400511` 不参与训练、screening 或 checkpoint selection：

```bash
mkdir -p /scratch/sts_agent/experiments/m7_pilot_gate

.venv/bin/python scripts/evaluate-m7.py \
  --method learned-heuristic \
  --report-label m7-control \
  --checkpoint /scratch/sts_agent/experiments/m7_control_pilot/seed-17/best-evaluation-checkpoint.pt \
  --seed-start 1400000 \
  --seed-count 512 \
  --policy-seed 17 \
  --output /scratch/sts_agent/experiments/m7_pilot_gate/control.json

.venv/bin/python scripts/evaluate-m7.py \
  --method learned-heuristic \
  --report-label m7-balanced \
  --checkpoint /scratch/sts_agent/experiments/m7_balanced_pilot/seed-17/best-evaluation-checkpoint.pt \
  --seed-start 1400000 \
  --seed-count 512 \
  --policy-seed 17 \
  --output /scratch/sts_agent/experiments/m7_pilot_gate/balanced.json

.venv/bin/python scripts/summarize-m7-evaluations.py \
  --evaluation /scratch/sts_agent/experiments/m7_pilot_gate/control.json \
  --evaluation /scratch/sts_agent/experiments/m7_pilot_gate/balanced.json \
  --reference-method m7-control \
  --output /scratch/sts_agent/experiments/m7_pilot_gate/summary.json
```

只有 `m7-balanced_minus_m7-control` 的 `final_floor` 两层 bootstrap 95% 区间下界大于 0、`act1_clear` 平均差不小于 0，且两份评测的错误计数都为 0，才继续扩大训练。否则先停在 pilot 审计，不启动正式三种子任务。
