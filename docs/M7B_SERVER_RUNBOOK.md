# M7-B 服务器运行手册

## 1. 运行边界

本轮只运行 M7-B seed-17 pilot：收集固定教师语料、纯监督蒸馏、teacher-action gate，并在前者通过后运行端到端 gate。不得启动 seed 29/43，不得访问 `3000000-3002047` 最终盲测。

沿用已有目录和权限，不要求管理员创建新目录：

```bash
ROOT=/scratch/sts_agent
WORKTREE=/scratch/sts_agent/.m7-pilot-worktree
EXPERIMENTS=/scratch/sts_agent/experiments
PYTHON=/scratch/sts_agent/.venv/bin/python
```

原 `/scratch/sts_agent` 工作树及 M6 原始产物保持不变。代码只在现有隔离 worktree 中切换，输出只写入现有 `experiments` 目录。

## 2. 更新隔离 worktree

将 `<M7B_REVISION>` 替换为用户提供的 tag 或 commit：

```bash
cd "$ROOT"
git fetch --tags origin

cd "$WORKTREE"
test -z "$(git status --porcelain)"
git switch --detach <M7B_REVISION>
git rev-parse HEAD
```

如果 worktree 不干净，立即停止并报告，不删除、不 reset、不移动已有文件。

复用之前已建立的原生扩展链接。动态定位扩展并设置环境：

```bash
cd "$WORKTREE"
EXTENSION=$(find build -type f -name 'slaythespire*.so' -print -quit)
test -n "$EXTENSION"
EXTENSION_DIR=$(dirname "$EXTENSION")
export PYTHONPATH="$WORKTREE/src:$EXTENSION_DIR"

"$PYTHON" - <<'PY'
import slaythespire
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY

nvidia-smi
```

CUDA 不可用时停止，不降级到 CPU 正式训练。

## 3. 预检

```bash
cd "$WORKTREE"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -B -m unittest discover -s tests -v

for script in \
  scripts/collect-m7b-teacher-corpus.py \
  scripts/build-m7b-replay-cache.py \
  scripts/train-m7b-distillation.py \
  scripts/evaluate-m7b-imitation.py \
  scripts/summarize-m7b-imitation.py \
  scripts/evaluate-m7.py \
  scripts/summarize-m7-evaluations.py \
  scripts/audit-m7b.py; do
  "$PYTHON" -B "$script" --help >/dev/null
done
```

测试不全通过时停止，不开始语料收集或训练。

## 4. 带宽参数

3090 + 18C/36T 的初始设置：

```bash
CORPUS_WORKERS=28
TORCH_THREADS=6
TORCH_INTEROP_THREADS=1
```

- 语料收集 CPU 占用不足时，将 `CORPUS_WORKERS` 提高到 32；影响桌面响应时降到 16。
- `TORCH_THREADS` 可在 4 到 10 之间调节；它只控制 CPU 辅助算子，不改变训练数据或 loss。
- `trace_batch_size`、`optimizer_batch_chunks` 和 phase multiplier 属于实验语义，固定在 config 中，不为追求占用率临时修改。
- 不并行启动两份 M7-B 训练；本轮只有 seed 17。

## 5. 收集持久教师语料

```bash
TRAIN_CORPUS="$EXPERIMENTS/m7b_teacher_train"
VALID_CORPUS="$EXPERIMENTS/m7b_teacher_validation"

cd "$WORKTREE"
"$PYTHON" -B scripts/collect-m7b-teacher-corpus.py \
  --seed-start 400000 \
  --seed-count 4096 \
  --workers "$CORPUS_WORKERS" \
  --progress-interval 64 \
  --output "$TRAIN_CORPUS"

"$PYTHON" -B scripts/collect-m7b-teacher-corpus.py \
  --seed-start 1500000 \
  --seed-count 512 \
  --workers "$CORPUS_WORKERS" \
  --progress-interval 64 \
  --output "$VALID_CORPUS"
```

每条 trace 先写临时文件再原子替换。命令中断后可原样重跑；已存在的有效 trace 会复用。命令结束必须生成两个 `manifest.json`，且五个 phase count 都大于 0。

将固定语料预编码为可校验 replay batch，避免每个 epoch 重复恢复和编码完整轨迹：

```bash
TRAIN_REPLAY="$EXPERIMENTS/m7b_teacher_train_replay"
VALID_REPLAY="$EXPERIMENTS/m7b_teacher_validation_replay"
M6_CHECKPOINT="$EXPERIMENTS/m6r_server_training/seed-17/best-evaluation-checkpoint.pt"

"$PYTHON" -B scripts/build-m7b-replay-cache.py \
  --corpus "$TRAIN_CORPUS/manifest.json" \
  --checkpoint "$M6_CHECKPOINT" \
  --trace-batch-size 64 \
  --workers "$CORPUS_WORKERS" \
  --output "$TRAIN_REPLAY"

"$PYTHON" -B scripts/build-m7b-replay-cache.py \
  --corpus "$VALID_CORPUS/manifest.json" \
  --checkpoint "$M6_CHECKPOINT" \
  --trace-batch-size 64 \
  --workers "$CORPUS_WORKERS" \
  --output "$VALID_REPLAY"
```

两个 replay manifest 会记录 corpus hash、逐 batch SHA-256 和 aggregate SHA-256。训练或评测读取前会验证这些值；验证失败时停止，不重建或覆盖原缓存。

## 6. 评测 M6 teacher-action 基线

```bash
M7B_RUN="$EXPERIMENTS/m7b_distillation"
M7B_IMITATION="$EXPERIMENTS/m7b_imitation_gate"
mkdir -p "$M7B_RUN" "$M7B_IMITATION"

"$PYTHON" -B scripts/evaluate-m7b-imitation.py \
  --checkpoint "$M6_CHECKPOINT" \
  --corpus "$VALID_CORPUS/manifest.json" \
  --replay-cache "$VALID_REPLAY/manifest.json" \
  --report-label m6-initial \
  --trace-batch-size 64 \
  --optimizer-batch-chunks 16 \
  --device cuda \
  --output "$M7B_IMITATION/m6-initial.json"
```

## 7. 启动训练

```bash
STOP_FILE="$M7B_RUN/STOP"
LOG="$M7B_RUN/train.log"
PID_FILE="$M7B_RUN/train.pid"
test ! -e "$STOP_FILE"

nohup "$PYTHON" -B scripts/train-m7b-distillation.py \
  --config config/m7b_noncombat_distillation.json \
  --run-seed 17 \
  --train-corpus "$TRAIN_CORPUS/manifest.json" \
  --validation-corpus "$VALID_CORPUS/manifest.json" \
  --train-replay-cache "$TRAIN_REPLAY/manifest.json" \
  --validation-replay-cache "$VALID_REPLAY/manifest.json" \
  --initialize-from "$M6_CHECKPOINT" \
  --output "$M7B_RUN" \
  --stop-file "$STOP_FILE" \
  --progress-interval-batches 4 \
  --torch-threads "$TORCH_THREADS" \
  --torch-interop-threads "$TORCH_INTEROP_THREADS" \
  >"$LOG" 2>&1 &
echo $! >"$PID_FILE"
```

监控：

```bash
tail -f "$LOG"
nvidia-smi
tail -n 5 "$M7B_RUN/seed-17/metrics.jsonl"
```

训练正常结束时，日志最后一行 `state` 为 `complete`，并存在：

```text
$M7B_RUN/seed-17/checkpoint.pt
$M7B_RUN/seed-17/best-evaluation-checkpoint.pt
$M7B_RUN/seed-17/best-validation.json
```

## 8. 安全暂停与恢复

优先使用 stop file：

```bash
touch "$STOP_FILE"
tail -f "$LOG"
```

训练会完成当前 trace batch、原子保存 checkpoint，然后以 `state=stopped` 退出。也可向 PID 发送 `SIGTERM`，行为相同：

```bash
kill -TERM "$(cat "$PID_FILE")"
```

恢复前删除 stop file，然后使用同一 config 和语料：

```bash
rm "$STOP_FILE"
nohup "$PYTHON" -B scripts/train-m7b-distillation.py \
  --config config/m7b_noncombat_distillation.json \
  --run-seed 17 \
  --train-corpus "$TRAIN_CORPUS/manifest.json" \
  --validation-corpus "$VALID_CORPUS/manifest.json" \
  --train-replay-cache "$TRAIN_REPLAY/manifest.json" \
  --validation-replay-cache "$VALID_REPLAY/manifest.json" \
  --resume "$M7B_RUN/seed-17/checkpoint.pt" \
  --output "$M7B_RUN" \
  --stop-file "$STOP_FILE" \
  --progress-interval-batches 4 \
  --torch-threads "$TORCH_THREADS" \
  --torch-interop-threads "$TORCH_INTEROP_THREADS" \
  >>"$LOG" 2>&1 &
echo $! >"$PID_FILE"
```

如果进程被 `SIGKILL` 或机器掉电，原有 checkpoint 仍有效，最多重算最后一个未保存 trace batch。

## 9. Teacher-action gate

```bash
M7B_CHECKPOINT="$M7B_RUN/seed-17/best-evaluation-checkpoint.pt"

"$PYTHON" -B scripts/evaluate-m7b-imitation.py \
  --checkpoint "$M7B_CHECKPOINT" \
  --corpus "$VALID_CORPUS/manifest.json" \
  --replay-cache "$VALID_REPLAY/manifest.json" \
  --report-label m7b \
  --trace-batch-size 64 \
  --optimizer-batch-chunks 16 \
  --device cuda \
  --output "$M7B_IMITATION/m7b.json"

"$PYTHON" -B scripts/summarize-m7b-imitation.py \
  --baseline "$M7B_IMITATION/m6-initial.json" \
  --candidate "$M7B_IMITATION/m7b.json" \
  --output "$M7B_IMITATION/summary.json"
```

`summary.json` 的 `verdict` 不是 `PASS` 时立即停止，不运行端到端 gate。

## 10. 端到端 gate

只有 teacher-action gate 通过才执行：

```bash
M7B_GATE="$EXPERIMENTS/m7b_end_to_end_gate"
mkdir -p "$M7B_GATE"

"$PYTHON" -B scripts/evaluate-m7.py \
  --method learned-heuristic \
  --report-label m6-initial \
  --checkpoint "$M6_CHECKPOINT" \
  --seed-start 1600000 \
  --seed-count 512 \
  --policy-seed 17 \
  --bootstrap-samples 10000 \
  --output "$M7B_GATE/m6-initial.json"

"$PYTHON" -B scripts/evaluate-m7.py \
  --method learned-heuristic \
  --report-label m7b \
  --checkpoint "$M7B_CHECKPOINT" \
  --seed-start 1600000 \
  --seed-count 512 \
  --policy-seed 17 \
  --bootstrap-samples 10000 \
  --output "$M7B_GATE/m7b.json"

"$PYTHON" -B scripts/summarize-m7-evaluations.py \
  --evaluation "$M7B_GATE/m6-initial.json" \
  --evaluation "$M7B_GATE/m7b.json" \
  --reference-method m6-initial \
  --bootstrap-samples 10000 \
  --output "$M7B_GATE/summary.json"

"$PYTHON" -B scripts/audit-m7b.py \
  --imitation-summary "$M7B_IMITATION/summary.json" \
  --end-to-end-summary "$M7B_GATE/summary.json" \
  --output "$M7B_GATE/audit.json"
```

`audit.json` 是唯一晋级结论。无论 PASS 或 FAIL，本轮都停止，不自动运行 seed 29/43 或最终盲测。

## 11. 时间预估

依据本地原生模拟器 smoke 和服务器硬件，保守预估：

| 阶段 | 预估耗时 |
|---|---:|
| 4608 条教师轨迹收集 | 20-45 分钟 |
| 两份 replay cache 构建 | 3-10 分钟 |
| 最多 20 epochs 蒸馏 | 18-20 分钟/epoch，通常早停 |
| 两次 teacher-action 评测 | 1-10 分钟 |
| 两份 512 局端到端评测与汇总 | 10-30 分钟 |
| 总计 | 约 1.5-3.5 小时 |

早停可能缩短训练。若实际吞吐明显偏离，先报告每 batch 时间、CPU 利用率和 GPU 利用率，不改变实验配置。

## 12. 结果保留

服务器保留全部原始 corpus。上传 GitHub Release 时只打包：

- 两份 corpus manifest 和 collection summary；
- 两份 replay manifest，不包含约 22 GB replay batch 数据；
- `resolved-config.json`、`manifest.json`、`metrics.jsonl`；
- completion checkpoint 与 best evaluation checkpoint；
- teacher-action baseline/candidate/summary；
- 端到端 baseline/candidate/summary/audit；
- SHA-256 文件。

不要打包逐局训练 corpus、虚拟环境、build 目录或临时文件。
