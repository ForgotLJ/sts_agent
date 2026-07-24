# M6 Ubuntu 服务器训练手册

本文档用于把修订版 M6 从 Windows 工作站迁移到 Ubuntu 22.04 服务器。代码通过 GitHub 同步；暂停 checkpoint、教师语料和实机差分门禁通过 GitHub Release 资产同步。

## 1. 迁移边界

服务器流水线保持正式配置不变：每个 run 仍使用 16 个并行环境、64 步 rollout、固定 seed `17/29/43` 和 5000 次更新。性能调节通过并行运行独立 seed、评估并发数和线程数完成，不会静默修改单个实验的训练分布。

Windows 的 `source-freeze.json` 不能直接在 Linux 使用，因为原生扩展、平台和运行时哈希不同。服务器会重新执行 Python 测试、1000 次前缀恢复、10000 局压力门禁、教师语料验证，并生成 Linux 专属冻结清单。实机 CommunicationMod 差分证据随迁移资产带入。

服务器还会从 Windows update `100` checkpoint 试跑到 update `101`。只有跨平台恢复 smoke 通过，正式训练才会启动。

## 2. 系统准备

首先确认 NVIDIA 驱动可用：

```bash
nvidia-smi
```

当前已知服务器曾出现 `nvidia-smi` 无法连接驱动的问题。在该命令正常显示 RTX 3090 前，不要启动训练。Ubuntu 可先查看推荐驱动：

```bash
sudo ubuntu-drivers devices
sudo apt-get install nvidia-driver-550
sudo reboot
```

驱动版本应根据 `ubuntu-drivers devices` 和所选 PyTorch CUDA wheel 调整，不应仅凭“机器里装有 3090”判断 CUDA 已可用。

安装基础工具：

```bash
sudo apt-get update
sudo apt-get install -y git build-essential cmake ninja-build tmux python3.11 python3.11-venv
```

如果 Ubuntu 软件源没有 Python 3.11，可使用 Conda 创建 Python 3.11 环境，再把其 Python 路径传给引导脚本。

## 3. 准备 GitHub 内容

在 Windows 项目目录生成迁移资产：

```powershell
cd D:\Project\STS\sts_agent
& C:\Users\ForgotLJ\.conda\envs\pytorch_env\python.exe `
  scripts\package-m6-server-assets.py
```

输出为 `dist\m6-server-assets.tar.gz`，旁边同时生成 `.sha256` 文件。代码推送到普通 GitHub 仓库；压缩包和校验文件上传到该仓库的 GitHub Release，不要提交 `experiments/`、`build/` 或整个教师语料目录。

迁移包包含：

- seed `17` 的 update `100` 可恢复 checkpoint；
- 1038 条正式教师轨迹；
- CommunicationMod 零差分门禁；
- Windows 冻结清单，仅用于来源追踪。

当前工程应以 `sts_agent` 作为 Git 仓库根目录：

```powershell
cd D:\Project\STS\sts_agent
git init
git add .
git status --short
git commit -m "Add portable M6 server training pipeline"
git branch -M main
git remote add origin <你的仓库地址>
git push -u origin main
```

`.gitignore` 会排除 `experiments/`、`build/`、`dist/` 和 `server_assets/`。代码推送完成后，在 GitHub Release 页面单独上传：

- `dist\m6-server-assets.tar.gz`；
- `dist\m6-server-assets.tar.gz.sha256`。

## 4. 克隆和构建

```bash
git clone <你的仓库地址> sts-agent
cd sts-agent
```

使用稳定 CUDA wheel 的示例：

```bash
bash scripts/server/bootstrap-ubuntu.sh \
  --python python3.11 \
  --build-jobs 32 \
  --torch-index-url https://download.pytorch.org/whl/cu124
```

该脚本创建 `.venv`、安装训练依赖、以 Release 模式编译 Linux `slaythespire*.so`，执行后端检查，并验证 PyTorch 能看到 GPU。

导入从 GitHub Release 下载的迁移包：

```bash
sha256sum -c m6-server-assets.tar.gz.sha256
.venv/bin/python scripts/import-m6-server-assets.py \
  ~/Downloads/m6-server-assets.tar.gz
```

导入器逐文件验证 SHA-256，并拒绝路径穿越和符号链接。

## 5. 查看资源计划

在真正启动前查看解析后的参数：

```bash
.venv/bin/python scripts/run-m6-server.py plan --profile balanced
```

对于 18 核 36 线程、单张 RTX 3090，建议从以下设置开始：

```bash
.venv/bin/python scripts/run-m6-server.py plan \
  --profile balanced \
  --parallel-train-runs 2 \
  --parallel-evaluations 3 \
  --stress-workers 32 \
  --torch-threads 2 \
  --omp-threads 1
```

主要带宽接口：

| 参数 | 作用 | 建议范围 |
|---|---|---:|
| `--parallel-train-runs` | 同时训练独立正式 seed，不改变单 run 配置 | `1–3` |
| `--parallel-evaluations` | 最终评估并发数 | `2–5` |
| `--stress-workers` | Linux 10000 局压力门禁的环境进程数 | `16–32` |
| `--torch-threads` | 每个训练进程的 PyTorch CPU 线程 | `1–4` |
| `--omp-threads` | 每个进程的 BLAS/OpenMP 线程 | 通常为 `1` |
| `--cuda-visible-devices` | 指定训练 GPU | 单卡为 `0` |

`--profile max` 会在 36 线程机器上尝试三个 seed 同时训练和五路评估。先观察 10–20 分钟；如果出现显存不足、上下文切换过多或单 run 吞吐显著下降，优先把 `--parallel-train-runs` 从 `3` 降到 `2`，而不是修改正式训练配置。

流水线默认拒绝在显存占用达到 500 MiB 的 GPU 上启动，防止误叠加任务。只有明确知道已有占用来源时才使用 `--allow-busy-gpu`。

监控命令：

```bash
watch -n 2 nvidia-smi
htop
.venv/bin/python scripts/run-m6-server.py status
```

## 6. 启动、暂停和恢复

后台启动完整流水线：

```bash
bash scripts/server/launch-m6.sh \
  --session m6-formal \
  --profile balanced \
  --parallel-train-runs 2 \
  --parallel-evaluations 3 \
  --stress-workers 32
```

有 `tmux` 时脚本创建 `m6-formal` 会话；否则使用 `nohup`。日志保存在 `experiments/m6r_server_pipeline/pipeline.log`。

安全暂停：

```bash
.venv/bin/python scripts/run-m6-server.py pause
```

流水线终止当前子进程并保留最近一次 checkpoint。再次运行相同的 `launch-m6.sh` 命令会自动跳过已完成工作，并从各 seed 最近 checkpoint 恢复。

## 7. 自动执行顺序

`all` 流水线依次执行：

1. 检查 `nvidia-smi` 和 PyTorch CUDA；
2. 运行 Linux 测试、恢复和 10000 局压力门禁；
3. 生成 Linux `source-freeze.json`；
4. 完成 Windows-to-Linux update `100 → 101` 恢复 smoke；
5. 并行完成 seed `17/29/43` 的 5000 次更新；
6. 冻结三份最佳 EMA checkpoint；
7. 并行执行 15 组、每组 1024 个未见 seed 的最终评估；
8. 汇总置信区间和配对统计；
9. 运行跨平台 29 项完成审计。

最终必须生成：

- `experiments/m6r_server_gates/checkpoint-freeze.json`；
- `experiments/m6r_server_evaluations/summary.json`；
- `experiments/m6r_server_pipeline/completion-audit.json`。

只有 `completion-audit.json` 的 `status` 为 `complete` 且 `checks_passed == checks_total == 29`，M6 才算正式完成。
