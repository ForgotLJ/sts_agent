#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${STS_PYTHON:-${project_root}/.venv/bin/python}"
session_name="m6-formal"

if [[ "${1:-}" == "--session" ]]; then
    session_name="$2"
    shift 2
fi
if [[ ! -x "${python_bin}" ]]; then
    echo "Python executable not found: ${python_bin}" >&2
    exit 1
fi

log_directory="${project_root}/experiments/m6r_server_pipeline"
mkdir -p "${log_directory}"
command=("${python_bin}" "${project_root}/scripts/run-m6-server.py" all "$@")

if command -v tmux >/dev/null 2>&1; then
    printf -v command_line '%q ' "${command[@]}"
    tmux new-session -d -s "${session_name}" \
        "cd $(printf '%q' "${project_root}") && ${command_line}2>&1 | tee $(printf '%q' "${log_directory}/pipeline.log")"
    echo "Started tmux session: ${session_name}"
    echo "Attach with: tmux attach -t ${session_name}"
else
    nohup "${command[@]}" >"${log_directory}/pipeline.log" 2>&1 &
    echo "$!" >"${log_directory}/launcher.pid"
    echo "Started background process: $!"
fi
