#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_command="python3.11"
venv_path="${project_root}/.venv"
build_jobs="$(nproc)"
torch_index_url=""
skip_python_deps=0
allow_no_cuda=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python) python_command="$2"; shift 2 ;;
        --venv) venv_path="$2"; shift 2 ;;
        --build-jobs) build_jobs="$2"; shift 2 ;;
        --torch-index-url) torch_index_url="$2"; shift 2 ;;
        --skip-python-deps) skip_python_deps=1; shift ;;
        --allow-no-cuda) allow_no_cuda=1; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

missing=()
for command_name in cmake ninja g++ git; do
    command -v "${command_name}" >/dev/null 2>&1 || missing+=("${command_name}")
done
if (( ${#missing[@]} > 0 )); then
    echo "Missing build tools: ${missing[*]}" >&2
    echo "Install them with: sudo apt-get install -y build-essential cmake ninja-build git" >&2
    exit 1
fi
if ! command -v "${python_command}" >/dev/null 2>&1; then
    echo "Python 3.11+ was not found: ${python_command}" >&2
    exit 1
fi

python_version="$(${python_command} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${python_version}" != "3.11" ]]; then
    echo "M6 transfer checkpoints require Python 3.11; found ${python_version}" >&2
    exit 1
fi

if [[ ! -x "${venv_path}/bin/python" ]]; then
    "${python_command}" -m venv "${venv_path}"
fi
python_bin="${venv_path}/bin/python"

if (( skip_python_deps == 0 )); then
    "${python_bin}" -m pip install --upgrade pip setuptools wheel
    if [[ -n "${torch_index_url}" ]]; then
        "${python_bin}" -m pip install torch --index-url "${torch_index_url}"
    fi
    "${python_bin}" -m pip install -e "${project_root}[training]"
fi

STS_PYTHON="${python_bin}" STS_BUILD_JOBS="${build_jobs}" \
    bash "${project_root}/scripts/server/build-lightspeed.sh"

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    if (( allow_no_cuda == 0 )); then
        echo "nvidia-smi is unavailable. Repair the NVIDIA driver before formal training." >&2
        exit 1
    fi
fi

PYTHONPATH="${project_root}/src" "${python_bin}" - <<'PY'
import json
import torch

print(json.dumps({
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}, sort_keys=True))
PY

echo "Bootstrap complete. Python: ${python_bin}"
