#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${STS_PYTHON:-${project_root}/.venv/bin/python}"
build_jobs="${STS_BUILD_JOBS:-$(nproc)}"
build_dir="${project_root}/build/sts_lightspeed-py311"

if [[ ! -x "${python_bin}" ]]; then
    echo "Python executable not found: ${python_bin}" >&2
    exit 1
fi

cmake \
    -S "${project_root}/vendor/sts_lightspeed" \
    -B "${build_dir}" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE="${python_bin}" \
    -DSTS_LIGHTSPEED_BUILD_TEST_DRIVER=OFF
cmake --build "${build_dir}" --parallel "${build_jobs}"

PYTHONPATH="${project_root}/src" "${python_bin}" \
    "${project_root}/scripts/check-lightspeed.py"
