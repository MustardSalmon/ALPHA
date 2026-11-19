#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${repo_dir}/build"
cuda_compiler="${CUDACXX:-${CUDA_HOME:-/usr/local/cuda-12.1}/bin/nvcc}"

cmake -S "${repo_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="${cuda_compiler}"
cmake --build "${build_dir}" --parallel "${BUILD_JOBS:-$(nproc)}"

echo "Executable: ${build_dir}/alpha_2d"
