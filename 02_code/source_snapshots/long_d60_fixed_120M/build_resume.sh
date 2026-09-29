#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

LOCAL_CUDA_ROOT="$(cd ../../../experiment5_multigel_memory/toolchain/cuda-local/usr 2>/dev/null && pwd || true)"
if [[ -n "${NVCC:-}" ]]; then
  NVCC_BIN="$NVCC"
elif [[ -x "$LOCAL_CUDA_ROOT/lib/nvidia-cuda-toolkit/bin/nvcc" ]]; then
  NVCC_BIN="$LOCAL_CUDA_ROOT/lib/nvidia-cuda-toolkit/bin/nvcc"
else
  NVCC_BIN="nvcc"
fi

if [[ -z "${HOST_CXX:-}" && -x "$LOCAL_CUDA_ROOT/bin/g++-12" ]]; then
  HOST_CXX="$LOCAL_CUDA_ROOT/bin/g++-12"
else
  HOST_CXX="${HOST_CXX:-$(command -v g++ || true)}"
fi

OUT="${OUT:-experiment2_resume}"
ARCH="${CUDA_ARCH:-sm_70}"

EXTRA_FLAGS=()
HOST_FLAGS=()
if [[ -n "$HOST_CXX" ]]; then
  HOST_FLAGS=(-ccbin "$HOST_CXX")
fi
if [[ -n "$LOCAL_CUDA_ROOT" && -d "$LOCAL_CUDA_ROOT/include" ]]; then
  export PATH="$LOCAL_CUDA_ROOT/bin:$LOCAL_CUDA_ROOT/lib/nvidia-cuda-toolkit/bin:$PATH"
  export LD_LIBRARY_PATH="$LOCAL_CUDA_ROOT/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
  EXTRA_FLAGS+=(
    -I"$LOCAL_CUDA_ROOT/include"
    -L"$LOCAL_CUDA_ROOT/lib/x86_64-linux-gnu"
    -L"$LOCAL_CUDA_ROOT/lib/nvidia-cuda-toolkit"
  )
fi

"$NVCC_BIN" \
  -O3 \
  -std=c++14 \
  -arch="$ARCH" \
  -allow-unsupported-compiler \
  "${HOST_FLAGS[@]}" \
  -Xcompiler -pthread \
  "${EXTRA_FLAGS[@]}" \
  -o "$OUT" \
  ligel.cu ligelSystem.cu libottomgridSystem.cu ligel_kernel.cu

echo "built $(pwd)/$OUT"
