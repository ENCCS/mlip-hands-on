#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?run inside a one-GPU allocation}"
mode=${1:?use sequential, plain or mps}
shift
case "$mode" in sequential|plain|mps) ;; *) echo 'use sequential, plain or mps' >&2; exit 2;; esac
here=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
if test "$mode" = mps; then
  command -v nvidia-cuda-mps-control >/dev/null
  mps_root=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/mlip-mps-${SLURM_JOB_ID}.XXXXXX")
  export CUDA_MPS_PIPE_DIRECTORY="$mps_root/pipe"
  export CUDA_MPS_LOG_DIRECTORY="$mps_root/log"
  mkdir -m 700 "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  trap 'printf "quit\n" | nvidia-cuda-mps-control >/dev/null || true' EXIT
  nvidia-cuda-mps-control -d
fi
python3 "$here/examples/lammps_group.py" --mode "$mode" "$@"
