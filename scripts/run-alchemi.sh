#!/usr/bin/env bash
# Run the readable silicon example on one already-allocated GPU.
set -euo pipefail
lesson_root=$1
sif=$2
model=$3
replicas=${4:-8}
steps=${5:-2000}
cells=${6:-2}

exec apptainer exec --cleanenv --nv \
    --env TORCH_COMPILE_DISABLE=1 --env TORCH_DISABLE_NATIVE_JIT=1 \
    --env MLIP_REPLICAS="$replicas" --env MLIP_STEPS="$steps" \
    --env MLIP_CELLS="$cells" \
    --bind "$model:/models/mace.model:ro" \
    --bind "$lesson_root/examples/alchemi_si_one_cell.py:/opt/mlip/alchemi_si.py:ro" \
    "$sif" python /opt/mlip/alchemi_si.py
