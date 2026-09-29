#!/usr/bin/env bash
# Relax the four example structures on one already-allocated GPU.
set -euo pipefail
lesson_root=$1
sif=$2
model=$3

exec apptainer exec --cleanenv --nv \
    --env TORCH_COMPILE_DISABLE=1 --env TORCH_DISABLE_NATIVE_JIT=1 \
    --bind "$model:/models/mace.model:ro" \
    --bind "$lesson_root/examples/alchemi_relax.py:/opt/mlip/alchemi_relax.py:ro" \
    --bind "$lesson_root/examples/starts:/opt/mlip/starts:ro" \
    "$sif" python /opt/mlip/alchemi_relax.py
