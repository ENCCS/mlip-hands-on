#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?run inside a GPU allocation}"
: "${MLIP_MODEL:?set the original MACE checkpoint path}"
: "${MLIP_ALCHEMI_SIF:?set the ALCHEMI SIF path}"
test -f "$MLIP_MODEL" && test -f "$MLIP_ALCHEMI_SIF"
here=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
exec apptainer exec --cleanenv --nv \
  --env TORCH_COMPILE_DISABLE=1 --env TORCH_DISABLE_NATIVE_JIT=1 \
  --bind "$MLIP_MODEL:/models/mace.model:ro" \
  --bind "$here/examples/alchemi_si.py:/opt/mlip/alchemi_si.py:ro" \
  "$MLIP_ALCHEMI_SIF" python /opt/mlip/alchemi_si.py \
  --model /models/mace.model "$@"
