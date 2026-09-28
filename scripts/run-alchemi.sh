#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?run inside a GPU allocation}"
: "${MLIP_MODEL:?set the original MACE checkpoint path}"
test -f "$MLIP_MODEL"
here=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
if [[ "${MLIP_ALCHEMI_RUNNER:-container}" == native ]]; then
  : "${MLIP_NATIVE_ALCHEMI_PYTHON:?set the reviewed native environment Python}"
  test -x "$MLIP_NATIVE_ALCHEMI_PYTHON"
  export TORCH_COMPILE_DISABLE=1 TORCH_DISABLE_NATIVE_JIT=1 PYTHONNOUSERSITE=1
  exec "$MLIP_NATIVE_ALCHEMI_PYTHON" "$here/examples/alchemi_si.py" \
    --model "$MLIP_MODEL" "$@"
fi
[[ "${MLIP_ALCHEMI_RUNNER:-container}" == container ]] || {
  echo 'MLIP_ALCHEMI_RUNNER must be container or native' >&2; exit 2;
}
: "${MLIP_ALCHEMI_SIF:?set the ALCHEMI SIF path}"
test -f "$MLIP_ALCHEMI_SIF"
exec apptainer exec --cleanenv --nv \
  --env TORCH_COMPILE_DISABLE=1 --env TORCH_DISABLE_NATIVE_JIT=1 \
  --bind "$MLIP_MODEL:/models/mace.model:ro" \
  --bind "$here/examples/alchemi_si.py:/opt/mlip/alchemi_si.py:ro" \
  "$MLIP_ALCHEMI_SIF" python /opt/mlip/alchemi_si.py \
  --model /models/mace.model "$@"
