#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?run inside a GPU allocation}"
: "${MLIP_NATIVE_PREFIX:?set the extracted MPI-enabled LAMMPS runtime root}"
: "${MLIP_MLIAP_MODEL:?set the pinned ML-IAP model export path}"
test -f "$MLIP_MLIAP_MODEL"
test -x "$MLIP_NATIVE_PREFIX/python/bin/python3"
test -x "$MLIP_NATIVE_PREFIX/mpi-prefix/bin/lmp"
here=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
source /software/sse2/init/hpc_init_sse.sh >/dev/null 2>&1
module purge >/dev/null 2>&1
module load GPU/buildenv-gcccuda/2026.03-cu13.0 >/dev/null 2>&1
gcc_runtime=/software/sse2/el9_gh200/manual/GCC/14.3.0/hpc1/lib64
export PATH="$MLIP_NATIVE_PREFIX/python/bin:$MLIP_NATIVE_PREFIX/mpi-prefix/bin:$PATH"
export LD_LIBRARY_PATH="$MLIP_NATIVE_PREFIX/mpi-prefix/lib64:$MLIP_NATIVE_PREFIX/python/lib:$gcc_runtime${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
exec "$MLIP_NATIVE_PREFIX/python/bin/python3" "$here/examples/lammps_si.py" \
  --mliap-model "$MLIP_MLIAP_MODEL" "$@"
