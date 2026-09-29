#!/usr/bin/env bash
# Source this in an Arrhenius GPU shell before the native LAMMPS examples.
: "${MLIP_NATIVE_PREFIX:?set extracted native runtime root}"
source /software/sse2/init/hpc_init_sse.sh >/dev/null 2>&1
module purge >/dev/null 2>&1
module load GPU/buildenv-gcccuda/2026.03-cu13.0 >/dev/null 2>&1
gcc_runtime=/software/sse2/el9_gh200/manual/GCC/14.3.0/hpc1/lib64
export MLIP_LMP="$MLIP_NATIVE_PREFIX/mpi-prefix/bin/lmp"
export PATH="$MLIP_NATIVE_PREFIX/python/bin:$MLIP_NATIVE_PREFIX/mpi-prefix/bin:$PATH"
export LD_LIBRARY_PATH="$MLIP_NATIVE_PREFIX/mpi-prefix/lib64:$MLIP_NATIVE_PREFIX/python/lib:$gcc_runtime${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1
export FI_PROVIDER=cxi
export MPIR_CVAR_CH4_OFI_ENABLE_HMEM=1
export SLURM_EXPORT_ENV=ALL
export MLIP_SRUN_MPI=pmi2
unset PYTHONPATH PYTHONHOME
