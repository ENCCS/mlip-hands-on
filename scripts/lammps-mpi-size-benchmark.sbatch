#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=100G
#SBATCH --time=00:15:00
#SBATCH --network=single_node_vni
# Cold 32,768-atom / 100-step coupled-MD gate. Override ranks, GPUs and
# memory together for separately reviewed 2- or 4-GH200 submissions.
set -euo pipefail
umask 077
here=$(pwd -P)
: "${MLIP_MPI_ARCHIVE:?verified MPI archive required}"
: "${MLIP_MPI_ARCHIVE_SHA256:?exact archive hash required}"
: "${MLIP_MLIAP_MODEL:?pinned ML-IAP export required}"
: "${MLIP_RESULTS_DIR:?existing private result root required}"
test "${SLURM_JOB_NUM_NODES:-}" = 1
case "${SLURM_NTASKS:-}" in 1|2|4) ;; *) echo 'requires one, two or four ranks' >&2; exit 1;; esac
test "${SLURM_GPUS_ON_NODE:-}" = "$SLURM_NTASKS"
test -f "$MLIP_MPI_ARCHIVE" && test -f "$MLIP_MLIAP_MODEL"
test -d "$MLIP_RESULTS_DIR"
test -f "$here/examples/lammps_mpi.py"
printf '%s  %s\n' "$MLIP_MPI_ARCHIVE_SHA256" "$MLIP_MPI_ARCHIVE" | sha256sum -c - >/dev/null
printf '%s  %s\n' \
  db578c556298ad3bb1f4a93faa50d540eb2b9792215e81ef7548dd7e20a746e8 \
  "$MLIP_MLIAP_MODEL" | sha256sum -c - >/dev/null
source /software/sse2/init/hpc_init_sse.sh >/dev/null 2>&1
module load GPU/buildenv-gcccuda/2026.03-cu13.0 >/dev/null 2>&1
: "${CUDA_VISIBLE_DEVICES:?allocation did not expose GPU identities}"
export MLIP_ALLOCATED_CUDA_DEVICES="$CUDA_VISIBLE_DEVICES"
scratch=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/mlip-mpi-size-${SLURM_JOB_ID}.XXXXXX")
tar -xzf "$MLIP_MPI_ARCHIVE" -C "$scratch"
test -x "$scratch/python/bin/python3"
test -f "$scratch/mpi-prefix/lib64/liblammps.so.0"
gcc_runtime=/software/sse2/el9_gh200/manual/GCC/14.3.0/hpc1/lib64
export LD_LIBRARY_PATH="$scratch/mpi-prefix/lib64:$scratch/python/lib:$gcc_runtime${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export FI_PROVIDER=cxi
export MPIR_CVAR_CH4_OFI_ENABLE_HMEM=1
export SLURM_EXPORT_ENV=ALL
result="$MLIP_RESULTS_DIR/lammps-mpi-size-$SLURM_JOB_ID"
mkdir -m 700 "$result"
timeout --signal=TERM --kill-after=5s 780s \
  srun --mpi=pmi2 --nodes=1 --ntasks="$SLURM_NTASKS" \
    --ntasks-per-node="$SLURM_NTASKS" --gpu-bind=none --cpu-bind=cores \
    --output="$result/rank-%t.out" --error="$result/rank-%t.err" \
    /bin/bash -c 'export CUDA_VISIBLE_DEVICES="$MLIP_ALLOCATED_CUDA_DEVICES"; exec "$@"' _ \
    "$scratch/python/bin/python3" "$here/examples/lammps_mpi.py" \
    --model "$MLIP_MLIAP_MODEL" --output-dir "$result" --cells 16 --steps 100
python3 - "$result" "$SLURM_NTASKS" <<'PY'
import json, math, pathlib, sys
root = pathlib.Path(sys.argv[1])
count = int(sys.argv[2])
rows = [json.loads((root / ('rank-%d.json' % i)).read_text()) for i in range(count)]
for i, row in enumerate(rows):
    assert row['state'] == 'completed' and row['rank'] == i
    assert row['local_rank'] == i and row['selected_device_index'] == i
    assert row['visible_gpus'] == count and row['atoms'] == 32768
    assert row['measured_steps'] == 100 and math.isfinite(row['final_energy_ev'])
assert all(math.isclose(row['final_energy_ev'], rows[0]['final_energy_ev'],
                        rel_tol=1e-9, abs_tol=1e-6) for row in rows[1:])
print('md_seconds_max=%.6f' % max(row['measured_seconds'] for row in rows))
PY
printf 'one-node MPI LAMMPS size benchmark passed; private evidence: %s\n' "$result"
