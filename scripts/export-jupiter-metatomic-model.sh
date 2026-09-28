#!/usr/bin/env bash
# Export the pinned MACE checkpoint for the Metatomic LAMMPS pair style.
set -euo pipefail
umask 077
: "${MLIP_JUPITER_ROOT:?set the private artifact root}"
: "${MLIP_JUPITER_ALCHEMI_ENV_ID:?set the qualified native environment job ID}"
: "${MLIP_MACE_MODEL:?set the original reviewed MACE checkpoint}"
: "${MLIP_METATOMIC_OVERLAY:?set the reviewed Metatomic overlay}"
: "${MLIP_MACE_EXPORT_OVERLAY:?set the reviewed MACE exporter overlay}"
: "${MLIP_METATOMIC_EXPORT_DIR:?set a fresh private export directory}"
case "$MLIP_JUPITER_ALCHEMI_ENV_ID" in ''|*[!0-9]*) exit 2 ;; esac
case "$MLIP_MACE_MODEL" in /*) ;; *) echo 'model path must be absolute' >&2; exit 2 ;; esac
case "$MLIP_MACE_MODEL" in *[!A-Za-z0-9_./-]*) echo 'model path contains unsupported characters' >&2; exit 2 ;; esac
test ! -e "$MLIP_METATOMIC_EXPORT_DIR"
test -x "$MLIP_METATOMIC_OVERLAY/bin/mtt"
test -d "$MLIP_MACE_EXPORT_OVERLAY/mace"
printf '%s  %s\n' 2ddb079cee0e131eaaf6912ba581b394551ead283e95c99cfe78c605d10b5736 "$MLIP_MACE_MODEL" | sha256sum -c - >/dev/null
module purge >/dev/null 2>&1
module load Stages/2025 GCC/13.3.0 Python/3.12.3 >/dev/null
mkdir -m 700 "$MLIP_METATOMIC_EXPORT_DIR"
export PYTHONPATH="$MLIP_MACE_EXPORT_OVERLAY:$MLIP_METATOMIC_OVERLAY"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
"$MLIP_JUPITER_ROOT/env-alchemi-$MLIP_JUPITER_ALCHEMI_ENV_ID/bin/python" - <<'PY'
from pathlib import Path
import os

target = Path(os.environ['MLIP_METATOMIC_EXPORT_DIR'])
model = os.environ['MLIP_MACE_MODEL']
(target / 'options.yaml').write_text(
    'architecture:\n'
    '  name: experimental.mace\n'
    '  model:\n'
    f'    mace_model: {model}\n'
    '    mace_head_target: energy\n'
    '  training:\n'
    '    num_epochs: 0\n'
    '    batch_size: 1\n'
    'training_set: dummy.xyz\n'
    'validation_set: dummy.xyz\n'
)
(target / 'dummy.xyz').write_text(
    '2\n'
    'Lattice="5.43 0 0 0 5.43 0 0 0 5.43" '
    'Properties=species:S:1:pos:R:3:forces:R:3 energy=-2.1 pbc="T T T"\n'
    'Si 0.0 0.0 0.0 0.0 0.0 0.0\n'
    'Si 2.715 2.715 2.715 0.0 0.0 0.0\n'
)
PY
cd "$MLIP_METATOMIC_EXPORT_DIR"
timeout 300 "$MLIP_METATOMIC_OVERLAY/bin/mtt" train options.yaml -o model.pt \
  >train.log 2>&1
test -s model.pt
sha256sum model.pt
