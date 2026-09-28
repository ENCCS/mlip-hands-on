#!/usr/bin/env bash
# Run on a JUPITER login node; leave the qualified ALCHEMI environment intact.
set -euo pipefail
umask 077
: "${MLIP_JUPITER_ROOT:?set the private artifact root}"
: "${MLIP_JUPITER_ALCHEMI_ENV_ID:?set the qualified native environment job ID}"
: "${MLIP_METATOMIC_OVERLAY:?set a fresh private Metatomic package directory}"
: "${MLIP_MACE_EXPORT_OVERLAY:?set a fresh private MACE exporter directory}"
case "$MLIP_JUPITER_ALCHEMI_ENV_ID" in ''|*[!0-9]*) exit 2 ;; esac
test ! -e "$MLIP_METATOMIC_OVERLAY" && test ! -e "$MLIP_MACE_EXPORT_OVERLAY"
mkdir -m 700 "$MLIP_METATOMIC_OVERLAY" "$MLIP_MACE_EXPORT_OVERLAY"
venv="$MLIP_JUPITER_ROOT/env-alchemi-$MLIP_JUPITER_ALCHEMI_ENV_ID"
test -x "$venv/bin/python"
module purge >/dev/null 2>&1
module load Stages/2025 GCC/13.3.0 Python/3.12.3 >/dev/null
"$venv/bin/python" -m pip install --target "$MLIP_METATOMIC_OVERLAY" \
  --no-deps --only-binary=:all: --no-cache-dir \
  'metatrain==2026.4.1' 'metatensor-core==0.2.5' \
  'metatensor-learn==0.5.0' 'metatensor-operations==0.5.0' \
  'metatensor-torch==0.10.6' 'metatomic-torch==0.1.17' \
  'jsonschema==4.26.0' 'ctypes-dlpack==0.1.0' \
  'vesin==0.6.1' 'wigners==0.4.1' \
  >"$MLIP_METATOMIC_OVERLAY/install.log" 2>&1
"$venv/bin/python" -m pip install --target "$MLIP_METATOMIC_OVERLAY" \
  --no-deps --no-build-isolation --no-cache-dir 'python-hostlist==2.3.0' \
  >>"$MLIP_METATOMIC_OVERLAY/install.log" 2>&1
"$venv/bin/python" -m pip install --target "$MLIP_MACE_EXPORT_OVERLAY" \
  --no-deps --only-binary=:all: --no-cache-dir 'mace-torch==0.3.14' \
  >"$MLIP_MACE_EXPORT_OVERLAY/install.log" 2>&1
PYTHONPATH="$MLIP_MACE_EXPORT_OVERLAY:$MLIP_METATOMIC_OVERLAY" \
  "$venv/bin/python" - <<'PY'
import metatrain, metatomic.torch, metatensor.torch, mace
print('Private Metatomic/MACE export overlays import successfully')
PY
