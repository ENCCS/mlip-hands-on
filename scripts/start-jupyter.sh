#!/bin/bash
# Run *inside* one reviewed GPU allocation; creates no Slurm job or tunnel.
set -euo pipefail
: "${SLURM_JOB_ID:?start this only inside an allocated GH200 shell}"
python -c 'import jupyterlab, jupytext, matplotlib; from jupytext import TextFileContentsManager' >/dev/null || {
  echo 'JupyterLab, Jupytext and Matplotlib are required for the MyST notebook chapters' >&2
  exit 1
}
here=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
settings_dir=$(mktemp -d "${TMPDIR:-/tmp}/mlip-jupyter-settings.XXXXXXXX")
install -D -m 0600 \
  "$here/jupyterlab-settings/@jupyterlab/docmanager-extension/plugin.jupyterlab-settings" \
  "$settings_dir/@jupyterlab/docmanager-extension/plugin.jupyterlab-settings"
export JUPYTERLAB_SETTINGS_DIR="$settings_dir"
cd "$here"
if [[ "${MLIP_JUPYTER_TLS:-0}" == 1 ]]; then
  : "${MLIP_JOB_LOG_DIR:?TLS mode needs a private job log directory}"
  tls_dir=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/mlip-jupyter-tls.XXXXXXXX")
  chmod 700 "$tls_dir"
  openssl req -x509 -nodes -newkey rsa:3072 -days 1 \
    -subj '/CN=localhost' -addext 'subjectAltName=DNS:localhost,IP:127.0.0.1' \
    -keyout "$tls_dir/key.pem" -out "$tls_dir/cert.pem" >/dev/null 2>&1
  chmod 600 "$tls_dir/key.pem" "$tls_dir/cert.pem"
  openssl x509 -in "$tls_dir/cert.pem" -noout -fingerprint -sha256 \
    >"$MLIP_JOB_LOG_DIR/jupyter-${SLURM_JOB_ID}.fingerprint"
  chmod 600 "$MLIP_JOB_LOG_DIR/jupyter-${SLURM_JOB_ID}.fingerprint"
  echo 'TLS notebook relay is ready; compare its certificate fingerprint before using the browser.'
  exec python -m jupyterlab --no-browser --ip=0.0.0.0 --port="${MLIP_JUPYTER_PORT:-8888}" \
    --certfile="$tls_dir/cert.pem" --keyfile="$tls_dir/key.pem" \
    --ServerApp.allow_remote_access=True --ServerApp.root_dir="$here" \
    --ServerApp.contents_manager_class=jupytext.TextFileContentsManager
fi
exec python -m jupyterlab --no-browser --ip=127.0.0.1 --port="${MLIP_JUPYTER_PORT:-8888}" \
  --ServerApp.allow_remote_access=False --ServerApp.root_dir="$here" \
  --ServerApp.contents_manager_class=jupytext.TextFileContentsManager
