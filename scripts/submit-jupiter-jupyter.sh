#!/usr/bin/env bash
# Run on a JUPITER login node. Submit one bounded private GPU notebook job.
set -euo pipefail
umask 077
: "${MLIP_ACCOUNT:?set the current JUPITER project account}"
: "${MLIP_LESSON_ROOT:?set the staged lesson root}"
: "${MLIP_NOTEBOOK_VENV:?set the private notebook environment}"
: "${MLIP_ENV_FILE:?set the private artifact environment file}"
: "${MLIP_JOB_LOG_DIR:?set a private job-log directory outside Git}"
test -f "$MLIP_LESSON_ROOT/scripts/jupiter-jupyter.sbatch"
test -x "$MLIP_NOTEBOOK_VENV/bin/python"
test -f "$MLIP_ENV_FILE"
mkdir -p -m 700 "$MLIP_JOB_LOG_DIR"
mode=$(stat -c %a "$MLIP_JOB_LOG_DIR")
owner=$(stat -c %u "$MLIP_JOB_LOG_DIR")
if (( (8#$mode & 077) != 0 )) || [[ "$owner" != "$(id -u)" ]]; then
  echo 'The job-log directory must be owner-only (setgid is allowed).' >&2
  exit 1
fi
time_limit=${MLIP_TIME_LIMIT:-02:00:00}
[[ "$time_limit" =~ ^[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]] || {
  echo 'MLIP_TIME_LIMIT must be HH:MM:SS.' >&2; exit 2;
}
# An unclear sbatch transport outcome is inspected by job identity, never
# replayed just to obtain a response.
job_id=$(sbatch --parsable --account="$MLIP_ACCOUNT" --time="$time_limit" \
  --output="$MLIP_JOB_LOG_DIR/jupyter-%j.log" \
  --error="$MLIP_JOB_LOG_DIR/jupyter-%j.err" \
  --export=ALL,MLIP_LESSON_ROOT,MLIP_NOTEBOOK_VENV,MLIP_ENV_FILE,MLIP_JOB_LOG_DIR \
  "$MLIP_LESSON_ROOT/scripts/jupiter-jupyter.sbatch")
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo 'Submission outcome needs read-only inspection.' >&2; exit 1; }
printf 'Jupyter job ID: %s\nPrivate token log: %s/jupyter-%s.log\n' \
  "$job_id" "$MLIP_JOB_LOG_DIR" "$job_id"
