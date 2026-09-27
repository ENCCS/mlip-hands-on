#!/usr/bin/env bash
# Run on Arrhenius login. The result is one bounded GPU Jupyter job.
set -euo pipefail
umask 077
: "${MLIP_ACCOUNT:?set your current GPU account}"
: "${MLIP_LESSON_ROOT:?set the staged lesson root}"
: "${MLIP_NOTEBOOK_VENV:?set the private GPU-node notebook venv}"
: "${MLIP_ENV_FILE:?set the private artifact environment file}"
: "${MLIP_JOB_LOG_DIR:?set a private log directory outside Git}"
test -f "$MLIP_LESSON_ROOT/scripts/arrhenius-jupyter.sbatch"
test -f "$MLIP_ENV_FILE"
test -x "$MLIP_NOTEBOOK_VENV/bin/python"
mkdir -p -m 700 "$MLIP_JOB_LOG_DIR"
mode=$(stat -c %a "$MLIP_JOB_LOG_DIR")
owner=$(stat -c %u "$MLIP_JOB_LOG_DIR")
if (( (8#$mode & 077) != 0 )) || [[ "$owner" != "$(id -u)" ]]; then
  echo 'The job log directory must be owner-only (setgid is allowed).' >&2
  exit 1
fi
reservation=()
if [[ -n "${MLIP_RESERVATION:-}" ]]; then
  reservation=(--reservation="$MLIP_RESERVATION")
fi
time_limit=${MLIP_TIME_LIMIT:-02:00:00}
[[ "$time_limit" =~ ^[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]] || {
  echo 'MLIP_TIME_LIMIT must be HH:MM:SS.' >&2; exit 2;
}

# --parsable gives one certain job ID. If transport fails, inspect squeue;
# never submit the same request again just because no response arrived.
job_id=$(sbatch --parsable --account="$MLIP_ACCOUNT" \
  "${reservation[@]}" --time="$time_limit" \
  --output="$MLIP_JOB_LOG_DIR/jupyter-%j.log" \
  "$MLIP_LESSON_ROOT/scripts/arrhenius-jupyter.sbatch")
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo 'Submission outcome needs read-only inspection.' >&2; exit 1; }
printf 'Jupyter job ID: %s\nPrivate token log: %s/jupyter-%s.log\n' \
  "$job_id" "$MLIP_JOB_LOG_DIR" "$job_id"
