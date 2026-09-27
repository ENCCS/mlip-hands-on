#!/usr/bin/env bash
# Run on your laptop. Both destinations stay on loopback behind SSH.
set -euo pipefail

usage() {
  echo 'Usage:' >&2
  echo '  connect-from-laptop.sh html SSH_ALIAS [LOCAL_PORT [REMOTE_PORT]]' >&2
  echo '  connect-from-laptop.sh notebook SSH_LOGIN SLURM_JOB_ID [LOCAL_PORT [REMOTE_PORT]]' >&2
  exit 2
}

port() {
  case "$1" in ''|*[!0-9]*) usage ;; esac
  (( 10#$1 >= 1024 && 10#$1 <= 65535 )) || usage
}

ssh_config=${MLIP_SSH_CONFIG:-"$HOME/.ssh/config"}
test -f "$ssh_config" || { echo "SSH config not found: $ssh_config" >&2; exit 2; }
ssh_args=(-F "$ssh_config" -o ExitOnForwardFailure=yes -o ServerAliveInterval=30)

hold_forward() {
  local host=$1 spec=$2
  if ssh "${ssh_args[@]}" -O check "$host" >/dev/null 2>&1; then
    # The site's authenticated master may be the only usable SSH login.
    # Own this specific forwarding explicitly and cancel it on exit.
    ssh "${ssh_args[@]}" -O forward -L "$spec" "$host"
    mlip_forward_host=$host
    mlip_forward_spec=$spec
    cleanup_forward() {
      ssh "${ssh_args[@]}" -O cancel -L "$mlip_forward_spec" "$mlip_forward_host" >/dev/null 2>&1 || true
    }
    trap cleanup_forward EXIT
    echo 'Press Enter to close this forwarding on the authenticated SSH connection.'
    read -r _ || true
    cleanup_forward
    trap - EXIT
    return
  fi
  # With no master, the dedicated SSH process owns the listener itself.
  exec ssh "${ssh_args[@]}" -S none -N -L "$spec" "$host"
}

case "${1:-}" in
  html)
    (( $# >= 2 && $# <= 4 )) || usage
    host=$2
    local_port=${3:-18766}
    remote_port=${4:-8766}
    port "$local_port"; port "$remote_port"
    ssh "${ssh_args[@]}" "$host" \
      "curl -fsS -o /dev/null http://127.0.0.1:${remote_port}/" || {
      echo 'The HTML server is not responding on the remote loopback port.' >&2
      exit 1
    }
    printf 'Connecting; if SSH reports no error, open http://127.0.0.1:%s/.\n' "$local_port"
    hold_forward "$host" "127.0.0.1:${local_port}:127.0.0.1:${remote_port}"
    ;;
  notebook)
    (( $# >= 3 && $# <= 5 )) || usage
    login=$2
    job_id=$3
    case "$job_id" in ''|*[!0-9]*) usage ;; esac
    local_port=${4:-18888}
    remote_port=${5:-8888}
    port "$local_port"; port "$remote_port"
    assignment=$(ssh "${ssh_args[@]}" "$login" \
      "squeue -h -j ${job_id} -o '%N|%T'")
    if [[ ! "$assignment" =~ ^([a-zA-Z0-9.-]+)\|RUNNING$ ]]; then
      echo 'The job is not one running allocation on one named node.' >&2
      exit 1
    fi
    node=${BASH_REMATCH[1]}
    job_fields=$(ssh "${ssh_args[@]}" "$login" "scontrol show job -o ${job_id}")
    if [[ ! " $job_fields " =~ [[:space:]]StdOut=([^[:space:]]+)[[:space:]] ]]; then
      echo 'Cannot identify the private Jupyter job log.' >&2
      exit 1
    fi
    log_path=${BASH_REMATCH[1]}
    if [[ ! "$log_path" =~ ^/[A-Za-z0-9_./-]+/jupyter-${job_id}\.log$ ]]; then
      echo 'This is not the expected Jupyter job log.' >&2
      exit 1
    fi
    fingerprint_path=${log_path%.log}.fingerprint
    recorded=$(ssh "${ssh_args[@]}" "$login" "test -f '$fingerprint_path' && sed -n 1p '$fingerprint_path'")
    observed=$(ssh "${ssh_args[@]}" "$login" \
      "timeout 10 openssl s_client -connect '$node:$remote_port' -servername localhost </dev/null 2>/dev/null | openssl x509 -noout -fingerprint -sha256")
    if [[ -z "$recorded" || "$recorded" != "$observed" ]]; then
      echo 'The private TLS fingerprint did not match the allocated node.' >&2
      exit 1
    fi
    echo 'The allocated notebook TLS certificate matches the private job record.'
    printf 'Connecting to allocated node %s through %s.\n' "$node" "$login"
    printf 'Open the private HTTPS Jupyter token URL using local port %s.\n' "$local_port"
    printf 'Your browser may warn about the self-signed certificate; this script checked its SHA-256 fingerprint.\n'
    printf 'Keep this terminal open and follow its close prompt.\n'
    hold_forward "$login" "127.0.0.1:${local_port}:${node}:${remote_port}"
    ;;
  *) usage ;;
esac
