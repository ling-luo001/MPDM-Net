#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_NAME="${EXPERIMENT_NAME:-rd_asymmetric_polar_demand_v3_full_seed1234}"
if [[ -z "${EXP_ROOT:-}" ]]; then
    if mountpoint -q /media/lz/WZZ; then
        EXP_ROOT="/media/lz/WZZ/mpdm_full_experiments/rd_asymmetric_polar_demand_v3"
    else
        EXP_ROOT="${HOME}/mpdm_full_experiments/rd_asymmetric_polar_demand_v3"
    fi
fi

RUN_DIR="${EXP_ROOT}/${EXPERIMENT_NAME}"
STDOUT_LOG="${EXP_ROOT}/${EXPERIMENT_NAME}.stdout.log"
PID_FILE="${EXP_ROOT}/${EXPERIMENT_NAME}.pid"

if [[ -f "${PID_FILE}" ]]; then
    pid="$(cat "${PID_FILE}")"
    if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
        printf 'RUNNING PID %s\n' "${pid}"
    else
        printf 'NOT RUNNING (stale pid file: %s)\n' "${PID_FILE}"
    fi
else
    matching_pid="$(pgrep -f "train.py.*--exp_name[[:space:]]+${EXPERIMENT_NAME}" | head -n 1 || true)"
    if [[ -n "${matching_pid}" ]]; then
        printf 'RUNNING PID %s (pid file missing)\n' "${matching_pid}"
    else
        printf 'NOT RUNNING\n'
    fi
fi

printf 'Run directory: %s\n' "${RUN_DIR}"
if [[ -d "${RUN_DIR}" ]]; then
    latest_g="$(find "${RUN_DIR}" -maxdepth 1 -type f -name 'g_????????.pth' -printf '%f\n' | sort | tail -n 1 || true)"
    latest_do="$(find "${RUN_DIR}" -maxdepth 1 -type f -name 'do_????????.pth' -printf '%f\n' | sort | tail -n 1 || true)"
    printf 'Latest generator checkpoint: %s\n' "${latest_g:-none}"
    printf 'Latest optimizer checkpoint: %s\n' "${latest_do:-none}"
fi

if [[ -f "${STDOUT_LOG}" ]]; then
    printf '\nLast 20 log lines:\n'
    tail -n 20 "${STDOUT_LOG}"
else
    printf 'Log not created yet: %s\n' "${STDOUT_LOG}"
fi
