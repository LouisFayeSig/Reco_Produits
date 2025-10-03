#!/usr/bin/env bash
set -euo pipefail
ARTIF_ROOT="${1:-artifacts}"
API_URL="${API_URL:-http://localhost:8080}"
ADMIN_KEY="${ADMIN_KEY:-change-me-please}"

cd "$(dirname "$0")/.."
ROOT="$PWD"
ARTDIR="${ROOT}/${ARTIF_ROOT}"

LATEST_DIR=$(ls -1d ${ARTDIR}/run_* | sort | tail -n1)
echo "[promote] latest = ${LATEST_DIR}"

# gérer previous/current
if [[ -L "${ARTDIR}/current" ]]; then
  CURR="$(readlink -f "${ARTDIR}/current")"
  if [[ -n "$CURR" && -d "$CURR" ]]; then
    ln -sfn "$CURR" "${ARTDIR}/previous"
  fi
fi
ln -sfn "${LATEST_DIR}" "${ARTDIR}/current"

# reload API
curl -s -X POST "${API_URL}/reload" \
  -H 'Content-Type: application/json' \
  -H "X-Admin-Key: ${ADMIN_KEY}" \
  -d "{\"artifact_dir\":\"${LATEST_DIR}\",\"params_panier_json\":\"${ARTDIR}/reco_panier_params.json\"}" \
  | jq . || true

echo "[promote] done."
