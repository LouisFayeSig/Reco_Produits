#!/usr/bin/env bash
set -euo pipefail
ARTIF_ROOT="${1:-artifacts}"
TARGET="${2:-previous}"   # 'previous' ou chemin explicite ex: artifacts/run_20251012_030001_daily
API_URL="${API_URL:-http://localhost:8080}"
ADMIN_KEY="${ADMIN_KEY:-change-me-please}"

cd "$(dirname "$0")/.."
ROOT="$PWD"
ARTDIR="${ROOT}/${ARTIF_ROOT}"

if [[ "$TARGET" == "previous" ]]; then
  if [[ ! -L "${ARTDIR}/previous" ]]; then
    echo "No 'previous' symlink found." >&2
    exit 1
  fi
  RUN_DIR="$(readlink -f "${ARTDIR}/previous")"
else
  RUN_DIR="$TARGET"
fi

if [[ ! -d "$RUN_DIR" ]]; then
  echo "Target run not found: $RUN_DIR" >&2
  exit 1
fi

ln -sfn "$RUN_DIR" "${ARTDIR}/current"

curl -s -X POST "${API_URL}/reload" \
  -H 'Content-Type: application/json' \
  -H "X-Admin-Key: ${ADMIN_KEY}" \
  -d "{\"artifact_dir\":\"${RUN_DIR}\",\"params_panier_json\":\"${ARTDIR}/reco_panier_params.json\"}" \
  | jq . || true

echo "[rollback] switched to $RUN_DIR"
