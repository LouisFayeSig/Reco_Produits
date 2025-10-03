#!/usr/bin/env bash
set -euo pipefail
ARTIF_ROOT="${1:-artifacts}"
KEEP_N="${2:-14}"   # nb de versions à garder

cd "$(dirname "$0")/.."   # repo root
ROOT="$PWD"
ARTDIR="${ROOT}/${ARTIF_ROOT}"

if [[ ! -d "$ARTDIR" ]]; then
  echo "Artifacts root not found: $ARTDIR" >&2
  exit 1
fi

runs=( $(ls -1d "${ARTDIR}"/run_* 2>/dev/null | sort) )
total=${#runs[@]}

echo "[cleanup] total runs: $total | keep last: $KEEP_N"

if (( total <= KEEP_N )); then
  echo "[cleanup] nothing to delete."
  exit 0
fi

to_del=$(( total - KEEP_N ))
del_list=( "${runs[@]:0:$to_del}" )

# ne jamais supprimer le symlink 'current' (dossier pointé peut être supprimé si ancien)
if [[ -L "${ARTDIR}/current" ]]; then
  current_target="$(readlink -f "${ARTDIR}/current")"
else
  current_target=""
fi

for d in "${del_list[@]}"; do
  if [[ "$d" == "$current_target" ]]; then
    echo "[cleanup] skip current target: $d"
    continue
  fi
  echo "[cleanup] rm -rf $d"
  rm -rf --one-file-system "$d"
done

# purge symlinks cassés dans artifacts/
find "$ARTDIR" -xtype l -print -delete || true

echo "[cleanup] done."
