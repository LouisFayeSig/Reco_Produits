SHELL := /bin/bash
.RECIPEPREFIX := >
#.ONESHELL:  # décommente si tu veux un shell unique par recette

# ========= Config =========
VENV_TRAIN ?= training/.venv-training
VENV_API   ?= api/.venv-api

API_DIR       ?= api
ARTIFACT_ROOT ?= artifacts
RUN_TAG       ?= daily

DIR_DATA ?= training/data
VENTES  ?= $(DIR_DATA)/df_ventes_grouped.parquet
DETAILS ?= $(DIR_DATA)/df_details_produits.parquet
PAIRS   ?= $(DIR_DATA)/df_couple_produit.parquet

SERVEUR ?= seveur

HOST    ?= 0.0.0.0
PORT    ?= 8080
WORKERS ?= 1

BASE_URL   ?= http://$(HOST):$(PORT)
HEALTH_URL ?= $(BASE_URL)/healthz
INFO_URL   ?= $(BASE_URL)/version
RELOAD_URL ?= $(BASE_URL)/reload

TRAIN_FLAGS ?= --make-current-symlink
CHECKER     ?= ops/save_precomputed.py

.PHONY: help data train precompute verify_run test check_api reload stop eval

.DEFAULT_GOAL := help

help:
> echo "Targets:"
> echo "  make data        - Pull les informations depuis le datawarehouse"
> echo "  make train       - Entraîne (venv: $(VENV_TRAIN)), précalcule puis vérifie les artefacts"
> echo "  make precompute  - Complète les pré-calculs manquants sur le dernier run"
> echo "  make verify_run  - Vérifie l'intégrité du dernier run (exit 2 si manquant)"
> echo "  make test        - Démarre l'API (venv: $(VENV_API)) sur $(HOST):$(PORT)"
> echo "  make check_api   - Health check rapide de l'API en local ($(HEALTH_URL) / $(INFO_URL))"
> echo "  make reload      - Demande à l'API de recharger le dernier run ($(RELOAD_URL))"
> echo "  make stop        - Arrête le serveur qui écoute sur PORT ($(PORT))"
> echo "  make eval        - Évalue offline (ops/eval_offline.py)"

# ========== Data ============
data:
> ( \
>   . "$(VENV_TRAIN)/bin/activate"; set -euo pipefail; \
>   export PYTHONNOUSERSITE=1; \
>   python ops/creation_dataset.py \
>     --nom_serveur "$(SERVEUR)" \
>     --dir_data "$(DIR_DATA)"; \
> )

# ========= 1) TRAIN =========
train:
> ( \
>   . "$(VENV_TRAIN)/bin/activate"; set -euo pipefail; \
>   export PYTHONNOUSERSITE=1; \
>   python training/train_daily.py \
>     --ventes "$(VENTES)" \
>     --details "$(DETAILS)" \
>     --pairs "$(PAIRS)" \
>     --artifact-root "$(ARTIFACT_ROOT)" \
>     $(TRAIN_FLAGS); \
> )
> $(MAKE) precompute
> $(MAKE) verify_run

# Complète les pré-calculs manquants (embeddings/popul./paires/meta) sur le dernier run
precompute:
> ( \
>   . "$(VENV_TRAIN)/bin/activate"; set -euo pipefail; \
>   export PYTHONNOUSERSITE=1; \
>   python "$(CHECKER)" --root "$(ARTIFACT_ROOT)" --details "$(DETAILS)"; \
> )

# Vérifie l’intégrité du dernier run (exit 0 si OK, 2 sinon)
verify_run:
> ( \
>   . "$(VENV_TRAIN)/bin/activate"; set -euo pipefail; \
>   export PYTHONNOUSERSITE=1; \
>   python "$(CHECKER)" --root "$(ARTIFACT_ROOT)" --check-only; \
> )

# ========= 2) TEST API =========
# Démarre l’API en pointant ARTIFACT_DIR vers le "root" (l’app choisit le dernier run)
test:
> ( \
>   . "$(VENV_API)/bin/activate"; set -euo pipefail; \
>   cd "$(API_DIR)"; \
>   ART_DIR="$(ARTIFACT_ROOT)"; \
>   case "$$ART_DIR" in /*) PASS="$$ART_DIR";; *) PASS="../$$ART_DIR";; esac; \
>   export ARTIFACT_DIR="$$PASS"; \
>   export DISABLE_MODEL_PICKLE=1; \
>   echo "ARTIFACT_DIR=$$PASS"; \
>   echo "Docs: uvicorn app:app --host $(HOST) --port $(PORT) --workers $(WORKERS)"; \
>   python -m uvicorn app:app --host "$(HOST)" --port "$(PORT)" --workers "$(WORKERS)"; \
> )

# Health check API (nécessite l’API démarrée)
check_api:
> curl -fsS "$(HEALTH_URL)" >/dev/null && echo "healthz OK" || (echo "healthz FAIL"; exit 2)
> curl -fsS "$(INFO_URL)" \
>   | python -c "import sys,json; print('artifact_dir_resolved:', json.load(sys.stdin).get('artifact_dir_resolved','?'))" \
>   || (echo "version FAIL"; exit 2)

# Demande à l’API de recharger le tout dernier run (endpoint /reload)
reload:
> curl -fsS -X POST "$(RELOAD_URL)" -H 'Content-Type: application/json' -d '{"artifact_dir":"auto"}' \
>   && echo "" || (echo "reload FAIL"; exit 2)

# Arrête le serveur (Uvicorn/Gunicorn) qui écoute sur PORT
stop:
> ( \
>   set -e; \
>   PORT="$(PORT)"; \
>   echo "== [stop] recherche de processus à arrêter sur le port $$PORT =="; \
>   killed=0; \
>   if command -v fuser >/dev/null 2>&1; then \
>     PIDS="$$(fuser -n tcp $$PORT 2>/dev/null || true)"; \
>     if [ -n "$$PIDS" ]; then \
>       echo "PIDs via fuser: $$PIDS"; \
>       kill -TERM $$PIDS 2>/dev/null || true; \
>       killed=1; \
>     fi; \
>   fi; \
>   if [ $$killed -eq 0 ] && command -v lsof >/dev/null 2>&1; then \
>     PIDS="$$(lsof -ti tcp:$$PORT -sTCP:LISTEN 2>/dev/null || true)"; \
>     if [ -n "$$PIDS" ]; then \
>       echo "PIDs via lsof: $$PIDS"; \
>       kill -TERM $$PIDS 2>/dev/null || true; \
>       killed=1; \
>     fi; \
>   fi; \
>   if [ $$killed -eq 0 ] && command -v ss >/dev/null 2>&1; then \
>     PIDS="$$(ss -lptn "sport = :$${PORT}" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | sort -u)"; \
>     if [ -n "$$PIDS" ]; then \
>       echo "PIDs via ss: $$PIDS"; \
>       kill -TERM $$PIDS 2>/dev/null || true; \
>       killed=1; \
>     fi; \
>   fi; \
>   if [ $$killed -eq 0 ]; then \
>     echo "Aucun listener détecté sur $$PORT, tentative pkill…"; \
>     pkill -u $$USER -f 'uvicorn .* --port $(PORT)' 2>/dev/null || true; \
>     pkill -u $$USER -f 'gunicorn .*app:app' 2>/dev/null || true; \
>   fi; \
>   echo '== [stop] terminé =='; \
> )

# ========= 3) EVAL OFFLINE =========
EVAL_VENTES    ?= $(VENTES)
EVAL_ARTIFACTS ?= $(ARTIFACT_ROOT)    # peut être "artifacts", "artifacts/run_current" ou un run précis
EVAL_BASKET_M  ?= 1
EVAL_K         ?= 5
EVAL_N         ?= 10000
EVAL_MODE      ?= auto                # auto | prof | global (selon ops/eval_offline.py)
EVAL_SEED      ?= 42

eval:
> @set -euo pipefail; \
> echo "== [eval] resolve run dir depuis '$(EVAL_ARTIFACTS)' =="; \
> R="$(EVAL_ARTIFACTS)"; \
> if [ -L "$$R/run_current" ]; then RUN_DIR="$$(readlink -f "$$R/run_current")"; \
> elif [ -f "$$R/mapping.pkl" ]; then RUN_DIR="$$R"; \
> else RUN_DIR="$$(ls -1dt "$$R"/run_* 2>/dev/null | head -n1)"; fi; \
> if [ -z "$$RUN_DIR" ]; then echo "ERREUR: aucun run trouvé sous $$R"; exit 1; fi; \
> echo "   RUN_DIR=$$RUN_DIR"; \
> . $(VENV_TRAIN)/bin/activate; \
> export PYTHONNOUSERSITE=1; \
> python ops/eval_offline.py \
>   --ventes "$(EVAL_VENTES)" \
>   --artifact-dir "$$RUN_DIR" \
>   --basket-m $(EVAL_BASKET_M) \
>   --k $(EVAL_K) \
>   --n-samples $(EVAL_N) \
>   --mode $(EVAL_MODE) \
>   --seed $(EVAL_SEED)