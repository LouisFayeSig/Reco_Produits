import os, json, re
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
from reco_service import RecommenderService
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
# ARTIFACT_DIR peut être :
#  • un run précis (…/run_YYYYMMDD_HHMMSS[_tag])
#  • un symlink vers un run (ex: …/run_current)
#  • un dossier racine (ex: …/artifacts) → on prendra le dernier run_*
ARTIFACT_DIR = "/home/lfaye/projects/reco/artifacts"
# Fichier JSON unique contenant soit un profil "plat" (basket_params_by_size), soit plusieurs profils avec {active, profiles{...}}
PARAMS_PANIER_JSON = "/home/lfaye/projects/reco/api/panier_params.json"
# Option : nom du profil à utiliser au démarrage si le fichier est multi-profils (sinon ignoré)
PANIER_PROFILE = os.environ.get("PANIER_PROFILE", "")
# Clé admin (header "X-Admin-Key") pour /reload si tu veux verrouiller
ADMIN_KEY = os.environ.get("ADMIN_KEY", "")

RE_RUN = re.compile(r"^run_\d{8}_\d{6}.*")  # ex: run_20250924_122145[_tag]


# ──────────────────────────────────────────────────────────────────────────────
# Résolution d'artefacts
# ──────────────────────────────────────────────────────────────────────────────
def _is_run_dir(p: Path) -> bool:
    return p.is_dir() and (p / "mapping.pkl").exists()

def _latest_run_dir(root: Path) -> Path | None:
    if not root.exists() or not root.is_dir():
        return None
    cands = [d for d in root.iterdir() if d.is_dir() and RE_RUN.match(d.name)]
    if not cands:
        return None
    cands.sort(key=lambda d: d.name, reverse=True)  # nom horodaté = ordre chrono
    return cands[0]

def _resolve_artifact_dir(art_dir_arg: str | None) -> Path:
    """
    - None / "" / "auto"  → utilise ARTIFACT_DIR global
    - Si c’est un symlink → résout
    - Si c’est un run (mapping.pkl présent) → OK
    - Si c’est un dossier racine → prend le dernier run_* dedans
    """
    base = Path(ARTIFACT_DIR) if not art_dir_arg or art_dir_arg == "auto" else Path(art_dir_arg)
    p = base
    if p.is_symlink():
        p = p.resolve()

    # Déjà un run
    if _is_run_dir(p):
        return p

    # Dossier racine: tenter dernier run_*
    if p.is_dir():
        # Si mapping.pkl est directement là, on considère que c'est un run
        if (p / "mapping.pkl").exists():
            return p
        last = _latest_run_dir(p)
        if last:
            return last

    raise FileNotFoundError(f"Aucun run valide trouvé à partir de: {base}")


# ──────────────────────────────────────────────────────────────────────────────
# Chargement des profils panier
# ──────────────────────────────────────────────────────────────────────────────
def _load_basket_params(path: str, profile: str | None = None):
    """
    Lit toujours PARAMS_PANIER_JSON (pas besoin de fournir un chemin au /reload).
    Supporte :
      - { "active": "name", "profiles": { "name": { "basket_params_by_size": {...} }, ... } }
    """
    if not path or not os.path.exists(path):
        return {}, None

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    profs = data.get("profiles", {})
    chosen = profile or data.get("active")
    if chosen and chosen in profs and "basket_params_by_size" in profs[chosen]:
        return profs[chosen]["basket_params_by_size"], chosen

    # Rien d’utilisable
    return {}, None


# ──────────────────────────────────────────────────────────────────────────────
# Lifecycle
# ──────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[INIT] artifact_dir (env) =", ARTIFACT_DIR)

    # Charge les paramètres de panier depuis PARAMS_PANIER_JSON (unique fichier)
    basket_params, prof_name = _load_basket_params(PARAMS_PANIER_JSON, PANIER_PROFILE or None)

    try:
        resolved = _resolve_artifact_dir("auto")  # ← 'auto' utilise ARTIFACT_DIR et prend le dernier run si besoin
        app.state.svc = RecommenderService(str(resolved), basket_params_by_size=basket_params)
        app.state.artifact_dir = resolved
        app.state.basket_profile = prof_name
        app.state.params_source = PARAMS_PANIER_JSON
        print("[INIT] artifact_dir (resolved) =", resolved)
        print("[OK] service prêt.")
    except Exception as e:
        app.state.svc = None
        app.state.artifact_dir = None
        print("[ERR] init:", repr(e), flush=True)
        raise
    yield


app = FastAPI(lifespan=lifespan, title="Reco Matériaux", version="1.0.0")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers & endpoints
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/healthz")
def healthz():
    return {"ok": True}

def _get_svc(request: Request) -> RecommenderService:
    svc = getattr(request.app.state, "svc", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return svc

@app.get("/version")
def version(request: Request):
    svc = _get_svc(request)
    info = svc.status_dict()
    info["artifact_dir_resolved"] = str(getattr(request.app.state, "artifact_dir", ""))
    info["basket_profile"] = getattr(request.app.state, "basket_profile", None)
    info["params_source"] = getattr(request.app.state, "params_source", None)
    info["basket_params_by_size"] = getattr(svc, "basket_params_by_size", {})
    return JSONResponse(info)


class RecPanierIn(BaseModel):
    profession: Optional[str] = None
    product_ids: list[int] = []
    k: int = 10
    expand: bool = False  # True pour afficher les libellés

@app.post("/recommend_panier")
def recommend_panier(inp: RecPanierIn, svc: RecommenderService = Depends(_get_svc)):
    recs = svc.recommend_panier(inp.profession, inp.product_ids, k=inp.k)
    if inp.expand:
        return {"recommended": svc.enrich_items(recs)}
    return {"recommended": recs}


class ReloadIn(BaseModel):
    # "auto" | root "artifacts" | run précis | symlink ; None → "auto"
    artifact_dir: str | None = None
    # Nom de profil à activer s’il y en a plusieurs dans PARAMS_PANIER_JSON
    profile: str | None = None

@app.post("/reload")
def reload_artifacts(inp: ReloadIn, request: Request, x_admin_key: str = Header(default="")):
    # Sécurité minimale
    if ADMIN_KEY and x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")

    # 1) Résoudre les artefacts (None / "auto" → dernier run sous ARTIFACT_DIR)
    try:
        resolved = _resolve_artifact_dir(inp.artifact_dir or "auto")
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2) Charger les params depuis PARAMS_PANIER_JSON (unique), en choisissant le profil demandé (sinon garde l'actuel/actif)
    current_profile = getattr(request.app.state, "basket_profile", None)
    desired_profile = inp.profile or current_profile
    basket_params, prof_name = _load_basket_params(PARAMS_PANIER_JSON, desired_profile)

    # 3) Créer une nouvelle instance de service
    new_svc = RecommenderService(str(resolved), basket_params_by_size=basket_params)

    # 4) Swap atomique
    request.app.state.svc = new_svc
    request.app.state.artifact_dir = resolved
    request.app.state.basket_profile = prof_name
    request.app.state.params_source = PARAMS_PANIER_JSON

    return {
        "ok": True,
        "artifact_dir_resolved": str(resolved),
        "basket_profile": prof_name,
        "has_params": bool(basket_params),
    }


class EnrichIn(BaseModel):
    ids: list[int]

@app.post("/products/enrich")
def products_enrich(inp: EnrichIn, svc: RecommenderService = Depends(_get_svc)):
    return {"items": svc.enrich_items(inp.ids)}
