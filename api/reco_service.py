import os, json, gzip, pickle, time
import numpy as np
from pathlib import Path
from scipy.sparse import load_npz

# Anti over-subscription (important en multi-workers uvicorn/gunicorn)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")


class RecommenderService:
    """
    Service de recommandation "basket-to-items".
    Charge les artefacts au démarrage, puis sert:
      - recommend_panier(profession: Optional[str], product_ids: list[int], k: int) -> list[int]

    Signaux utilisés:
      • Paires A->B par métier + fallback global ("ANY") + paires inverses (B->A)
      • Similarité d’embeddings LightFM (item_emb_norm)
      • Popularité par métier et globale
      • Bonus MDD (marque propre) via item_is_pl.npy (0/1)

    Diversité:
      • MMR (Maximal Marginal Relevance)
      • Cap par gamme optionnel
    """

    # ──────────────────────────────────────────────────────────────────────
    # INIT / CONFIG
    # ──────────────────────────────────────────────────────────────────────
    def __init__(self, artifact_dir: str, basket_params_by_size: dict | None = None):
        self.artifact_dir = Path(artifact_dir)
        self.started_at = time.time()

        # -- Poids de base (surmontés par les profils par taille de panier)
        self.W_PAIR  = float(os.getenv("W_PAIR",  "1.0"))
        self.W_EMB   = float(os.getenv("W_EMB",   "0.3"))
        self.W_POPP  = float(os.getenv("W_POPP",  "0.2"))
        self.W_POPG  = float(os.getenv("W_POPG",  "0.0"))
        self.pair_any_weight = float(os.getenv("PAIR_ANY_WEIGHT", "0.6"))

        # -- MDD: nouveaux réglages (recommandés)
        self.W_PL_BASE = float(os.getenv("W_PL_BASE", "0.0"))          # bonus additif constant pour is_pl=1
        self.W_PL_PER_BASK_PL = float(os.getenv("W_PL_PER_BASK_PL", "0.0"))  # bonus additif * ratio MDD observé

        # Back-compat: si anciens env vars sont fournis, on les mappe.
        #   W_PL_ABS -> s'ajoute à W_PL_BASE
        #   PL_MULT  -> on l'approxime par un W_PL_BASE additif moyen (léger) si non nul
        _old_abs = float(os.getenv("W_PL_ABS", "0.0"))
        if abs(_old_abs) > 1e-12:
            self.W_PL_BASE += _old_abs
        _pl_mult = float(os.getenv("PL_MULT", "1.0"))
        if abs(_pl_mult - 1.0) > 1e-6:
            # approx légère : un multiplicateur 1.05 ~ +0.05 additif (dépend du range)
            # on reste prudent pour éviter de casser le ranking → +0.02 par 0.05 de mult
            self.W_PL_BASE += max(0.0, (_pl_mult - 1.0)) * 0.40

        # -- Pools (taille des candidats)
        self.TOP_PAIRS = int(os.getenv("TOP_PAIRS", "2000"))
        self.TOP_POPP  = int(os.getenv("TOP_POPP",  "1200"))
        self.TOP_POPG  = int(os.getenv("TOP_POPG",  "300"))
        self.top_pairs_any_frac = float(os.getenv("TOP_PAIRS_ANY_FRAC", "0.3"))
        self.pair_quota = int(os.getenv("PAIR_QUOTA", "0"))  # 0 = off

        # -- Paires inverses (prédécesseurs) pour mieux couvrir les séquences
        self.USE_REV        = bool(int(os.getenv("USE_REV", "1")))
        self.TOP_REV_PAIRS  = int(os.getenv("TOP_REV_PAIRS", "1000"))
        self.REV_ANY_WEIGHT = float(os.getenv("REV_ANY_WEIGHT", "0.6"))
        self.GAMMA_COVER    = float(os.getenv("GAMMA_COVER", "0.05"))  # bonus "coverage"

        # -- Diversité
        self.diversify     = bool(int(os.getenv("DIVERSIFY", "0")))
        self.mmr_lambda    = float(os.getenv("MMR_LAMBDA", "0.7"))
        self.cap_per_gamme = int(os.getenv("CAP_PER_GAMME", "0"))

        # -- Cache LRU simple
        self._cache_ttl   = float(os.getenv("CACHE_TTL", "30"))
        self._cache_max   = int(os.getenv("CACHE_MAX", "10000"))
        self._cache       = {}
        self._cache_order = []

        # -- Auto-détection de métier (prof-mix)
        self.profmix_topk    = int(os.getenv("PROFMIX_TOPK", "3"))
        self.profmix_temp    = float(os.getenv("PROFMIX_TEMP", "0.5"))
        self.profmix_minmass = float(os.getenv("PROFMIX_MINMASS", "0.15"))

        # -- Profils par taille de panier (m -> dict de poids)
        self.basket_params_by_size = basket_params_by_size or {}

        # Charge artefacts + pré-calculs
        self._load_artifacts()

    # ──────────────────────────────────────────────────────────────────────
    # LOAD ARTEFACTS
    # ──────────────────────────────────────────────────────────────────────
    def _load_artifacts(self):
        adir = Path(self.artifact_dir)

        # mappings
        with open(adir/"mapping.pkl","rb") as f:
            (self.user_id_map, self.user_feature_map,
             self.item_id_map, self.item_feature_map) = pickle.load(f)
        self.idx2item = {v: k for k, v in self.item_id_map.items()}

        # noms des métiers indexés par u (utile pour les paires)
        self.user_names_by_idx = [None] * len(self.user_id_map)
        for name, u in self.user_id_map.items():
            self.user_names_by_idx[u] = name

        self.n_users = len(self.user_id_map)
        self.n_items = len(self.item_id_map)
        self.item_feat_inv = {v: k for k, v in self.item_feature_map.items()}

        # features / interactions
        self.user_features = load_npz(adir/"user_features.npz") if (adir/"user_features.npz").exists() else None
        self.item_features = load_npz(adir/"item_features.npz") if (adir/"item_features.npz").exists() else None
        self.interactions  = load_npz(adir/"interactions.npz").tocsr() if (adir/"interactions.npz").exists() else None

        # pré-calculs
        self.item_emb_norm    = np.load(adir/"item_emb_norm.npy", mmap_mode="r") if (adir/"item_emb_norm.npy").exists() else None
        self.train_pop        = np.load(adir/"train_pop.npy") if (adir/"train_pop.npy").exists() else None
        self.pop_by_prof_norm = np.load(adir/"pop_by_prof_norm.npy", mmap_mode="r") if (adir/"pop_by_prof_norm.npy").exists() else None
        self.top_popp_by_prof = np.load(adir/"top_popp_by_prof.npy", mmap_mode="r") if (adir/"top_popp_by_prof.npy").exists() else None

        # paires indexées (triées)
        self.pair_boost_idx = None
        if (adir/"pair_boost_idx.pkl").exists():
            with open(adir/"pair_boost_idx.pkl","rb") as f:
                self.pair_boost_idx = pickle.load(f)
        elif (adir/"pair_boost.pkl.gz").exists():
            # fallback: construit pair_boost_idx depuis la version brute
            with gzip.open(adir/"pair_boost.pkl.gz","rb") as f:
                pair_boost = pickle.load(f)
            pbi = {}
            for (prof, A), neigh in pair_boost.items():
                idxs, vals = [], []
                for b, s in neigh.items():
                    j = self.item_id_map.get(b)
                    if j is not None:
                        idxs.append(j); vals.append(float(s))
                if idxs:
                    idxs = np.asarray(idxs, dtype=np.int32)
                    vals = np.asarray(vals, dtype=np.float32)
                    order = np.argsort(idxs)
                    m = float(vals.max())
                    if m > 0: vals /= m
                    pbi[(str(prof), int(A))] = (idxs[order], vals[order])
            self.pair_boost_idx = pbi
        else:
            self.pair_boost_idx = {}

        # paires inverses
        self.pair_rev_idx = {}
        path_rev = adir/"pair_rev_idx.pkl"
        if path_rev.exists():
            with open(path_rev, "rb") as f:
                self.pair_rev_idx = pickle.load(f)

        # marque propre (0/1) par index item
        path_pl = adir/"item_is_pl.npy"
        if path_pl.exists():
            try:
                self.item_is_pl = np.load(path_pl, mmap_mode="r")
            except Exception:
                self.item_is_pl = np.zeros(self.n_items, dtype=np.int8)
        else:
            self.item_is_pl = np.zeros(self.n_items, dtype=np.int8)

        # modèle (si besoin pour recalculer emb)
        self.model = None
        if (self.item_emb_norm is None) and (adir/"model.pkl").exists() and not bool(int(os.getenv("DISABLE_MODEL_PICKLE","0"))):
            with open(adir/"model.pkl","rb") as f:
                self.model = pickle.load(f)

        # libellés
        self.product_meta = {}
        pm_path = adir / "product_meta.json.gz"
        if pm_path.exists():
            try:
                with gzip.open(pm_path, "rt", encoding="utf-8") as f:
                    self.product_meta = json.load(f)
            except Exception:
                self.product_meta = {}

        # pré-calculs dépendants
        self._precompute()

    # ──────────────────────────────────────────────────────────────────────
    # PRECOMPUTE
    # ──────────────────────────────────────────────────────────────────────
    def _precompute(self):
        adir = Path(self.artifact_dir)

        # 1) Embeddings items normalisés
        if self.item_emb_norm is None:
            if (self.model is not None) and (self.item_features is not None):
                _, emb = self.model.get_item_representations(features=self.item_features)
                emb = emb.astype(np.float32)
                emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
                self.item_emb_norm = emb
                try: np.save(adir/"item_emb_norm.npy", emb)
                except Exception: pass
            else:
                raise RuntimeError("item_emb_norm absent et aucun modèle dispo pour le recalcul.")

        # 2) Popularité & ordre global (+ normalisation 0..1)
        if self.train_pop is None:
            if self.interactions is not None:
                csr = self.interactions.tocsr()
                self.train_pop = np.asarray(csr.sum(axis=0)).ravel().astype(np.float32)
                try: np.save(adir/"train_pop.npy", self.train_pop)
                except Exception: pass
            else:
                self.train_pop = np.zeros(self.n_items, dtype=np.float32)

        self.pop_order_global = np.argsort(-self.train_pop)
        tp = self.train_pop.astype(np.float32, copy=True)
        lo, hi = float(tp.min()), float(tp.max())
        self.pop_norm_global = (tp - lo) / (hi - lo + 1e-8) if hi > lo else np.zeros_like(tp, dtype=np.float32)

        # 3) item_to_gamme (cap diversité éventuel)
        if not hasattr(self, "item_to_gamme") or self.item_to_gamme is None:
            path_itg = adir/"item_to_gamme.npy"
            if path_itg.exists():
                try:
                    self.item_to_gamme = np.load(path_itg, mmap_mode="r")
                except Exception:
                    self.item_to_gamme = None
            if self.item_to_gamme is None:
                if (self.item_features is None) or (self.item_feat_inv is None):
                    self.item_to_gamme = np.full(self.n_items, -1, dtype=np.int32)
                else:
                    itf = self.item_features.tocsr()
                    out = np.full(self.n_items, -1, dtype=np.int32)
                    for i in range(self.n_items):
                        cols = itf.indices[itf.indptr[i]:itf.indptr[i+1]]
                        gid = -1
                        for c in cols:
                            tok = self.item_feat_inv.get(int(c))
                            if isinstance(tok, str) and tok.startswith("gamme:"):
                                try: gid = int(tok.split(":",1)[1]); break
                                except: pass
                        out[i] = gid
                    self.item_to_gamme = out
                    try: np.save(path_itg, out)
                    except Exception: pass

        # 4) Paires ANY (agrégats globaux) + inverses ANY
        if not hasattr(self, "pair_boost_idx_any") or self.pair_boost_idx_any is None:
            self.pair_boost_idx_any = {}
            pbi = self.pair_boost_idx or {}
            from collections import defaultdict
            agg = defaultdict(dict)   # A -> {j:score_max}
            for (_prof, A), (idxs, vals) in pbi.items():
                A = int(A)
                for j, v in zip(idxs, vals):
                    jj = int(j)
                    prev = agg[A].get(jj)
                    agg[A][jj] = float(v) if prev is None else max(prev, float(v))
            for A, d in agg.items():
                if not d: continue
                idxs = np.fromiter(d.keys(), dtype=np.int32)
                vals = np.fromiter(d.values(), dtype=np.float32)
                order = np.argsort(idxs)
                self.pair_boost_idx_any[A] = (idxs[order], vals[order])

        if not hasattr(self, "pair_rev_idx_any") or self.pair_rev_idx_any is None:
            self.pair_rev_idx_any = {}
            if self.pair_rev_idx:
                from collections import defaultdict
                agg = defaultdict(dict)  # B -> {jA:score_max}
                for (_prof, B_id), (idxsA, vals) in self.pair_rev_idx.items():
                    for jA, v in zip(idxsA, vals):
                        prev = agg[int(B_id)].get(int(jA))
                        agg[int(B_id)][int(jA)] = v if prev is None else max(prev, v)
                for B_id, d in agg.items():
                    if d:
                        idxs = np.fromiter(d.keys(), dtype=np.int32)
                        vals = np.fromiter(d.values(), dtype=np.float32)
                        order = np.argsort(idxs)
                        self.pair_rev_idx_any[int(B_id)] = (idxs[order], vals[order])

        # 5) Centroïdes métiers pour auto-détection
        if hasattr(self, "top_popp_by_prof") and self.top_popp_by_prof is not None:
            d = self.item_emb_norm.shape[1]
            prof_centroids = np.zeros((self.n_users, d), dtype=np.float32)
            for u in range(self.n_users):
                topj = self.top_popp_by_prof[u]
                if isinstance(topj, np.ndarray) and topj.size:
                    V = self.item_emb_norm[topj]
                    c = V.mean(axis=0)
                    n = np.linalg.norm(c) + 1e-8
                    prof_centroids[u] = (c / n).astype(np.float32)
            self.prof_centroids = prof_centroids
        else:
            self.prof_centroids = np.zeros((self.n_users, self.item_emb_norm.shape[1]), dtype=np.float32)

        if not hasattr(self, "started_at"):
            self.started_at = time.time()

    # ──────────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────────
    def _cap_by_gamme(self, cand_sorted: np.ndarray, s_sorted: np.ndarray, cap: int) -> np.ndarray:
        """Limite le nombre d’items par gamme dans un ranking déjà trié."""
        if cap <= 0 or cand_sorted.size == 0:
            return cand_sorted
        out, seen = [], {}
        itg = getattr(self, "item_to_gamme", None)
        for j in cand_sorted:
            gid = int(itg[j]) if itg is not None and j < itg.size else -1
            if seen.get(gid, 0) < cap:
                out.append(j)
                seen[gid] = seen.get(gid, 0) + 1
        return np.asarray(out, dtype=np.int32)

    def enrich_items(self, ids: list[int]) -> list[dict]:
        """Enrichit les ids (libellés, univers/gamme) + expose is_pl si dispo."""
        out = []
        pm = self.product_meta or {}
        for pid in ids:
            pid = int(pid)
            meta = pm.get(str(pid)) or pm.get(pid)
            rec = {"id": pid}
            if meta: rec.update(meta)
            # expose is_pl si possible
            try:
                j = self.item_id_map.get(pid)
                if j is not None and j < len(self.item_is_pl):
                    rec["is_pl"] = int(self.item_is_pl[j])
            except Exception:
                pass
            out.append(rec)
        return out

    def _mmr(self, cand_idx: np.ndarray, scores: np.ndarray, k: int, lam: float) -> np.ndarray:
        """
        MMR (Maximal Marginal Relevance) : équilibre pertinence/diversité.
        Renvoie indices *globaux* des items retenus.
        """
        assert cand_idx.ndim == 1 and scores.ndim == 1 and cand_idx.size == scores.size
        n = cand_idx.size
        if n == 0 or k <= 0:
            return np.empty(0, dtype=np.int32)

        k = min(int(k), n)
        lam = float(lam)

        E = self.item_emb_norm[cand_idx]   # [n,d] déjà L2-normalisé
        chosen_local = []
        selected = np.zeros(n, dtype=bool)
        max_sim = np.zeros(n, dtype=np.float32)

        for _ in range(k):
            mmr = lam * scores - (1.0 - lam) * max_sim
            mmr_mask = np.where(selected, -1e9, mmr)
            i = int(np.argmax(mmr_mask))
            if selected[i]:
                break
            chosen_local.append(i)
            selected[i] = True

            sims = E @ E[i]
            max_sim = np.maximum(max_sim, sims.astype(np.float32))

        chosen_local = np.asarray(chosen_local, dtype=np.int32)
        return cand_idx[chosen_local]

    def _softmax_topk(self, sims: np.ndarray, topk: int, temp: float) -> tuple[np.ndarray, np.ndarray]:
        """Top-k + softmax(1/temp) restreint aux k meilleurs métiers."""
        if sims.ndim != 1 or sims.size == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
        k = min(topk, sims.size)
        part = np.argpartition(-sims, k-1)[:k]
        x = sims[part] / max(1e-6, temp)
        x -= x.max()
        w = np.exp(x); w /= (w.sum() + 1e-8)
        order = np.argsort(-w)
        return part[order].astype(np.int32), w[order].astype(np.float32)

    def status_dict(self) -> dict:
        def _shape(x):
            try: return list(x.shape)
            except: return None
        def _nnz(x):
            try: return int(x.nnz)
            except: return None

        return {
            "artifact_dir": str(self.artifact_dir),
            "uptime_sec": round(time.time() - getattr(self, "started_at", time.time()), 3),
            "counts": {
                "n_users": int(len(getattr(self, "user_id_map", {}) or {})),
                "n_items": int(len(getattr(self, "item_id_map", {}) or {})),
                "interactions_shape": _shape(getattr(self, "interactions", None)),
                "interactions_nnz": _nnz(getattr(self, "interactions", None)),
            },
            "precomputed_present": {
                "item_emb_norm": self.item_emb_norm is not None,
                "train_pop": self.train_pop is not None,
                "pop_by_prof_norm": self.pop_by_prof_norm is not None,
                "top_popp_by_prof": self.top_popp_by_prof is not None,
                "pair_boost_idx": self.pair_boost_idx is not None,
                "prof_centroids": hasattr(self, "prof_centroids"),
            },
            "weights": {
                "W_PAIR": self.W_PAIR, "W_EMB": self.W_EMB,
                "W_POPP": self.W_POPP, "W_POPG": self.W_POPG,
                "PAIR_ANY_WEIGHT": self.pair_any_weight,
            },
            "pools": {
                "TOP_PAIRS": self.TOP_PAIRS, "TOP_POPP": self.TOP_POPP,
                "TOP_POPG": self.TOP_POPG, "TOP_PAIRS_ANY_FRAC": self.top_pairs_any_frac,
                "PAIR_QUOTA": self.pair_quota,
            },
            "diversity": {
                "enabled": self.diversify, "mmr_lambda": self.mmr_lambda, "cap_per_gamme": self.cap_per_gamme,
            },
            "brand_bonus": {
                "present": self.item_is_pl is not None,
                "W_PL_BASE": self.W_PL_BASE,
                "W_PL_PER_BASK_PL": self.W_PL_PER_BASK_PL,
            },
            "profmix": {
                "topk": self.profmix_topk, "temp": self.profmix_temp, "minmass": self.profmix_minmass,
            },
            "product_meta": bool(getattr(self, "product_meta", {})),
        }

    # ──────────────────────────────────────────────────────────────────────
    # CANDIDATES (accepte un mélange de métiers)
    # ──────────────────────────────────────────────────────────────────────
    def _build_candidates_fast(self,
                               prof_ids: list[int] | None,
                               prof_w: np.ndarray | None,  # non utilisé pour le pool, utile au scoring
                               basket_ids: list[int],
                               TOP_PAIRS: int, TOP_POPP: int, TOP_POPG: int,
                               any_frac: float) -> np.ndarray:
        arrays = []

        # 1) PAIRS (PRO + ANY) pour chaque item du panier
        for a in basket_ids:
            if prof_ids:
                for u in prof_ids:
                    name = self.user_names_by_idx[int(u)]
                    entry = self.pair_boost_idx.get((name, int(a)))
                    if entry is not None:
                        nb_idx, _ = entry
                        if nb_idx.size:
                            arrays.append(nb_idx[:min(TOP_PAIRS, nb_idx.size)])
            entry_any = self.pair_boost_idx_any.get(int(a))
            if entry_any is not None and entry_any[0].size and any_frac > 0:
                nb_idx_any, _ = entry_any
                top_any = min(int(max(1, any_frac * TOP_PAIRS)), nb_idx_any.size)
                arrays.append(nb_idx_any[:top_any])

        # 2) Paires inverses (A proposés quand on observe b)
        if self.USE_REV:
            for b in basket_ids:
                if prof_ids:
                    for u in prof_ids:
                        name = self.user_names_by_idx[int(u)]
                        entry = self.pair_rev_idx.get((name, int(b)))
                        if entry is not None:
                            idxsA, _ = entry
                            if idxsA.size:
                                arrays.append(idxsA[:min(self.TOP_REV_PAIRS, idxsA.size)])
                entry_any = getattr(self, "pair_rev_idx_any", {}).get(int(b))
                if entry_any is not None and entry_any[0].size and self.REV_ANY_WEIGHT > 0:
                    idxsA, _ = entry_any
                    top_any = min(int(max(1, any_frac * self.TOP_REV_PAIRS)), idxsA.size)
                    arrays.append(idxsA[:top_any])

        # 3) Pop métier (union) + pop globale
        if prof_ids and TOP_POPP > 0 and (self.top_popp_by_prof is not None):
            for u in prof_ids:
                top_prof = self.top_popp_by_prof[int(u)]
                if isinstance(top_prof, np.ndarray):
                    arrays.append(top_prof[:min(TOP_POPP, top_prof.size)])
                else:
                    arrays.append(np.asarray(top_prof[:min(TOP_POPP, len(top_prof))], dtype=np.int32))

        if TOP_POPG > 0:
            arrays.append(self.pop_order_global[:TOP_POPG].astype(np.int32, copy=False))

        if not arrays:
            return np.empty(0, dtype=np.int32)

        # union + retrait du panier
        cand_idx = np.unique(np.concatenate(arrays).astype(np.int32, copy=False))
        if basket_ids:
            basket_j = [self.item_id_map.get(int(a)) for a in basket_ids if int(a) in self.item_id_map]
            basket_j = [j for j in basket_j if j is not None]
            if basket_j:
                cand_idx = np.setdiff1d(cand_idx, np.asarray(basket_j, dtype=np.int32), assume_unique=True)
        return cand_idx

    # ──────────────────────────────────────────────────────────────────────
    # PARAMS PAR TAILLE DE PANIER
    # ──────────────────────────────────────────────────────────────────────
    def _cfg_for_basket(self, m: int) -> dict:
        return (self.basket_params_by_size.get(m)
                or self.basket_params_by_size.get(str(m))
                or {})

    # ──────────────────────────────────────────────────────────────────────
    # RECOMMANDATION PANIER (profession optionnelle avec auto-mix)
    # ──────────────────────────────────────────────────────────────────────
    def recommend_panier(self, profession: str | None, product_ids: list[int], k: int = 10) -> list[int]:
        if not product_ids:
            raise ValueError("Panier vide")

        # 1) Panier filtré → ids connus du modèle
        basket = [int(p) for p in product_ids if int(p) in self.item_id_map]
        if not basket:
            raise ValueError("Panier vide ou items inconnus du modèle")

        # 2) Config selon taille de panier
        m = max(1, len(basket))
        cfg = self._cfg_for_basket(m)

        W_PAIR   = float(cfg.get("W_PAIR",   self.W_PAIR))
        W_EMB    = float(cfg.get("W_EMB",    self.W_EMB))
        W_POPP   = float(cfg.get("W_POPP",   self.W_POPP))
        W_POPG   = float(cfg.get("W_POPG",   self.W_POPG))
        PAIR_ANY = float(cfg.get("PAIR_ANY_WEIGHT", self.pair_any_weight))

        TOP_PAIRS = int(cfg.get("TOP_PAIRS", self.TOP_PAIRS))
        TOP_POPP  = int(cfg.get("TOP_POPP",  self.TOP_POPP))
        TOP_POPG  = int(cfg.get("TOP_POPG",  self.TOP_POPG))
        ANY_FRAC  = float(cfg.get("TOP_PAIRS_ANY_FRAC", self.top_pairs_any_frac))
        PAIR_QUOTA = int(cfg.get("PAIR_QUOTA", self.pair_quota))

        DIVERSIFY   = bool(cfg.get("DIVERSIFY", self.diversify))
        MMR_LAMBDA  = float(cfg.get("MMR_LAMBDA", self.mmr_lambda))
        CAP_GAMME   = int(cfg.get("CAP_PER_GAMME", self.cap_per_gamme))

        # MDD (profil override)
        W_PL_BASE = float(cfg.get("W_PL_BASE", self.W_PL_BASE))
        W_PL_PER_BASK_PL = float(cfg.get("W_PL_PER_BASK_PL", self.W_PL_PER_BASK_PL))

        # 3) Détermine le/les métiers (mono si fourni, sinon auto-mix)
        prof_ids, prof_w = None, None
        if profession and (profession in self.user_id_map):
            u = self.user_id_map[profession]
            prof_ids = [u]
            prof_w   = np.array([1.0], dtype=np.float32)
        else:
            b_idx = [self.item_id_map[a] for a in basket]
            v = self.item_emb_norm[b_idx].mean(axis=0)
            v /= (np.linalg.norm(v) + 1e-8)
            sims = self.prof_centroids @ v
            top_u, w = self._softmax_topk(sims, self.profmix_topk, self.profmix_temp)
            if top_u.size and float(w.sum()) >= self.profmix_minmass:
                prof_ids = top_u.tolist()
                prof_w   = w
            else:
                prof_ids, prof_w = None, None  # fallback global (pas de métier)

        # 4) Cache key (insensible à l’ordre du panier, encode les poids & le mix de métiers)
        profkey = tuple() if (prof_ids is None) else tuple((int(u), int(1000*float(w))) for u, w in zip(prof_ids, prof_w))
        key = ("PMIX", tuple(sorted(basket)), int(k),
               int(W_PAIR*1000), int(W_EMB*1000), int(W_POPP*1000), int(W_POPG*1000),
               TOP_PAIRS, TOP_POPP, TOP_POPG, int(ANY_FRAC*1000), int(PAIR_ANY*1000),
               int(DIVERSIFY), int(MMR_LAMBDA*1000), CAP_GAMME, PAIR_QUOTA, profkey,
               int(W_PL_BASE*1000), int(W_PL_PER_BASK_PL*1000))
        now = time.time()
        hit = self._cache.get(key)
        if hit is not None:
            recs, ts = hit
            if now - ts <= self._cache_ttl:
                return recs
            self._cache.pop(key, None)

        # 5) Pool candidats
        cand_idx = self._build_candidates_fast(prof_ids, prof_w, basket, TOP_PAIRS, TOP_POPP, TOP_POPG, ANY_FRAC)
        n = cand_idx.size
        if n == 0:
            return []

        # 6) Scoring vectorisé
        s = np.zeros(n, dtype=np.float32)

        # (a) PAIRES (PRO mixé + ANY)
        if W_PAIR > 0:
            for a in basket:
                if prof_ids:
                    for uu, ww in zip(prof_ids, prof_w):
                        name = self.user_names_by_idx[int(uu)]
                        entry = self.pair_boost_idx.get((name, int(a)))
                        if entry is not None and entry[0].size:
                            nb_idx, nb_vals = entry
                            _, ia, ib = np.intersect1d(nb_idx, cand_idx, assume_unique=True, return_indices=True)
                            if ia.size:
                                s[ib] += (W_PAIR * float(ww)) * nb_vals[ia]
                entry_any = self.pair_boost_idx_any.get(int(a))
                if PAIR_ANY > 0 and entry_any is not None and entry_any[0].size:
                    nb_idx, nb_vals = entry_any
                    _, ia, ib = np.intersect1d(nb_idx, cand_idx, assume_unique=True, return_indices=True)
                    if ia.size:
                        s[ib] += (W_PAIR * PAIR_ANY) * nb_vals[ia]

        # (a2) PAIRES inverses (proposer A quand on observe b)
        if W_PAIR > 0 and self.USE_REV:
            for b in basket:
                if prof_ids:
                    for uu, ww in zip(prof_ids, prof_w):
                        name = self.user_names_by_idx[int(uu)]
                        entry = self.pair_rev_idx.get((name, int(b)))
                        if entry is not None and entry[0].size:
                            idxsA, valsA = entry
                            _, ia, ib = np.intersect1d(idxsA, cand_idx, assume_unique=True, return_indices=True)
                            if ia.size:
                                s[ib] += (W_PAIR * float(ww)) * valsA[ia]
                entry_any = getattr(self, "pair_rev_idx_any", {}).get(int(b))
                if self.REV_ANY_WEIGHT > 0 and entry_any is not None and entry_any[0].size:
                    idxsA, valsA = entry_any
                    _, ia, ib = np.intersect1d(idxsA, cand_idx, assume_unique=True, return_indices=True)
                    if ia.size:
                        s[ib] += (W_PAIR * self.REV_ANY_WEIGHT) * valsA[ia]

        # (b) EMBEDDINGS: E_cand @ mean(E_basket)
        if W_EMB > 0:
            b_idx = [self.item_id_map[a] for a in basket]
            v = self.item_emb_norm[b_idx].mean(axis=0)
            s += W_EMB * (self.item_emb_norm[cand_idx] @ v)

        # (c) POPULARES: PRO mixée + GLOBALE
        if W_POPP > 0 and prof_ids and (self.pop_by_prof_norm is not None):
            for uu, ww in zip(prof_ids, prof_w):
                s += (W_POPP * float(ww)) * self.pop_by_prof_norm[int(uu)][cand_idx]
        if W_POPG > 0:
            s += W_POPG * self.pop_norm_global[cand_idx]

        # (d) BONUS couverture (nombre d’items du panier reliés au candidat)
        if self.GAMMA_COVER > 0:
            cover = np.zeros(n, dtype=np.int16)

            def _bump(indices):
                if indices is None: return
                _, _, ib = np.intersect1d(indices, cand_idx, assume_unique=True, return_indices=True)
                if ib.size: cover[ib] += 1

            for a in basket:
                if prof_ids:
                    for uu in prof_ids:
                        name = self.user_names_by_idx[int(uu)]
                        entry = self.pair_boost_idx.get((name, int(a)))
                        if entry is not None and entry[0].size: _bump(entry[0])
                entry_any = self.pair_boost_idx_any.get(int(a))
                if entry_any is not None and entry_any[0].size: _bump(entry_any[0])

                if self.USE_REV:
                    if prof_ids:
                        for uu in prof_ids:
                            name = self.user_names_by_idx[int(uu)]
                            entry = self.pair_rev_idx.get((name, int(a)))
                            if entry is not None and entry[0].size: _bump(entry[0])
                    entry_any = getattr(self, "pair_rev_idx_any", {}).get(int(a))
                    if entry_any is not None and entry_any[0].size: _bump(entry_any[0])

            s += self.GAMMA_COVER * cover.astype(np.float32)

        # (e) BONUS MDD : additif W_PL_BASE + W_PL_PER_BASK_PL * ratio_MDD_dans_panier
        if (W_PL_BASE != 0.0) or (W_PL_PER_BASK_PL != 0.0):
            b_idx = [self.item_id_map[a] for a in basket]
            basket_pl_ratio = float(self.item_is_pl[b_idx].mean()) if len(b_idx) else 0.0
            pl_mask = self.item_is_pl[cand_idx].astype(np.float32, copy=False)
            pl_bonus = W_PL_BASE + (W_PL_PER_BASK_PL * basket_pl_ratio)
            if pl_bonus != 0.0:
                s += pl_bonus * pl_mask

        # 7) Tri / diversité / quotas
        order = np.argsort(-s)
        cand_sorted = cand_idx[order]
        s_sorted    = s[order]

        if self.diversify or CAP_GAMME > 0:
            if CAP_GAMME > 0:
                capped = self._cap_by_gamme(cand_sorted, s_sorted, CAP_GAMME)
                if capped.size == 0:
                    return []
                # réaligne les scores
                pos = {int(j): i for i, j in enumerate(cand_idx[order])}
                s_sorted = np.array([s[pos[int(j)]] for j in capped], dtype=np.float32)
                cand_sorted = capped

            pool_size = min(5 * k, cand_sorted.size)
            top_idx_global = self._mmr(cand_sorted[:pool_size], s_sorted[:pool_size], k, self.mmr_lambda)
            recs = [self.idx2item[i] for i in top_idx_global]
        else:
            # éventuel quota "pair" (optionnel; ici désactivé par défaut)
            if self.pair_quota > 0:
                # étiquette pair/non-pair
                is_pair = np.zeros(cand_sorted.size, dtype=bool)
                for a in basket:
                    for entry in (self.pair_boost_idx.get((self.user_names_by_idx[int(prof_ids[0])] if (prof_ids and len(prof_ids)==1) else None, int(a))),
                                  self.pair_boost_idx_any.get(int(a))):
                        if entry is None or not entry[0].size:
                            continue
                        nb_idx, _ = entry
                        _, _, ib = np.intersect1d(nb_idx, cand_sorted, assume_unique=True, return_indices=True)
                        if ib.size:
                            is_pair[ib] = True

                pair_ids    = cand_sorted[is_pair]
                pair_scores = s_sorted[is_pair]
                non_ids     = cand_sorted[~is_pair]
                non_scores  = s_sorted[~is_pair]

                k_pair = min(self.pair_quota, pair_ids.size, k)
                k_non  = k - k_pair
                op = np.argsort(-pair_scores)[:k_pair]
                on = np.argsort(-non_scores)[:k_non]
                topk_idx = np.concatenate([pair_ids[op], non_ids[on]])
                recs = [self.idx2item[i] for i in topk_idx]
            else:
                recs = [self.idx2item[i] for i in cand_sorted[:k]]

        # cache LRU (simple)
        self._cache[key] = (recs, now)
        self._cache_order.append(key)
        if len(self._cache_order) > self._cache_max:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)

        return recs
