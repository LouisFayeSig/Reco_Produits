#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
(Re)génère les artefacts pré-calculés nécessaires à l'API pour un run LightFM.

Usage :
  # vérifier le dernier run sous artifacts/ (sans rien écrire)
  python tools/save_precomputed.py --root artifacts --check-only

  # compléter un run précis (écrit ce qui manque)
  python tools/save_precomputed.py --run-dir artifacts/run_20250924_122145

  # compléter le dernier run + inférer item_to_gamme & product_meta depuis df_details_produits.parquet
  python tools/save_precomputed.py --root artifacts --details data/df_details_produits.parquet
"""

from __future__ import annotations
import os, re, json, time, gzip, pickle, sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from scipy.sparse import load_npz

RE_RUN = re.compile(r"^run_\d{8}_\d{6}.*")  # run_YYYYMMDD_HHMMSS[_tag]

NEEDED = [
    "mapping.pkl",
    "user_features.npz",
    "item_features.npz",
    "interactions.npz",
]

# Cibles attendues par l'API (au moins une des deux pour les paires)
TARGETS = [
    "item_emb_norm.npy",
    "train_pop.npy",
    "pop_by_prof_norm.npy",
    "top_popp_by_prof.npy",
    "item_is_pl.npy",
    # paires :
    # soit pair_boost_idx.pkl présent
    # soit pair_boost.pkl.gz présent et on fabrique idx
]

OPTIONAL_TARGETS = [
    "item_to_gamme.npy",
    "product_meta.json.gz",
    "pair_boost_idx.pkl",   # peut être recalculé depuis pair_boost.pkl.gz
]


def tsec() -> float:
    return time.perf_counter()


def _latest_run_dir(root: Path) -> Optional[Path]:
    if not root.exists() or not root.is_dir():
        return None
    cands = [d for d in root.iterdir() if d.is_dir() and RE_RUN.match(d.name)]
    if not cands:
        return None
    cands.sort(key=lambda d: d.name, reverse=True)
    return cands[0]


def resolve_run_dir(run_dir: Optional[str], root: Optional[str]) -> Path:
    if run_dir:
        p = Path(run_dir)
        if p.is_symlink():
            p = p.resolve()
        if not p.exists():
            raise FileNotFoundError(f"run-dir introuvable: {p}")
        return p
    if root:
        r = Path(root)
        if r.is_symlink():
            r = r.resolve()
        # si root pointe déjà sur un run (mapping présent), on l’utilise
        if (r / "mapping.pkl").exists():
            return r
        last = _latest_run_dir(r)
        if last:
            return last
        raise FileNotFoundError(f"Aucun run_* trouvé sous {root}")
    # défaut : artifacts/run_current s’il existe, sinon artifacts/ dernier
    default_root = Path("artifacts")
    cur = default_root / "run_current"
    if cur.exists():
        return cur.resolve()
    last = _latest_run_dir(default_root)
    if last:
        return last
    raise FileNotFoundError("Aucun run valide trouvé (essaye --root artifacts ou --run-dir ...)")


def load_mapping(adir: Path):
    with open(adir / "mapping.pkl", "rb") as f:
        user_id_map, user_feat_map, item_id_map, item_feat_map = pickle.load(f)
    # normalise en dict natifs
    return dict(user_id_map), dict(user_feat_map), dict(item_id_map), dict(item_feat_map)


def ensure_item_emb_norm(adir: Path) -> bool:
    """Construit item_emb_norm.npy si absent (nécessite model.pkl + item_features.npz)."""
    out = adir / "item_emb_norm.npy"
    if out.exists():
        return False
    model_pkl = adir / "model.pkl"
    if not model_pkl.exists():
        raise RuntimeError("item_emb_norm.npy manquant et model.pkl absent : lancer dans l'environnement training, ou regénérer via train_daily.py.")
    try:
        with open(model_pkl, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Impossible de charger model.pkl (version NumPy différente ?) : {e}")
    item_features = load_npz(adir / "item_features.npz")
    _, emb = model.get_item_representations(features=item_features)
    emb = emb.astype(np.float32)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    np.save(out, emb)
    return True


def ensure_popularity(adir: Path, user_id_map: Dict, topk_per_prof: int = 3000) -> Dict[str, bool]:
    """Construit train_pop.npy / pop_by_prof_norm.npy / top_popp_by_prof.npy si absents."""
    changed = {"train_pop": False, "pop_by_prof_norm": False, "top_popp_by_prof": False}

    inter = load_npz(adir / "interactions.npz").tocsr()

    tp = adir / "train_pop.npy"
    if not tp.exists():
        train_pop = np.asarray(inter.sum(axis=0)).ravel().astype(np.float32)
        np.save(tp, train_pop)
        changed["train_pop"] = True

    ppf = adir / "pop_by_prof_norm.npy"
    tpp = adir / "top_popp_by_prof.npy"
    if not ppf.exists() or not tpp.exists():
        # pop métier normalisée
        profs_sorted = sorted(user_id_map.keys(), key=lambda p: user_id_map[p])
        rows = []
        for prof in profs_sorted:
            u = user_id_map[prof]
            vec = np.asarray(inter.getrow(u).todense()).ravel().astype(np.float32)
            vmin, vmax = float(vec.min()), float(vec.max())
            if vmax > vmin:
                rows.append(((vec - vmin) / (vmax - vmin + 1e-8)).astype(np.float32))
            else:
                rows.append(np.zeros_like(vec, dtype=np.float32))
        pop_by_prof = np.vstack(rows)
        if not ppf.exists():
            np.save(ppf, pop_by_prof)
            changed["pop_by_prof_norm"] = True
        # top par métier (borné à n_items)
        top_k = min(int(topk_per_prof), pop_by_prof.shape[1])
        top_popp = np.argsort(-pop_by_prof, axis=1)[:, :top_k]
        if not tpp.exists():
            np.save(tpp, top_popp)
            changed["top_popp_by_prof"] = True

    return changed


def ensure_pair_boost_idx(adir: Path, item_id_map: Dict[int, int]) -> bool:
    """Construit pair_boost_idx.pkl depuis pair_boost.pkl.gz si idx absent."""
    idx_path = adir / "pair_boost_idx.pkl"
    if idx_path.exists():
        return False
    src = adir / "pair_boost.pkl.gz"
    if not src.exists():
        # rien à faire
        return False
    with gzip.open(src, "rb") as f:
        pair_boost = pickle.load(f)

    pair_boost_idx = {}
    for (prof, A), neigh in pair_boost.items():
        idxs, vals = [], []
        for b, s in neigh.items():
            j = item_id_map.get(b)
            if j is None:
                continue
            idxs.append(int(j))
            vals.append(float(s))
        if idxs:
            idxs = np.asarray(idxs, dtype=np.int32)
            vals = np.asarray(vals, dtype=np.float32)
            m = float(vals.max()) if vals.size else 0.0
            if m > 0:
                vals = vals / m
            order = np.argsort(idxs)
            pair_boost_idx[(str(prof), int(A))] = (idxs[order], vals[order])

    with open(idx_path, "wb") as f:
        pickle.dump(pair_boost_idx, f, protocol=pickle.HIGHEST_PROTOCOL)
    return True


def ensure_item_to_gamme_and_meta(adir: Path, details_path: Optional[str], item_id_map: Dict[int, int]) -> Dict[str, bool]:
    """Construit item_to_gamme.npy et product_meta.json.gz si absents et si détails fournis."""
    changed = {"item_to_gamme": False, "product_meta": False}
    itg = adir / "item_to_gamme.npy"
    pm = adir / "product_meta.json.gz"

    if (itg.exists() and pm.exists()) or not details_path:
        return changed

    try:
        import pandas as pd
    except Exception:
        # si pandas absent dans l'env, on ne force pas
        return changed

    details = Path(details_path)
    if not details.exists():
        return changed

    df = pd.read_parquet(details)

    # item_to_gamme
    if not itg.exists():
        n_items = int(max(item_id_map.values())) + 1 if item_id_map else 0
        out = np.full(n_items, -1, dtype=np.int32)
        idx = df.set_index("ID_PRODUIT", drop=False)
        for pid, j in item_id_map.items():
            if pid in idx.index:
                row = idx.loc[pid]
                gid = int(row["ID_GAMME"]) if not pd.isna(row.get("ID_GAMME")) else -1
                out[int(j)] = gid
        np.save(itg, out)
        changed["item_to_gamme"] = True

    # product_meta
    if not pm.exists():
        def _s(x):
            if x is None or (isinstance(x, float) and np.isnan(x)): return None
            return str(x).strip()
        product_meta = {}
        idx = df.set_index("ID_PRODUIT", drop=False)
        for pid in item_id_map.keys():
            if pid in idx.index:
                row = idx.loc[pid]
                product_meta[int(pid)] = {
                    "lib": _s(row.get("LIB_PRODUIT")),
                    "id_gamme": int(row["ID_GAMME"]) if not pd.isna(row.get("ID_GAMME")) else None,
                    "id_univers": int(row["ID_UNIVERS"]) if not pd.isna(row.get("ID_UNIVERS")) else None,
                    "lib_gamme": _s(row.get("LIB_GAMME")),
                    "lib_univers": _s(row.get("LIB_UNIVERS")),
                }
            else:
                product_meta[int(pid)] = {"lib": None, "id_gamme": None, "id_univers": None,
                                          "lib_gamme": None, "lib_univers": None}
        with gzip.open(pm, "wt", encoding="utf-8") as f:
            json.dump(product_meta, f, ensure_ascii=False)
        changed["product_meta"] = True

    return changed


def integrity(adir: Path) -> Dict[str, Any]:
    """Retourne un rapport d'intégrité des artefacts nécessaires à l'API."""
    report = {"base_ok": True, "missing_base": [], "targets_ok": True, "missing_targets": [], "pair_source": None}
    # base
    for name in NEEDED:
        if not (adir / name).exists():
            report["missing_base"].append(name)
    report["base_ok"] = len(report["missing_base"]) == 0

    # cibles
    miss = []
    for name in TARGETS:
        if not (adir / name).exists():
            miss.append(name)

    # paires: il faut au moins idx OU brut
    have_idx = (adir / "pair_boost_idx.pkl").exists()
    have_raw = (adir / "pair_boost.pkl.gz").exists()
    if not have_idx and not have_raw:
        miss.append("pair_boost_idx.pkl (ou pair_boost.pkl.gz)")
        report["pair_source"] = "missing"
    else:
        report["pair_source"] = "idx" if have_idx else "raw"

    report["missing_targets"] = miss
    report["targets_ok"] = len(miss) == 0
    return report


def main():
    import argparse
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--run-dir", help="Chemin d'un run spécifique (ex: artifacts/run_20250924_122145)")
    g.add_argument("--root", help="Dossier racine des runs (ex: artifacts) → prend le dernier run_*")
    ap.add_argument("--details", help="Parquet df_details_produits pour générer item_to_gamme & product_meta", default=None)
    ap.add_argument("--topk-per-prof", type=int, default=3000)
    ap.add_argument("--check-only", action="store_true", help="Ne rien écrire, juste vérifier l'intégrité (exit 0/2)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    run_dir = resolve_run_dir(args.run_dir, args.root)
    print(f"[INFO] run_dir = {run_dir}")
    t0 = tsec()

    rep0 = integrity(run_dir)
    if args.check_only:
        print(json.dumps(rep0, indent=2, ensure_ascii=False))
        if rep0["base_ok"] and rep0["targets_ok"]:
            sys.exit(0)
        else:
            sys.exit(2)

    # base requis
    if not rep0["base_ok"]:
        print("[ERR] Artefacts de base manquants:", rep0["missing_base"])
        sys.exit(2)

    # mapping
    user_id_map, user_feat_map, item_id_map, item_feat_map = load_mapping(run_dir)

    # 1) embeddings items normalisés
    try:
        changed_emb = ensure_item_emb_norm(run_dir)
        if changed_emb: print("[OK] item_emb_norm.npy (créé)")
    except Exception as e:
        print(f"[WARN] item_emb_norm.npy: {e}")

    # 2) popularités / top
    ch_pop = ensure_popularity(run_dir, user_id_map, topk_per_prof=args.topk_per_prof)
    for k, v in ch_pop.items():
        if v: print(f"[OK] {k} (créé)")

    # 3) paires indexées
    ch_pairs = ensure_pair_boost_idx(run_dir, item_id_map)
    if ch_pairs:
        print("[OK] pair_boost_idx.pkl (créé)")

    # 4) item_to_gamme & product_meta
    ch_meta = ensure_item_to_gamme_and_meta(run_dir, args.details, item_id_map)
    for k, v in ch_meta.items():
        if v: print(f"[OK] {k} (créé)")

    # rapport final
    rep = integrity(run_dir)
    print(json.dumps(rep, indent=2, ensure_ascii=False))

    dt = tsec() - t0
    if rep["base_ok"] and rep["targets_ok"]:
        print(f"[DONE] Intégrité OK | ⏱️ {dt:.2f}s")
        sys.exit(0)
    else:
        print(f"[DONE] Intégrité INCOMPLETE | ⏱️ {dt:.2f}s")
        sys.exit(2)


if __name__ == "__main__":
    main()
