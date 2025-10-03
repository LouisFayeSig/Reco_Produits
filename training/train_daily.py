#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Entraînement quotidien LightFM + export artefacts et pré-calculs.

Points clés :
- Possibilité d’entraîner sur tout le catalogue (--include-all-items) sans exploser le nb de features
  grâce à --cold-identity-token (par défaut 0) : le token 'prod:PID' n'est ajouté qu'aux items
  avec ventes (min-occ), sinon ils reçoivent un token générique 'nof:1'.
- Paires co-achat robustes (par métier + global) → lift positif, index direct et inverse.
- MDD (marque propre) : token 'brand_pl' + vecteur binaire exporté (item_is_pl.npy) pour
  un bonus côté API si besoin.
"""

from __future__ import annotations
import os, ast, json, time, gzip, pickle, platform
from pathlib import Path
from typing import Dict, Any
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.sparse import save_npz, csr_matrix
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import auc_score
from sklearn.model_selection import train_test_split


# ----------------------------- Utils -----------------------------
def tsec() -> float: return time.perf_counter()
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def as_list(x) -> list:
    if x is None or (isinstance(x, float) and np.isnan(x)): return []
    if isinstance(x, (list, tuple)): return list(x)
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, str):
        xs = x.strip()
        # list-likes
        if (xs.startswith("[") and xs.endswith("]")) or (xs.startswith("(") and xs.endswith(")")):
            try:
                val = ast.literal_eval(xs)
                if isinstance(val, (list, tuple, np.ndarray)):
                    return list(val)
            except Exception:
                pass
        # csv
        if "," in xs:
            return [s.strip() for s in xs.split(",") if s.strip()]
        return [xs]
    return [x]

def norm01(vec: np.ndarray) -> np.ndarray:
    vmin, vmax = float(vec.min()), float(vec.max())
    if vmax <= vmin: return np.zeros_like(vec, dtype=np.float32)
    return ((vec - vmin) / (vmax - vmin + 1e-8)).astype(np.float32)


# ============================= MAIN =============================
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ventes", default="training/data/df_ventes_grouped.parquet", help="Parquet df_ventes_grouped")
    ap.add_argument("--details", default="training/data/df_details_produits.parquet", help="Parquet df_details_produits")
    ap.add_argument("--pairs", default="training/data/df_couple_produit.parquet", help="Parquet df_couple_produit")
    ap.add_argument("--artifact-root", default="artifacts")
    ap.add_argument("--run-tag", default="")

    # LightFM
    ap.add_argument("--no-components", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--loss", type=str, default="warp")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--patience", type=int, default=5, help="nb d'epochs sans amélioration avant arrêt")

    # Build / filtrages
    ap.add_argument("--min-occ", type=int, default=1, help="min occurrences produits dans ventes")
    ap.add_argument("--include-all-items", action="store_true",
                    help="inclure tout le catalogue (df_details) dans items=")
    ap.add_argument("--cold-identity-token", type=int, default=0, choices=[0,1],
                    help="1=ajoute 'prod:PID' même aux items sans ventes. 0 par défaut (réduit les features)")
    ap.add_argument("--symmetrize-pairs", action="store_true")
    ap.add_argument("--max-items", type=int, default=0, help="cap max sur nb d'items (0 = aucun cap)")
    ap.add_argument("--topk-per-prof", type=int, default=3000)
    ap.add_argument("--make-current-symlink", action="store_true")

    args = ap.parse_args()
    t0 = tsec()

    ventes_pq  = Path(args.ventes)
    details_pq = Path(args.details)
    pairs_pq   = Path(args.pairs)

    # ----------------------------- Load -----------------------------
    print("== Load data ==")
    df_ventes  = pd.read_parquet(ventes_pq)
    df_details = pd.read_parquet(details_pq)
    df_pairs   = pd.read_parquet(pairs_pq)

    # Types sûrs
    df_ventes["PROFESSION_CLIENT"] = df_ventes["PROFESSION_CLIENT"].astype(str)
    df_ventes["NB_OCCURENCES_COMMANDE"] = df_ventes.get("NB_OCCURENCES_COMMANDE", 1).astype(int)
    df_ventes["LIST_ID_PRODUITS"] = df_ventes["LIST_ID_PRODUITS"].apply(lambda lst: [int(x) for x in lst])

    for c in ["ID_PRODUIT", "ID_UNIVERS", "ID_GAMME"]:
        if c in df_details.columns:
            df_details[c] = pd.to_numeric(df_details[c], errors="coerce").astype("Int64")

    # ID_MARQUE binaire : 1 si MDD, 0 sinon (manquant -> 0)
    if "ID_MARQUE" in df_details.columns:
        df_details["ID_MARQUE"] = pd.to_numeric(df_details["ID_MARQUE"], errors="coerce").fillna(0).astype(int)
        df_details["ID_MARQUE"] = (df_details["ID_MARQUE"] == 1).astype(int)
    else:
        df_details["ID_MARQUE"] = 0

    # pairs types
    df_pairs["PROFESSION_CLIENT"] = df_pairs["PROFESSION_CLIENT"].apply(as_list)
    for c in ["ID_PRODUIT_A","ID_PRODUIT_B","LIST_ID_UNIVERS_A","LIST_ID_UNIVERS_B","LIST_ID_GAMME_A","LIST_ID_GAMME_B"]:
        if c in df_pairs.columns:
            df_pairs[c] = pd.to_numeric(df_pairs[c], errors="coerce").astype("Int64")

    # ----------------------------- Popularité & univers d'items -----------------------------
    print("== Build popularity & item universe ==")
    from collections import Counter
    cnt = Counter()
    for r in df_ventes.itertuples(index=False):
        w = int(getattr(r, "NB_OCCURENCES_COMMANDE", 1))
        for pid in getattr(r, "LIST_ID_PRODUITS"):
            cnt[int(pid)] += w

    sales_products = {p for p, c in cnt.items() if c >= args.min_occ}

    pair_products = set()
    if "ID_PRODUIT_A" in df_pairs.columns and "ID_PRODUIT_B" in df_pairs.columns:
        pair_products |= set(df_pairs["ID_PRODUIT_A"].dropna().astype(int).values.tolist())
        pair_products |= set(df_pairs["ID_PRODUIT_B"].dropna().astype(int).values.tolist())

    catalog_products = set()
    if args.include_all_items and "ID_PRODUIT" in df_details.columns:
        catalog_products = set(df_details["ID_PRODUIT"].dropna().astype(int).values.tolist())

    if args.include_all_items:
        all_products_set = sales_products | pair_products | catalog_products
    else:
        all_products_set = sales_products

    if not all_products_set and pair_products:
        all_products_set = pair_products

    # Cap optionnel
    if args.max_items and args.max_items > 0 and len(all_products_set) > args.max_items:
        def _score(pid): return cnt.get(pid, 0)
        all_products = [p for p, _ in sorted(((p, _score(p)) for p in all_products_set),
                                            key=lambda kv: (-kv[1], kv[0]))[:args.max_items]]
    else:
        all_products = sorted(all_products_set)

    print(f"   produits (ventes, min-occ={args.min_occ}) : {len(sales_products)}")
    print(f"   + pairs uniques : {len(pair_products)} | + catalog : {len(catalog_products)}")
    print(f"   => all_products retenus : {len(all_products)} (max-items={args.max_items})")

    # Users
    prof_from_ventes = set(map(str, df_ventes["PROFESSION_CLIENT"].unique().tolist()))
    prof_from_pairs = set()
    for lst in df_pairs["PROFESSION_CLIENT"].tolist():
        for p in as_list(lst): prof_from_pairs.add(str(p))
    all_professions = sorted(prof_from_ventes | prof_from_pairs)
    print(f"Professions: {len(all_professions)}")

    # ----------------------------- Features items -----------------------------
    print("== Build item feature tokens ==")
    details_idx = df_details.set_index("ID_PRODUIT", drop=False)

    item_feature_tokens: Dict[int, list[str]] = {}
    for pid in all_products:
        pid = int(pid)
        toks = ["nof:1"]  # token commun pour relier les items “cold”
        has_sales = pid in sales_products

        # identité produit uniquement si ventes ou si forcé par flag
        if has_sales or args.cold_identity_token == 1:
            toks.append(f"prod:{pid}")

        if pid in details_idx.index:
            row = details_idx.loc[pid]
            g = int(row["ID_GAMME"])   if not pd.isna(row["ID_GAMME"])   else -1
            u = int(row["ID_UNIVERS"]) if not pd.isna(row["ID_UNIVERS"]) else -1
            if g >= 0: toks.append(f"gamme:{g}")
            if u >= 0: toks.append(f"univers:{u}")
            # MDD
            if int(row.get("ID_MARQUE", 0)) == 1:
                toks.append("brand_pl")

        item_feature_tokens[pid] = toks

    # ----------------------------- Paires (co-achat) → Lift -----------------------------
    print("== Build pairs (counts -> lift) ==")
    # Comptes produits globaux
    cnt_prod = cnt  # déjà calculé
    N = float(sum(cnt_prod.values())) if cnt_prod else 1.0

    # pair counts par métier : (prof, A) -> {B: occ_AB}
    pair_counts: Dict[tuple[str,int], Dict[int,int]] = {}

    def inc_pair(prof: str, A: int, B: int, w: int = 1):
        if A == B: return
        d = pair_counts.setdefault((prof, int(A)), {})
        d[int(B)] = d.get(int(B), 0) + int(w)

    # 1) À partir de df_pairs (si dispo)
    if {"ID_PRODUIT_A","ID_PRODUIT_B","OCCURENCE_PAIR"}.issubset(df_pairs.columns):
        for r in df_pairs.itertuples(index=False):
            A = int(getattr(r, "ID_PRODUIT_A"))
            B = int(getattr(r, "ID_PRODUIT_B"))
            occ = int(getattr(r, "OCCURENCE_PAIR", 1))
            profs = as_list(getattr(r, "PROFESSION_CLIENT"))
            for p in profs:
                inc_pair(str(p), A, B, occ)
                if args.symmetrize_pairs:
                    inc_pair(str(p), B, A, occ)

    # 2) Compléter depuis co-baskets ventes (par métier)
    for r in df_ventes.itertuples(index=False):
        pro = str(getattr(r, "PROFESSION_CLIENT"))
        prods = [int(x) for x in getattr(r, "LIST_ID_PRODUITS") if int(x) in item_feature_tokens]
        prods = list(set(prods))
        # toutes les paires ordonnées A->B
        for A, B in combinations(prods, 2):
            inc_pair(pro, A, B, 1)
            inc_pair(pro, B, A, 1)

    # 3) Convertir en lift positif et normaliser par max
    pair_boost: Dict[tuple[str,int], Dict[int,float]] = {}
    for (prof, A), neigh in pair_counts.items():
        occ_A = max(1, cnt_prod.get(A, 0))
        new = {}
        for B, occ_AB in neigh.items():
            occ_B = max(1, cnt_prod.get(B, 0))
            pcond = occ_AB / occ_A            # p(B|A)
            pB    = occ_B / N                 # p(B)
            lift  = np.log((pcond / (pB + 1e-12)) + 1e-12)  # log-lift
            if lift > 0:
                new[int(B)] = float(lift)
        # normalisation par max pour stabilité
        if new:
            m = max(new.values())
            if m > 0:
                for b in list(new.keys()):
                    new[b] = new[b] / m
        if new:
            pair_boost[(prof, int(A))] = new

    # ----------------------------- Dataset & matrices -----------------------------
    print("== Dataset.fit ==")
    dataset = Dataset()
    all_item_feat_tokens = sorted({t for pid in all_products for t in item_feature_tokens.get(pid, ["nof:1"])})
    dataset.fit(
        users=all_professions,
        items=all_products,
        user_features=all_professions,      # identité métier
        item_features=all_item_feat_tokens  # tokens items
    )

    print("== Build features/interactions ==")
    user_features = dataset.build_user_features([(u, [u]) for u in all_professions])
    item_features = dataset.build_item_features([(pid, item_feature_tokens.get(pid, ["nof:1"])) for pid in all_products])

    # interactions = ventes uniquement
    triples = []
    dropped = 0
    for r in df_ventes.itertuples(index=False):
        prof = str(getattr(r, "PROFESSION_CLIENT"))
        w = float(getattr(r, "NB_OCCURENCES_COMMANDE", 1))
        for pid in getattr(r, "LIST_ID_PRODUITS"):
            pid = int(pid)
            if pid in item_feature_tokens:
                triples.append((prof, pid, w))
            else:
                dropped += 1
    interactions, _ = dataset.build_interactions(triples)

    # Cast mémoire
    user_features = user_features.tocsr().astype(np.float32)
    item_features = item_features.tocsr().astype(np.float32)
    interactions  = interactions.tocsr().astype(np.float32)

    print(f"Shapes: users={user_features.shape} items={item_features.shape} inter={interactions.shape} nnz={interactions.nnz}")

    # ----------------------------- Train + early stopping -----------------------------
    rows, cols = interactions.nonzero()
    data = interactions.data
    idx = np.arange(len(rows))
    train_idx, val_idx = train_test_split(idx, test_size=0.1, random_state=42)

    train_mat = csr_matrix((data[train_idx], (rows[train_idx], cols[train_idx])), shape=interactions.shape)
    val_mat   = csr_matrix((data[val_idx],   (rows[val_idx],   cols[val_idx])),   shape=interactions.shape)

    print("== Train LightFM avec early stopping ==")
    model = LightFM(no_components=args.no_components, learning_rate=args.lr, loss=args.loss, random_state=42)

    best_auc = -np.inf
    wait = 0
    patience = args.patience
    min_delta = 0.001

    for epoch in range(args.epochs):
        model.fit_partial(train_mat,
                          user_features=user_features, item_features=item_features,
                          epochs=1, num_threads=args.threads)

        auc = auc_score(model, val_mat, train_interactions=train_mat,
                        user_features=user_features, item_features=item_features,
                        num_threads=args.threads).mean()
        print(f"Epoch {epoch+1}/{args.epochs} - AUC: {auc:.4f}")

        if auc > best_auc + min_delta:
            best_auc = auc; wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping à l'epoch {epoch+1} (meilleur AUC: {best_auc:.4f})")
                break

    # ----------------------------- Sauvegarde artefacts -----------------------------
    ts = time.strftime("run_%Y%m%d_%H%M%S")
    if args.run_tag: ts = f"{ts}_{args.run_tag}"
    run_dir = Path(args.artifact_root) / ts
    ensure_dir(run_dir)

    mapping = (dataset._user_id_mapping, dataset._user_feature_mapping,
               dataset._item_id_mapping, dataset._item_feature_mapping)
    with open(run_dir/"mapping.pkl","wb") as f: pickle.dump(mapping, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(run_dir/"model.pkl","wb") as f: pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    save_npz(run_dir/"user_features.npz", user_features)
    save_npz(run_dir/"item_features.npz", item_features)
    save_npz(run_dir/"interactions.npz",  interactions)

    # pairs → index direct
    item_id_map = dataset._item_id_mapping
    pair_boost_idx: Dict[tuple[str,int], tuple[np.ndarray,np.ndarray]] = {}
    for (prof, A), neigh in pair_boost.items():
        idxs, vals = [], []
        for b, s in neigh.items():
            j = item_id_map.get(b)
            if j is None: continue
            idxs.append(int(j)); vals.append(float(s))
        if idxs:
            idxs = np.asarray(idxs, dtype=np.int32)
            vals = np.asarray(vals, dtype=np.float32)
            order = np.argsort(idxs)
            pair_boost_idx[(str(prof), int(A))] = (idxs[order], vals[order])

    # sauvegarde "plain" (sans defaultdict/lambda)
    with gzip.open(run_dir/"pair_boost.pkl.gz","wb") as f:
        plain = { (str(p), int(a)) : {int(b): float(v) for b, v in d.items()} for (p,a), d in pair_boost.items() }
        pickle.dump(plain, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(run_dir/"pair_boost_idx.pkl","wb") as f:
        pickle.dump(pair_boost_idx, f, protocol=pickle.HIGHEST_PROTOCOL)

    # index inverse des paires : (prof, B_id) -> (idxs_A_j, vals)
    item_idx2id = {v: k for k, v in item_id_map.items()}
    pair_rev_idx: Dict[tuple[str,int], tuple[np.ndarray,np.ndarray]] = {}
    tmp: Dict[tuple[str,int], tuple[list[int], list[float]]] = {}

    for (prof, A_id), (idxs_B, vals) in pair_boost_idx.items():
        jA = item_id_map.get(A_id)
        if jA is None: continue
        for jB, v in zip(idxs_B, vals):
            B_id = int(item_idx2id[int(jB)])
            Lidx, Lval = tmp.setdefault((prof, B_id), ([], []))
            Lidx.append(int(jA)); Lval.append(float(v))

    for key, (Lidx, Lval) in tmp.items():
        idxs = np.asarray(Lidx, dtype=np.int32)
        vals = np.asarray(Lval, dtype=np.float32)
        order = np.argsort(idxs)
        pair_rev_idx[key] = (idxs[order], vals[order])

    with open(run_dir/"pair_rev_idx.pkl", "wb") as f:
        pickle.dump(pair_rev_idx, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ----------------------------- Precompute -----------------------------
    print("== Precompute ==")
    _, emb = model.get_item_representations(features=item_features)
    emb = emb.astype(np.float32)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    np.save(run_dir/"item_emb_norm.npy", emb)

    train_csr = interactions.tocsr()
    train_pop = np.asarray(train_csr.sum(axis=0)).ravel().astype(np.float32)
    np.save(run_dir/"train_pop.npy", train_pop)

    user_id_map = dataset._user_id_mapping
    profs_sorted = sorted(user_id_map.keys(), key=lambda p: user_id_map[p])
    rows = []
    for prof in profs_sorted:
        u = user_id_map[prof]
        vec = np.asarray(train_csr.getrow(u).todense()).ravel().astype(np.float32)
        rows.append(norm01(vec))
    pop_by_prof_norm = np.vstack(rows) if rows else np.zeros((0, len(all_products)), dtype=np.float32)
    np.save(run_dir/"pop_by_prof_norm.npy", pop_by_prof_norm)
    topk = min(int(args.topk_per_prof), pop_by_prof_norm.shape[1]) if pop_by_prof_norm.size else 0
    top_popp_by_prof = np.argsort(-pop_by_prof_norm, axis=1)[:, :topk] if topk > 0 else np.zeros((pop_by_prof_norm.shape[0], 0), dtype=np.int32)
    np.save(run_dir/"top_popp_by_prof.npy", top_popp_by_prof)

    # Extras API : item_to_gamme + item_is_pl + product_meta
    print("== Export item_to_gamme & item_is_pl & product_meta ==")
    n_items = int(max(item_id_map.values())) + 1 if item_id_map else 0
    item_to_gamme = np.full(n_items, -1, dtype=np.int32)
    item_is_pl    = np.zeros(n_items, dtype=np.int8)

    idxd = df_details.set_index("ID_PRODUIT", drop=False)
    product_meta: Dict[int, Dict[str, Any]] = {}

    for pid, j in item_id_map.items():
        pid = int(pid); j = int(j)
        if pid in details_idx.index:
            item_is_pl[j] = int(details_idx.loc[pid].get("ID_MARQUE", 0))
        if pid in idxd.index:
            row = idxd.loc[pid]
            gid = int(row["ID_GAMME"]) if not pd.isna(row.get("ID_GAMME")) else -1
            uid = int(row["ID_UNIVERS"]) if not pd.isna(row.get("ID_UNIVERS")) else -1
            item_to_gamme[j] = gid
            product_meta[pid] = {
                "lib": str(row.get("LIB_PRODUIT")) if row.get("LIB_PRODUIT") is not None else None,
                "id_gamme": None if gid < 0 else gid,
                "id_univers": None if uid < 0 else uid,
                "lib_gamme": None if pd.isna(row.get("LIB_GAMME")) else str(row.get("LIB_GAMME")),
                "lib_univers": None if pd.isna(row.get("LIB_UNIVERS")) else str(row.get("LIB_UNIVERS")),
            }
        else:
            product_meta[pid] = {"lib": None, "id_gamme": None, "id_univers": None,
                                 "lib_gamme": None, "lib_univers": None}

    np.save(run_dir/"item_to_gamme.npy", item_to_gamme)
    np.save(run_dir/"item_is_pl.npy", item_is_pl)
    with gzip.open(run_dir/"product_meta.json.gz","wt", encoding="utf-8") as f:
        json.dump(product_meta, f, ensure_ascii=False)

    # versions & params
    with open(run_dir/"versions.json","w") as f:
        import lightfm as _lfm, numpy as _np, scipy as _sc
        json.dump({"python": platform.python_version(),
                   "numpy": _np.__version__, "scipy": _sc.__version__, "lightfm": _lfm.__version__}, f, indent=2)
    with open(run_dir/"params.json","w") as f:
        json.dump({
            "no_components": args.no_components, "epochs": args.epochs, "lr": args.lr, "loss": args.loss,
            "threads": args.threads, "min_occ": args.min_occ,
            "include_all_items": bool(args.include_all_items),
            "cold_identity_token": int(args.cold_identity_token),
            "topk_per_prof": args.topk_per_prof,
            "n_users": len(all_professions), "n_items": len(all_products),
            "interactions_nnz": int(interactions.nnz)
        }, f, indent=2)

    # symlink courant
    if args.make_current_symlink:
        root = Path(args.artifact_root)
        cur = root / "run_current"
        try:
            if cur.exists() or cur.is_symlink(): cur.unlink()
        except Exception:
            pass
        try:
            cur.symlink_to(run_dir.resolve())
            print(f"[symlink] {cur} -> {run_dir}")
        except Exception as e:
            print(f"[warn] symlink run_current: {e}")

    dt = tsec() - t0
    print(f"✅ Terminé. Artefacts → {run_dir} | ⏱️ {dt:.1f}s")


if __name__ == "__main__":
    main()
