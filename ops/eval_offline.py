#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline evaluation des reco avec la même logique que l'API.
- Charge la classe RecommenderService et les artefacts.
- Échantillonne des commandes de df_ventes_grouped.parquet.
- Tente de prédire le(s) produit(s) manquant(s) d'une commande.
- Rapporte Hit@K, Recall@K, MRR@K (global + par métier).
"""

import os, random, argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import re

# Importe la logique EXACTE de reco
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "api"))
from reco_service import RecommenderService

def _as_int_list(x):
    if isinstance(x, (list, tuple, np.ndarray)): return [int(v) for v in x]
    if isinstance(x, str):
        x = x.strip()
        if x.startswith("[") and x.endswith("]"):
            try:
                import ast
                return [int(v) for v in ast.literal_eval(x)]
            except Exception:
                pass
        if "," in x:
            return [int(s.strip()) for s in x.split(",") if s.strip()]
        try: return [int(x)]
        except: return []
    try: return [int(x)]
    except: return []

RE_RUN = re.compile(r"^run_\d{8}_\d{6}.*")  # ex: run_20250924_122145[_tag]

def _is_run_dir(p: Path) -> bool:
    return p.is_dir() and (p / "mapping.pkl").exists()

def _latest_run_dir(root: Path) -> Path | None:
    if not root.exists() or not root.is_dir():
        return None
    cands = [d for d in root.iterdir() if d.is_dir() and RE_RUN.match(d.name)]
    if not cands:
        return None
    cands.sort(key=lambda d: d.name, reverse=True)
    return cands[0]

def _resolve_artifact_dir(art_dir_env: str) -> Path:
    p = Path(art_dir_env)
    # suit le symlink si besoin
    if p.is_symlink():
        p = p.resolve()
    # déjà un run ?
    if _is_run_dir(p):
        return p
    # racine artifacts -> dernier run
    if p.is_dir():
        # si mapping.pkl est directement là, on considère que c'est un run
        if (p / "mapping.pkl").exists():
            return p
        last = _latest_run_dir(p)
        if last:
            return last
    raise FileNotFoundError(f"Aucun run valide trouvé à partir de: {art_dir_env}")

def mrr_at_k(recs, targets):
    # plus petite position (1-based) d’un target dans recs ; 0 si absent
    pos = min((i+1 for i, r in enumerate(recs) if r in targets), default=0)
    return 1.0/pos if pos > 0 else 0.0

def evaluate(
    ventes_parquet: str,
    artifact_dir: str,
    n_samples: int = 5000,
    basket_m: int = 1,
    k: int = 5,
    seed: int = 42,
    profession_mode: str = "auto",  # "auto" (détection métier), "use_order" (métier réel), "none" (global)
):
    rng = np.random.default_rng(seed)

    # Charge artefacts
    resolved = _resolve_artifact_dir(artifact_dir)
    svc = RecommenderService(str(resolved), basket_params_by_size={})


    # Charge commandes
    df = pd.read_parquet(ventes_parquet)
    df["PROFESSION_CLIENT"] = df["PROFESSION_CLIENT"].astype(str)
    df["LIST_ID_PRODUITS"] = df["LIST_ID_PRODUITS"].apply(_as_int_list)

    # Filtre commandes éligibles (>= basket_m + 1 produits connus du modèle)
    rows = []
    for r in df.itertuples(index=False):
        prods = [p for p in getattr(r, "LIST_ID_PRODUITS") if p in svc.item_id_map]
        if len(prods) >= basket_m + 1:
            rows.append((str(getattr(r, "PROFESSION_CLIENT")), prods))
    if not rows:
        raise RuntimeError("Aucune commande exploitable (trop courte ou items inconnus).")

    # Échantillonnage
    if n_samples > 0 and n_samples < len(rows):
        rows = [rows[i] for i in rng.choice(len(rows), size=n_samples, replace=False)]
    else:
        rng.shuffle(rows)

    # Accumulateurs
    tot = 0
    hit_sum, recall_sum, mrr_sum = 0.0, 0.0, 0.0
    per_prof = {}

    for prof, prods in rows:
        # tire un sous-panier basket_m et targets = reste
        prods = list(map(int, prods))
        basket_idx = rng.choice(len(prods), size=basket_m, replace=False)
        basket = [prods[i] for i in basket_idx]
        targets = [p for p in prods if p not in basket]
        if len(targets) == 0:
            continue

        # détermine “profession” selon mode
        if profession_mode == "use_order":
            profession = prof
        elif profession_mode == "none":
            profession = None
        else:  # "auto" = mix auto dans RecommenderService
            profession = None

        # reco
        try:
            recs = svc.recommend_panier(profession, basket, k=k)
        except Exception as e:
            # skip si métier inconnu etc.
            continue

        # métriques
        hit = 1.0 if any(r in targets for r in recs) else 0.0
        recall = len([r for r in recs if r in targets]) / min(k, len(targets))
        mrr = mrr_at_k(recs, set(targets))

        hit_sum += hit; recall_sum += recall; mrr_sum += mrr; tot += 1

        # par métier (métier réel de la commande, pour lecture business)
        d = per_prof.setdefault(prof, {"n":0, "hit":0.0, "recall":0.0, "mrr":0.0})
        d["n"] += 1; d["hit"] += hit; d["recall"] += recall; d["mrr"] += mrr

    if tot == 0:
        raise RuntimeError("Aucun cas évalué.")

    report = {
        "artifact_dir": artifact_dir,
        "n_used_orders": tot,
        "basket_m": basket_m,
        "k": k,
        "mode": profession_mode,
        "metrics": {
            "hit@k": hit_sum / tot,
            "recall@k": recall_sum / tot,
            "mrr@k": mrr_sum / tot,
        }
    }
    return report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ventes", default='/home/lfaye/projects/reco/training/data/df_ventes_grouped.parquet', help="df_ventes_grouped.parquet")
    ap.add_argument("--artifact-dir", default='/home/lfaye/projects/reco/artifacts', help="Dossier de run (mapping.pkl présent)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--basket-m", type=int, default=1)
    ap.add_argument("--max-basket-m", type=int, default=6)
    ap.add_argument("--n-samples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["auto", "use_order", "none"], default="auto",
                    help="auto=mix métier par embeddings ; use_order=utilise le métier de la ligne ; none=global sans métier")
    ap.add_argument("--out", default="", help="Chemin JSON de sortie (optionnel)")
    args = ap.parse_args()

    resolved = _resolve_artifact_dir(args.artifact_dir)
    m = args.basket_m
    while m < args.max_basket_m:
        rep = evaluate(args.ventes, str(resolved), n_samples=args.n_samples,
               basket_m=m, k=args.k, seed=args.seed, profession_mode=args.mode)
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(rep, f, indent=2, ensure_ascii=False)

        m+=1        

if __name__ == "__main__":
    main()
