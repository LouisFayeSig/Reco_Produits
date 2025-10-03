# Reco Matériaux — Training & API

Système de recommandation **« panier → compléments »** pour le négoce BTP, basé sur **LightFM** + **heuristiques de paires** (directionnelles et inverses), avec :
- Professions (métier) explicites **ou** auto‑détection à partir du panier
- Paires directionnelles A→B et **paires inverses** ?→B pour retrouver les « prédécesseurs »
- **Diversification** (MMR) et **cap par gamme**
- **Boost Marque Propre / MDD** (poids additifs & multiplicatifs)
- **Profils de paramètres par taille de panier** (avec `default` et buckets `"N+"`)
- Résolution d’artefacts : choix **automatique** du dernier `run_*` ou `symlink` `run_current`
- Enrichissement libellés/infos produit pour debug & UX

---

## Sommaire
1. Structure du repo  
2. Installation des environnements  
3. Jeux de données attendus  
4. Entraînement (training)
5. Artefacts et son contenu  
6. Api  
7. Profils de paramètres panier
8. Détails des poids de ranking  
9. Evaluation offline  
10. Logique de ranking: comment-ça-marche  
11. Makefile & tâches utiles  
12. Dépannage & faq  
13. Conseils d'usage & réglages  

---

## 1. Structure du repo 

├─ api/
│  ├─ app.py                 # FastAPI (découvre le run le + récent)
│  ├─ reco_service.py        # Service de reco (MMR, pairs, reverse pairs, MDD…)
│  ├─ panier_params.json     # Paramètres des profiles de recherche 
│  ├─ requirements-api.txt   # Paramètres du .venv-api
│  └─ .venv-api/             # venv API
│
├─ ops/
│  ├─ save_precomputed.py    # (optionnel) recalc embeddings/popularités pour un run
│  ├─ creation_dataset.py    # Création des datasets avec appel warehouse
│  ├─ clean_up.sh            # Automatisation du clean des artifacts (optionnel)
│  ├─ promote_and_reaload.sh # Automatisation du loading d'artifact (optionnel)
│  ├─ rollback.sh            # Automatisation du rollback du précédent artifact (optionnel)
│  └─ eval_offline.py        # évaluation offline (Hit@K, MRR…)
│
├─ training/
│  ├─ .venv-training         # venv Training
│  ├─ train_daily.py         # Entraînement + export complet des artefacts
│  ├─ requirements-train.txt # Paramètres du .venv-training
│  ├─ apt.txt                # Dépendances pour le curl de la data
│  └─ data/ (localisation optionnelle des dataset)
│      │
│      ├─ df_couple_produit.parquet 
│      ├─ df_ventes_grouped.parquet 
│      └─ df_details_produits.parquet 
│
├─ artifacts/
│  ├─ run_YYYYMMDD_HHMMSS[_tag]/  # sorties d’un training
│  └─ run_current -> run_...      # symlink vers le dernier run "actif"
│
├─ .env                      # Variable environnement locale, exemple en pull
├─ Makefile                  # Action automatique en une ligne
└─ README.md  ← (vous êtes ici)

---

## 2. Installation & environnements

### a. Création env training première utilisation
chmod +x training/setup.shchmod +x training/setup.sh
./training/setup.sh

### b. Création env api
chmod +x api/setup.shchmod +x api/setup.sh
./api/setup.sh

**Note** : si vous avez eu des soucis de compatibilité NumPy/Pandas/Scipy, nous avons pinné des versions compatibles dans les requirements.

---

## 3. Jeux de données attendus 

- Création possible via la commande:
python creation_dataset.py \
  --flag1
  --flag2

- ou make data --flag1 --flag2 

### Flags de création:
  --nom_serveur: Nom du serveur datawarehouse, à demander à l'administrateur
  --dir_data : Dossier de stockage de la data, par défaut dans l'architecture training/data/

### Constitution de la data  
- **df_ventes_grouped.parquet**, colonnes:
  - PROFESSION_CLIENT (str)
  - NB_OCCURENCES_COMMANDE (int, par défaut 1)
  - LIST_ID_PRODUITS (list[int])

- **df_details_produits.parquet**, colonnes:
  - ID_PRODUIT (int)
  - ID_GAMME (int, optionnel)
  - ID_UNIVERS (int, optionnel)
  - LIB_PRODUIT, LIB_GAMME, LIB_UNIVERS (optionnel)
  - ID_MARQUE (0/1) → 1 = marque propre (MDD) → sert au boost

- **df_couple_produit.parquet** (facultatif mais recommandé), colonnes:
  - ID_PRODUIT_A, ID_PRODUIT_B (int)
  - OCCURENCE_PAIR (int)
  - PROFESSION_CLIENT (str | list[str])

---

## 4. Entraînenement (Training)

### Commande rapide avec **make**:
make train \
  --flag1
  --flag2
  ...

En fin de run, make appelle verify_artifacts pour s’assurer que tous les artefacts attendus sont présents.

### Alternative:
python training/train_daily.py \
  --flag1
  --flag2
  ... 

### a. Flags de base:
--ventes : localisation du db des ventes
--details : localisation du db des détails sur les produits
--pairs : localisation du db des produits par pairs
--artifact-root : défini par défaut le nom de dossiers des artifacts
--run-tag : permet de changer le nom de l'enregistrement de l'artifact, sinon date & heur par défaut

### b. Flags du model lightFM:
--no-components : 64, 128, 256 plus c'est haut, plus demande du temps d'apprentissage mais plus profond
--epochs : nombre d'epoque d'apprentissage mais capé par la patience
--lr : learning rate, de base 0.05 mais peu diminuer pour avoir un apprentissage plus profond
--loss : la métrique qui permet d'avoir un ranking, par défaut sur warp
--threads : parallélisation de la tâche, par défaut sur 8, attention au capacité de de la machine
--patience : nb d'epochs sans amélioration avant arrêt

### c. Flags de construction & de filtrage (les plus utiles):
--min-occ : min occurrences produits dans les ventes
--include-all-items : inclut tout le catalogue (sinon, seuls les items avec ventes ≥ --min-occ)
--cold-identity-token 0 : pas de feature prod:PID pour les items sans ventes (réduit fortement le nb de features)
--symmetrize-pairs : ajoute B→A en plus de A→B, et construit les paires inverses
--max-items N : cap sur nb d’items (0 = pas de cap) — utile si >300k produits
--topk-per-prof : nb max d’items populaires « métiers » exportés (défaut 3000)
--make-current-symlink : met à jour artifacts/run_current -> artifacts/run_...

---

## 5. Artefacts & son contenu

Dans chaque artifacts/run_* :
- **Modèle & mappings**
  - mapping.pkl : user_id_map, user_feature_map, item_id_map, item_feature_map
  - user_features.npz, item_features.npz, interactions.npz
  - model.pkl (LightFM, utilisé uniquement si recalcul embeddings)

- **Pré‑calculs**
  - item_emb_norm.npy : embeddings items L2‑normalisés
  - train_pop.npy : popularité globale train
  - pop_by_prof_norm.npy : popularité normalisée par métier
  - top_popp_by_prof.npy : indices des items les plus populaires par métier
  - item_to_gamme.npy : mapping item → gamme
  - item_is_pl.npy : 0/1 marque propre (MDD)

- **Paires**
  - pair_boost.pkl.gz : paires brutes (fallback si pas d’index)
  - pair_boost_idx.pkl : paires indexées (alignées aux indices items)
  - pair_rev_idx.pkl : index inverse des paires (pour scorer les « prédécesseurs »)

- **Enrichissement & infos**
  - product_meta.json.gz : libellés & méta (univers/gamme)
  - versions.json, params.json

---

## 6. APi

### Démarrage rapide en local:
make test

-> ouvre par défaut le port 8080 sauf si précision de PORT=XXXX
-> accessible dans un navigateur ici avec les détails: http://localhost:8080/docs ou  http://localhost:8080/"Endpoint"

La variable ARTIFACT_DIR est positionnée vers la racine artifacts et l’API choisit automatiquement:
  - un run précis s’il est donné,
  - /sinon le symlink run_current s’il existe,
  - /sinon le dernier run_* par ordre lexicographique (timestamp dans le nom).

### Endpoints:
- GET /healthz → {"ok": true}
- GET /version → métadonnées du service (comptes, poids actifs, profil panier actif, artefact choisi)

- POST /recommend_panier
{
  "profession": "PLAQUISTE",  // ou null/""  → auto-détection métier
  "product_ids": [274426, 373353],
  "k": 10,
  "expand": true  // ajoute libellés/infos produits
}

- POST /products/enrich → enrichit une liste d’IDs avec product_meta (si dispo)

- POST /reload (protégé via X-Admin-Key si ADMIN_KEY défini)
{
  "artifact_dir": "artifacts" | "artifacts/run_current" | "artifacts/run_YYYYMMDD_HHMMSS_tag",
  "profile": "precision_pair_heavy_v2"  // optionnel, voir profils ci-dessous
}

---

## 7. Profils de panier

Les poids / tailles de pools / diversité / boost MDD peuvent être réglés via un JSON optionnel (ex: api/panier_params.json).
Si le panier est supérieur à 5, on prend le dernier profil
La profession est optionnelle : si absente, le service détecte automatiquement un mix de métiers (softmax top-K sur la similarité aux centroïdes métiers).
Les poids env (W_PAIR…) existent toujours, mais les profils priment.

Exemple de profil:
"precision_pair_heavy_v2": {
      "basket_params_by_size": {
        "1": { "W_PAIR": 0.75, "W_EMB": 0.20, "W_POPP": 0.15, "W_POPG": 0.0,
               "PAIR_ANY_WEIGHT": 0.40,
               "TOP_PAIRS": 3500, "TOP_POPP": 1000, "TOP_POPG": 100, "TOP_PAIRS_ANY_FRAC": 0.25,
               "PAIR_QUOTA": 2,
               "DIVERSIFY": 1, "MMR_LAMBDA": 0.75, "CAP_PER_GAMME": 2,
               "W_PL_BASE": 0.12, "W_PL_PER_BASK_PL": 0.18 }}}

2 profils existants: **precision_pair_heavy_v2** & **diversity_friendly_v2**

---

## 8. Détails des poids de ranking

### 🧮 Pondération des scores
Ces coefficients déterminent l’importance relative des différentes sources d’information pour scorer les candidats (avec des exemples):

- **`W_PAIR = 0.60`** : Favorise les liens de co-achat (A→B).  
  ↗ Précision sur les compléments. ↘ Moins de place pour les embeddings et la popularité.

- **`W_EMB = 0.30`** : Similarité d’embeddings (LightFM).  
  ↗ Proximité sémantique (usage/gamme). ↘ Moins de diversité métier.

- **`W_POPP = 0.20`** : Popularité par métier.  
  ↗ Renforce les best-sellers métier. ↘ Moins de neutralité.

- **`W_POPG = 0.00`** : Popularité globale.  
  ↗ Tire vers les hyper-vendus. 0 = désactivé pour limiter les biais.

### 🧺 Construction du pool de candidats
Définit quels articles peuvent être évalués :

- **`TOP_PAIRS = 2400`** : Voisins de paires par item du panier.  
  ↗ Plus de compléments potentiels. ↘ Plus de bruit.

- **`TOP_POPP = 1600`** : Populaires métier ajoutés au pool.  
  ↗ Utile si panier peu informatif.

- **`TOP_POPG = 300`** : Populaires globaux ajoutés au pool.  
  ↗ Ouvre aux best-sellers transverses.

- **`TOP_PAIRS_ANY_FRAC = 0.30`** : Part de voisins fallback “ANY”.  
  ↗ Couverture en cas de manque PRO. ↘ Plus strict PRO.

- **`PAIR_QUOTA = 2`** : Minimum de recommandations issues de paires.  
  ↗ Assure la logique de panier.

### 🎨 Diversité et contrôle assortiment
Définit une diversité dans la proposition des articles:

- **`DIVERSIFY = 1`** : Active MMR pour éviter les doublons.

- **`MMR_LAMBDA = 0.7`** : 70% pertinence, 30% diversité.

- **`CAP_PER_GAMME = 2`** : Max 2 articles par gamme dans le top.  
  ↗ Améliore la variété perçue.

### 🏷️ Marque propre (MDD)
Mettre en avant les marques propres de l'entreprise:

- **`W_PL_BASE = 0.10`** : Bonus constant pour tout item MDD.
- **`W_PL_PER_BASK_PL = 0.15`** : Bonus proportionnel à la part de MDD dans le panier.  
  ↗ Renforce la cohérence MDD.

--- 

## 9.Evaluation offline
Pour tester à quel point le modèle est performant pour suggérer les produits adéquats

### Démarrage en local d'une séance d'évaluation:
. .venv-train/bin/activate
python ops/eval_offline.py \
  --flag1
  --flag2
  ...

### Flags:
--ventes : db des ventes groupés
--artifact-dir : dossier des artifacts
--k : le top K de recommendation, par défaut 10
--basket-m : la taille M du panier +1, par défaut m=1, donc panier=2 
--max-basket-m : la taille Max du panier -1 pour tester différentes taille de panier, allant de M à Max
--n-samples : le nombre d'échantillon à tester, par défaut 10000
--seed : le pickage des échnatillons est aléatoire ou seeder, par défaut 42
--out : sortie des résultats optionnels dans un chemin de JSON, sinon affichage console
--mode : choix=["auto", "use_order", "none"], par défaut auto, 
  auto=mix métier par embeddings ; use_order=utilise le métier de la ligne ; none=global sans métier

### Métriques d'évaluation:
- **Hit@K** : est-ce qu’au moins un des produits “attendus” (retenus de la commande) est dans le top-K ?
- **Recall@K** : proportion des produits attendus couverts par le top-K.
- **MRR@K** : quelle est la position du premier bon item (plus c’est haut, mieux c’est).

---

## 10. Logique de ranking – comment ça marche

### a. Construction du pool candidats (rapide, NumPy)
- Paires A→B spécifiques au métier (et mix de métiers si auto)
- Paires “ANY” (fallback global) – fraction configurable
- Paires inverses (reverse) pour retrouver des “prédécesseurs” plausibles
- Populaires métier, puis populaires globaux
- Retrait du panier courant

### b. Scoring (linéaire)
- W_PAIR × (paires pro + PAIR_ANY_WEIGHT × paires ANY)
- W_EMB × (similarité au centroïde du panier)
- W_POPP × popularité métier mixée (si pro connue/auto)
- W_POPG × popularité globale
- Bonus couverture (liens multiples avec les items du panier)
- Boost MDD : additif W_PL_BASE + multiplicatif W_PL_PER_BASK_PL sur les candidats dont item_is_pl=1

### c. Diversification
- Cap par gamme (CAP_PER_GAMME)
- MMR (DIVERSIFY=1, MMR_LAMBDA)

### d. Quotas
- PAIR_QUOTA : nombre mini d’items issus de paires (si activé)

---

## 11. Makefile – tâches utiles

Dans le shell:
- make help, pour avoir les différentes options du make
- make data, pour créer la data
- make train, pour lancer le training du dataset
- make verify_artifacts, vérifie la présence de:
mapping.pkl, (user|item)_features.npz, interactions.npz, item_emb_norm.npy, train_pop.npy, pop_by_prof_norm.npy, top_popp_by_prof.npy, item_to_gamme.npy, item_is_pl.npy, product_meta.json.gz, versions.json, params.json, et pair_boost_idx.pkl (ou pair_boost.pkl.gz), pair_rev_idx.pkl.
- make test, pour lancer l'api en local sur le PORT=8080 par défaut → http://localhost:8080/docs
- make stop, pour stopper l'api en local sur le PORT=8080 par défaut

---

## 12. Dépannage & FAQ

### Aucun run valide trouvé au démarrage API
→ Assurez-vous d’avoir au moins un artifacts/run_* complet.
→ Utilisez --make-current-symlink pour créer run_current, ou positionnez ARTIFACT_DIR vers un run précis.

### item_emb_norm absent et aucun modèle dispo
→ Soit item_emb_norm.npy manque, soit model.pkl est absent (ou DISABLE_MODEL_PICKLE=1).
→ Refaites un training complet ou exécutez ops/save_precomputed.py sur un run.

### ABI NumPy/Pandas/Scipy (dtype size changed…)
→ (Re)créez la venv et installez les versions pinnées des requirements.

### LightFM wheel / build
→ Les requirements-train pinnenet une version compatible. Si vous buildiez localement, assurez-vous d’avoir un toolchain C correct.

### Évals impossible : mapping.pkl manquant via run_current
→ Créez/renouvelez le symlink run_current (ou passez un run_* explicite en --artifact-dir).

---

## 13. Conseils d’usage & réglages

- **Inclure tout le catalogue** : 
  --include-all-items pour éviter que le filtre popularité limite trop (sinon on passe parfois de 300k → 2k).
  Si nécessaire, utilisez --max-items pour limiter la taille en dev.

- **Mettre en avant les MDD** :
  Exportez item_is_pl.npy (fait par train_daily.py)
  Ajustez W_PL_ABS, PL_MULT, et éventuellement W_PL_PER_BASK_PL, W_PL_BASE dans les profils.

- **Profils faciles à maintenir** : 
  utilisez "default", "N+" et quelques clés précises ("1", "2"…) plutôt que dupliquer les mêmes blocs.

- **Panier sans profession** : 
  l’API auto-détecte un mix de métiers via les centroïdes d’items.
  Réglages : PROFMIX_TOPK, PROFMIX_TEMP, PROFMIX_MINMASS.