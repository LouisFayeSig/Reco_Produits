import pandas as pd
import pyodbc
import numpy as np
import argparse
from pathlib import Path
from itertools import combinations
import os

def connect():
    # Essaie dans cet ordre (Windows → Linux/macOS)
    prefer = [
        os.getenv("MSSQL_DRIVER"),                 # facultatif pour forcer
        "ODBC Driver 18 for SQL Server",
        "ODBC Driver 17 for SQL Server",
        "SQL Server",                              # Windows
    ]
    available = set(d.strip() for d in pyodbc.drivers())
    driver = next((d for d in prefer if d and d in available), None)
    if not driver:
        raise RuntimeError(
            "Aucun driver MSSQL trouvé. Installez msodbcsql18/17 (Linux/macOS) "
            "ou le driver ODBC 18 (Windows)."
        )

    server = os.environ["MSSQL_SERVER"]               # ex: host,1433 ou host\\instance
    db = os.getenv("MSSQL_DATABASE", "")
    trusted = os.getenv("MSSQL_TRUSTED_CONNECTION", "").lower() in {"1","true","yes"}

    parts = [f"DRIVER={{{driver}}}", f"SERVER={server}"]
    if db: parts.append(f"DATABASE={db}")

    if trusted:
        parts.append("Trusted_Connection=yes")
    else:
        parts += [f"UID={os.environ['MSSQL_USER']}", f"PWD={os.environ['MSSQL_PASSWORD']}"]

    # Petite règle simple: Driver 18 → Encrypt=yes par défaut
    if "ODBC Driver 18" in driver:
        parts.append(f"Encrypt={os.getenv('MSSQL_ENCRYPT','yes')}")
        tsc = os.getenv("MSSQL_TRUST_SERVER_CERT")
        if tsc: parts.append(f"TrustServerCertificate={tsc}")

    return pyodbc.connect(";".join(parts))

def main(): 
    ap = argparse.ArgumentParser()
    ap.add_argument("--nom_serveur", help="Nom du serveur datawarehouse")
    ap.add_argument("--dir_data", default="training/data/", help="Dossier de stockage de la data")

    args = ap.parse_args()
    dir_data = Path(args.dir_data)
    dir_data.mkdir(parents=True, exist_ok=True)
    ventes_pq  = dir_data / 'df_ventes_grouped.parquet'
    details_pq = dir_data / 'df_details_produits.parquet'
    pairs_pq   = dir_data / 'df_couple_porduit.parquet'


    # Connection à la base de données recette

    conn = connect()
    print('Connection au data warehouse')

    # ----- 1. Détails produits ------
    # Récupération de la gamme et de l'univers en fonction de l'id Gamme et Univers
    query_univers = "SELECT [ID_UNIVERS], [LIB_UNIVERS] FROM DWH.REF_UNIVERS_PRODUIT WHERE ID_ENSEIGNE = 1"
    df_univers = pd.read_sql(query_univers, conn)

    query_univers = "SELECT [ID_GAMME], [LIB_GAMME] FROM DWH.REF_GAMME_PRODUIT WHERE ID_ENSEIGNE = 1"
    df_gamme= pd.read_sql(query_univers, conn)

    # Récupération du détails produits et rajout détails gamme et univers
    query_produit = "SELECT [ID_PRODUIT], [LIB_PRODUIT], [ID_UNIVERS], [ID_GAMME] FROM DWH.D_PRODUIT WHERE ID_ENSEIGNE = 1"
    df_produit = pd.read_sql(query_produit, conn)
    df_details_produits = pd.merge(df_produit, df_gamme, on="ID_GAMME", how="left")
    df_details_produits = pd.merge(df_details_produits, df_univers, on="ID_UNIVERS", how="left")

    # Mise en valeur des marques propres depuis le nom du produit
    marques = {
        'ALYE': 'Marque_Alye',
        'LITT': 'Marque_Litt'
    } # Dictionnaire des correspondances

    # Création des colonnes si elles n'existent pas
    df_details_produits['LIB_MARQUE'] = np.nan
    df_details_produits['ID_MARQUE'] = 0

    # Parcours des mots-clés et mise à jour
    for mot, marque in marques.items():
        mask = df_details_produits['LIB_PRODUIT'].str.contains(mot, case=False, na=False)
        df_details_produits.loc[mask, 'LIB_MARQUE'] = marque
        df_details_produits.loc[mask, 'ID_MARQUE'] = 1 # 1 pour marques propres, 0 pour marques extérieurs

    # Sauvegarde en parquet du df
    df_details_produits.to_parquet(details_pq)
    print('Création et sauvegarde du dataframe des details produits')


    # ----- 2. Ventes clients ------
    # Récupération de la profession client en fonction de son ID_client
    query_client = "SELECT [ID_CLIENT], [PROFESSION_CLIENT] FROM DWH.D_CLIENT WHERE ID_ENSEIGNE = 1"
    df_client = pd.read_sql(query_client, conn)
    
    # Récupération des commandes clients et rajout de la profession
    query_ventes = "SELECT [ID_CLIENT], [ID_PRODUIT], [NUMERO_COMMANDE_CLIENT] FROM DWH.F_COMMANDES_CLIENTS WHERE ID_ENSEIGNE = 1"
    df_ventes = pd.read_sql(query_ventes, conn)
    df_ventes = pd.merge(df_ventes, df_client, on="ID_CLIENT", how="left")

    # Epuration des commandes si ID_PRODUIT mal renseigné
    df_ventes = df_ventes[df_ventes['ID_PRODUIT'] != 0]

    # Rajout de ID_GAMME et ID_UNIVERS pour chaque ID_PRODUIT
    df_ventes = pd.merge(df_ventes, df_details_produits, on="ID_PRODUIT", how="left")

    # Groupage des commandes par leur NUMERO_COMMANDE_CLIENT
    df_ventes_grouped = df_ventes.groupby('NUMERO_COMMANDE_CLIENT').agg({
        'ID_PRODUIT': list,
        'ID_CLIENT': 'first',
        'PROFESSION_CLIENT': 'first',
        'ID_UNIVERS': list,
        'ID_GAMME': list,
        'NUMERO_COMMANDE_CLIENT': 'count' 
    }).rename(columns={'NUMERO_COMMANDE_CLIENT': 'NB_OCCURENCES_COMMANDE',
                    'ID_PRODUIT': 'LIST_ID_PRODUITS',
                    'ID_UNIVERS': 'LIST_ID_UNIVERS',
                    'ID_GAMME': 'LIST_ID_GAMME'
                    }).reset_index()
    
    # Sauvegarde en parquet du df
    df_ventes_grouped.to_parquet(ventes_pq)
    print('Création et sauvegarde du dataframe des achats clients')


    # ----- 3. Couples de produits achetés ensemble ------
    
    df = df_ventes_grouped[df_ventes_grouped['LIST_ID_PRODUITS'].str.len() > 1]

    # Explode pour obtenir toutes les combinaisons
    df_exploded = df.explode(['LIST_ID_PRODUITS', 'LIST_ID_UNIVERS', 'LIST_ID_GAMME'])
    df_exploded = df_exploded.rename(columns={
        'LIST_ID_PRODUITS': 'ID_PRODUIT',
        'LIST_ID_UNIVERS': 'ID_UNIVERS',
        'LIST_ID_GAMME': 'ID_GAMME'
    })

    # Merge sur lui-même pour créer les couples (ID_PRODUIT_A, ID_PRODUIT_B)
    df_pairs = (
        df_exploded.merge(df_exploded, on='NUMERO_COMMANDE_CLIENT')
        .query('ID_PRODUIT_x < ID_PRODUIT_y')  # éviter doublons et self-pairs
    )

    df_pairs = (
        df_pairs.groupby(['ID_PRODUIT_x', 'ID_PRODUIT_y', 'ID_UNIVERS_x', 'ID_UNIVERS_y', 'ID_GAMME_x', 'ID_GAMME_y'])
        .agg({
            'NB_OCCURENCES_COMMANDE': 'sum',
            'PROFESSION_CLIENT_x': lambda x: list(set(x))
        })
        .reset_index()
        .rename(columns={
            'ID_PRODUIT_x': 'ID_PRODUIT_A',
            'ID_PRODUIT_y': 'ID_PRODUIT_B',
            'ID_UNIVERS_x': 'LIST_ID_UNIVERS_A',
            'ID_UNIVERS_y': 'LIST_ID_UNIVERS_B',
            'ID_GAMME_x': 'LIST_ID_GAMME_A',
            'ID_GAMME_y': 'LIST_ID_GAMME_B',
            'NB_OCCURENCES_COMMANDE': 'OCCURENCE_PAIR',
            'PROFESSION_CLIENT_x': 'PROFESSION_CLIENT'
        })
    )

    # Sauvegarde du dataframe de couples produits
    df_pairs.to_parquet(pairs_pq)
    print('Creation et sauvegarde du dataframe de couples de produits')

if __name__ == "__main__":
    main()