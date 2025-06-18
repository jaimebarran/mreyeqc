import os
import pandas as pd
import re

def create_bids_csv_with_ratings(bids_csv_path, ratings_csv_path):
    """
    Fusionne les ratings de rating.csv dans bids_csv.csv et sauvegarde le résultat
    dans un nouveau fichier nommé bids_csv_rating.csv.

    Args:
        bids_csv_path (str): Chemin vers le fichier bids_csv.csv.
        ratings_csv_path (str): Chemin vers le fichier rating.csv.
    """
    print("--- Démarrage de l'ajout des ratings ---")
    
    # Définir le chemin du fichier de sortie
    output_dir = os.path.dirname(bids_csv_path)
    output_path = os.path.join(output_dir, 'bids_csv_rating.csv')

    print(f"Fichier BIDS source : {bids_csv_path}")
    print(f"Fichier de ratings : {ratings_csv_path}")
    print(f"Fichier de sortie : {output_path}")
    print("-" * 50)

    try:
        # Charger les fichiers en s'assurant que les IDs sont lus comme du texte
        df_bids = pd.read_csv(bids_csv_path, dtype=str)
        df_ratings = pd.read_csv(ratings_csv_path, dtype=str)
        print("Fichiers CSV chargés avec succès.")
    except FileNotFoundError as e:
        print(f"❌ ERREUR: Fichier non trouvé. Veuillez vérifier les chemins. {e}")
        return
    except Exception as e:
        print(f"❌ ERREUR: Problème lors de la lecture d'un fichier CSV : {e}")
        return

    # --- Préparation pour une fusion robuste ---

    def normalize_subject_id(sub_id):
        """
        Normalise un ID de sujet (ex: 'sub-001', '1', '001') en un format standard ('001').
        """
        if pd.isna(sub_id):
            return None
        # Supprime tous les caractères non numériques
        cleaned_id = re.sub(r'\D', '', str(sub_id))
        if cleaned_id:
            # Met en forme sur 3 chiffres avec des zéros devant
            return cleaned_id.zfill(3)
        return None

    # Appliquer la normalisation sur les deux DataFrames
    df_bids['sub_norm'] = df_bids['sub'].apply(normalize_subject_id)
    df_ratings['sub_norm'] = df_ratings['sub'].apply(normalize_subject_id)
    print("Identifiants des sujets normalisés pour la correspondance.")

    # Gérer les doublons dans le fichier de ratings pour éviter les fusions incorrectes
    if df_ratings['sub_norm'].duplicated().any():
        num_duplicates = df_ratings['sub_norm'].duplicated().sum()
        print(f"⚠️ AVERTISSEMENT: {num_duplicates} sujet(s) en double trouvé(s) dans le fichier de ratings.")
        print("Suppression des doublons, seule la première note pour chaque sujet sera conservée.")
        df_ratings = df_ratings.drop_duplicates(subset=['sub_norm'], keep='first')

    # --- Fusion des deux fichiers ---
    print("Fusion des données...")
    # Fusion à gauche pour conserver tous les sujets de bids_csv.csv
    df_final = pd.merge(
        df_bids,
        df_ratings[['sub_norm', 'average_rating']],
        on='sub_norm',
        how='left'
    )

    # Renommer la colonne et supprimer la colonne de normalisation temporaire
    df_final.rename(columns={'average_rating': 'rating'}, inplace=True)
    df_final = df_final.drop(columns=['sub_norm'])
    
    ratings_added = df_final['rating'].notna().sum()
    print(f"{ratings_added} ratings ont été ajoutés au fichier.")

    # --- Sauvegarde du nouveau fichier ---
    try:
        df_final.to_csv(output_path, index=False)
        print(f"\n✅ Le nouveau fichier a été sauvegardé avec succès ici : {output_path}")
    except Exception as e:
        print(f"❌ ERREUR: Impossible de sauvegarder le fichier de sortie : {e}")

    print("--- Script terminé ---")

# --- Définir les chemins de vos fichiers ---
# Fichier BIDS principal
bids_csv_file = '/Users/cyriltelley/Desktop/MSE/Second_semester/PA-MReye/Codes/MREyeQC_PA/data/bids_csv/bids_csv.csv'
# Fichier contenant les ratings
ratings_file = '/Users/cyriltelley/Desktop/MSE/Second_semester/PA-MReye/Codes/MREyeQC_PA/data/bids_csv/rating.csv'

# --- Exécuter la fonction ---
create_bids_csv_with_ratings(bids_csv_file, ratings_file)