import os
import pandas as pd

def calculate_and_combine_ratings(
    base_ratings_dir,
    additional_ratings_path,
    output_csv_path
):
    """
    Lit les fichiers rating.csv, calcule la moyenne des notes par sujet,
    ajoute des notes pré-calculées depuis un fichier additionnel pour les sujets manquants,
    et sauvegarde le résultat dans un nouveau CSV.

    Args:
        base_ratings_dir (str): Le dossier de base contenant les dossiers des évaluateurs.
        additional_ratings_path (str): Chemin vers le fichier CSV contenant les notes déjà moyennées.
        output_csv_path (str): Chemin pour sauvegarder le CSV final.
    """
    all_ratings_df = pd.DataFrame()
    rater_folders = ['bene', 'jaime', 'meri']

    print(f"Lancement du traitement des fichiers de notation depuis : {base_ratings_dir}")
    print("-" * 50)

    # --- Étape 1 : Calculer la moyenne des notes des 3 évaluateurs ---
    for rater_folder in rater_folders:
        rating_file_path = os.path.join(base_ratings_dir, rater_folder, 'ratings.csv')

        if os.path.exists(rating_file_path):
            print(f"Fichier de notation trouvé : {rating_file_path}")
            try:
                df = pd.read_csv(rating_file_path, dtype={'sub': str})
                if 'sub' in df.columns and 'rating' in df.columns:
                    # Convertir la colonne 'rating' en numérique, les erreurs deviendront NaN
                    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
                    # Supprimer les lignes où le rating n'a pas pu être converti
                    df.dropna(subset=['rating'], inplace=True)
                    all_ratings_df = pd.concat([all_ratings_df, df[['sub', 'rating']]], ignore_index=True)
                else:
                    print(f"AVERTISSEMENT: '{rating_file_path}' n'a pas les colonnes 'sub' ou 'rating'. Fichier ignoré.")
            except Exception as e:
                print(f"ERREUR lors de la lecture de '{rating_file_path}': {e}. Fichier ignoré.")
        else:
            print(f"AVERTISSEMENT: '{rating_file_path}' non trouvé. Fichier ignoré.")

    if all_ratings_df.empty:
        print("\nAucune donnée de notation valide n'a été collectée. Arrêt du script.")
        return

    print("\n--- Agrégation des notations ---")
    average_ratings = all_ratings_df.groupby('sub')['rating'].mean().reset_index()
    average_ratings.rename(columns={'rating': 'average_rating'}, inplace=True)
    average_ratings['sub'] = average_ratings['sub'].apply(lambda x: f"{int(float(x)):03d}")
    print(f"{len(average_ratings)} sujets traités à partir des 3 évaluateurs.")

    # --- Étape 2 : Ajouter les notes depuis le fichier additionnel ---
    print("\n--- Ajout des notes depuis le fichier supplémentaire ---")
    final_df = average_ratings # On commence avec les données que l'on a déjà

    if os.path.exists(additional_ratings_path):
        print(f"Lecture du fichier de ratings supplémentaire : {additional_ratings_path}")
        try:
            df_additional = pd.read_csv(additional_ratings_path, dtype=str)
            
            # Déterminer le nom de la colonne de rating ('average_rating' ou 'rating')
            rating_col_name = 'average_rating' if 'average_rating' in df_additional.columns else 'rating'

            if 'sub' in df_additional.columns and rating_col_name in df_additional.columns:
                # Normaliser les colonnes pour la fusion
                df_additional.rename(columns={rating_col_name: 'average_rating'}, inplace=True)
                df_additional['sub'] = df_additional['sub'].apply(lambda x: f"{int(float(x)):03d}")
                df_additional['average_rating'] = pd.to_numeric(df_additional['average_rating'], errors='coerce')
                df_additional.dropna(subset=['average_rating'], inplace=True)

                # Isoler les sujets qui ne sont pas déjà dans notre liste
                processed_subs = set(average_ratings['sub'])
                missing_subs_df = df_additional[~df_additional['sub'].isin(processed_subs)]
                
                if not missing_subs_df.empty:
                    print(f"{len(missing_subs_df)} nouveau(x) sujet(s) trouvé(s) à ajouter.")
                    # Concaténer les deux listes de sujets
                    final_df = pd.concat([average_ratings, missing_subs_df[['sub', 'average_rating']]], ignore_index=True)
                else:
                    print("Aucun nouveau sujet à ajouter depuis le fichier supplémentaire.")

            else:
                print("AVERTISSEMENT: Le fichier de ratings supplémentaire n'a pas les colonnes 'sub' ou de rating. Fichier ignoré.")
        
        except Exception as e:
            print(f"ERREUR lors du traitement du fichier supplémentaire : {e}. Le fichier sera ignoré.")
    else:
        print("AVERTISSEMENT: Fichier de ratings supplémentaire non trouvé. Ignoré.")

    # --- Étape 3 : Sauvegarde du résultat final ---
    # Trier le résultat final par sujet pour un fichier propre
    final_df.sort_values(by='sub', inplace=True)

    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        final_df.to_csv(output_csv_path, index=False)
        print(f"\n✅ Fichier final ({len(final_df)} sujets) sauvegardé avec succès ici : {output_csv_path}")
    except Exception as e:
        print(f"❌ ERREUR lors de la sauvegarde du CSV final : {e}")

# --- Définir les chemins ---
# Dossier contenant les 3 sous-dossiers des évaluateurs
base_ratings_directory = '/Users/cyriltelley/Library/CloudStorage/OneDrive-SharedLibraries-HESSO/Franceschiello Benedetta - 1_Cyril_Telley/QC/Evaluation/eye_qc_83/2_ratings/2_remaining_68'
# Fichier contenant les ratings déjà moyennés pour les sujets manquants
additional_ratings_file = '/Users/cyriltelley/Desktop/MSE/Second_semester/PA-MReye/Codes/MREyeQC_PA/data/rating/ratings.csv'
# Chemin du fichier de sortie final
output_final_csv_path = '/Users/cyriltelley/Desktop/MSE/Second_semester/PA-MReye/Codes/MREyeQC_PA/data/bids_csv/rating.csv'

# --- Exécuter la fonction ---
calculate_and_combine_ratings(
    base_ratings_directory,
    additional_ratings_file,
    output_final_csv_path
)