import pandas as pd
import os

def merge_ratings_to_bids_csv(
    ratings_csv_path,
    renaming_log_path,
    bids_csv_path
):
    """
    Met à jour un fichier CSV BIDS avec des ratings en utilisant un fichier de log pour le mappage.
    Cette version normalise les identifiants pour assurer la correspondance (ex: '3' devient '003').
    """
    print("--- Démarrage de la mise à jour des ratings ---")
    print(f"Fichier des ratings: {ratings_csv_path}")
    print(f"Fichier de mappage (log): {renaming_log_path}")
    print(f"Fichier BIDS à mettre à jour: {bids_csv_path}")
    print("-" * 60)

    try:
        df_ratings = pd.read_csv(ratings_csv_path, dtype=str)
        df_log = pd.read_csv(renaming_log_path, dtype=str)
        df_bids = pd.read_csv(bids_csv_path, dtype=str)
        print("Tous les fichiers CSV ont été chargés avec succès.")
    except FileNotFoundError as e:
        print(f"❌ ERREUR: Fichier non trouvé : {e.filename}. Veuillez vérifier les chemins.")
        return
    except Exception as e:
        print(f"❌ ERREUR: Un problème est survenu lors de la lecture d'un fichier CSV : {e}")
        return

    # --- 3. Création du mappage : Old ID -> New ID ---
    # On normalise les clés pour qu'elles soient toujours sur 3 chiffres (ex: '003')
    old_to_new_map = {}
    print("Création du dictionnaire de mappage (Old -> New) avec normalisation des IDs...")
    for index, row in df_log.iterrows():
        try:
            old_id = row['Old_Subject_Number']
            new_id = row['New_Subject_Number']
            # LA CORRECTION EST ICI : on formate l'ID sur 3 chiffres
            normalized_old_id = f"{int(old_id):03d}"
            old_to_new_map[normalized_old_id] = new_id
        except (ValueError, TypeError):
            print(f"AVERTISSEMENT: L'ID '{row['Old_Subject_Number']}' dans renaming_log.csv n'est pas un nombre valide et sera ignoré.")

    print(f"{len(old_to_new_map)} mappages créés depuis {os.path.basename(renaming_log_path)}.")

    # --- 4. Création du mappage final : New ID -> Rating ---
    final_rating_map = {}
    skipped_ratings = 0
    
    for index, row in df_ratings.iterrows():
        try:
            old_sub_id_from_ratings = row['sub']
            rating = row['rating']
            
            # LA CORRECTION EST ICI : on formate aussi l'ID de ratings.csv avant la recherche
            normalized_old_sub_id = f"{int(old_sub_id_from_ratings):03d}"
            
            new_sub_id = old_to_new_map.get(normalized_old_sub_id)
            
            if new_sub_id:
                final_rating_map[new_sub_id] = rating
            else:
                # Cet avertissement ne devrait plus apparaître pour les cas comme '3' vs '003'
                print(f"AVERTISSEMENT: Le sujet '{old_sub_id_from_ratings}' (normalisé en '{normalized_old_sub_id}') n'a pas été trouvé dans renaming_log.csv.")
                skipped_ratings += 1
        except (ValueError, TypeError):
            print(f"AVERTISSEMENT: L'ID '{row['sub']}' dans ratings.csv n'est pas un nombre valide et sera ignoré.")
            skipped_ratings += 1
            
    print(f"{len(final_rating_map)} ratings prêts à être transférés.")
    if skipped_ratings > 0:
        print(f"{skipped_ratings} ratings ont été ignorés faute de correspondance.")
    print("-" * 60)

    # --- 5. Mise à jour du fichier BIDS ---
    if 'rating' not in df_bids.columns:
        print("Création de la colonne 'rating' dans le fichier BIDS.")
        df_bids['rating'] = pd.NA
    
    df_bids = df_bids.set_index('sub')
    updated_count = 0
    not_found_in_bids = 0

    for new_id, rating_value in final_rating_map.items():
        # On normalise aussi le nouvel ID avant de chercher dans le fichier BIDS final
        normalized_new_id = f"{int(new_id):03d}"
        if normalized_new_id in df_bids.index:
            df_bids.loc[normalized_new_id, 'rating'] = rating_value
            updated_count += 1
        else:
            print(f"AVERTISSEMENT: Le sujet final '{new_id}' (normalisé en '{normalized_new_id}') n'a pas été trouvé dans bids_csv.csv.")
            not_found_in_bids += 1

    df_bids = df_bids.reset_index()

    print(f"\n{updated_count} lignes ont été mises à jour dans la colonne 'rating'.")
    if not_found_in_bids > 0:
        print(f"{not_found_in_bids} sujets n'ont pas été trouvés dans le fichier BIDS final.")

    # --- 6. Sauvegarde du fichier mis à jour ---
    try:
        df_bids.to_csv(bids_csv_path, index=False)
        print(f"\n✅ Le fichier '{os.path.basename(bids_csv_path)}' a été sauvegardé avec succès.")
    except Exception as e:
        print(f"❌ ERREUR: Impossible de sauvegarder le fichier mis à jour : {e}")
        return
        
    print("--- Script terminé ---")

# --- Définissez vos chemins ici ---
ratings_file = '/Users/cyriltelley/Library/CloudStorage/OneDrive-SharedLibraries-HESSO/Franceschiello Benedetta - 1_Cyril_Telley/QC/Evaluation/eye_qc_83/2_ratings/excluded_subjects_ratings/ratings.csv'
renaming_log_file = '/Users/cyriltelley/Desktop/MSE/Second_semester/PA-MReye/Codes/MREyeQC_PA/data/excluded_subjects_imgs/renaming_log.csv'
bids_target_file = '/Users/cyriltelley/Desktop/MSE/Second_semester/PA-MReye/Codes/MREyeQC_PA/data/bids_csv/bids_csv_rating.csv'

# --- Exécution du script ---
merge_ratings_to_bids_csv(
    ratings_file,
    renaming_log_file,
    bids_target_file
)