import os
import pandas as pd
import shutil
import re

def process_and_rename_segmentations(
    nnunet_inference_path,
    aux_csv_path,
    renaming_log_csv_path,
    output_seg_folder
):
    """
    Traite et renomme les fichiers de segmentation en se basant sur plusieurs mappages CSV.
    Cette version gère de manière flexible différents formats d'identifiants.
    """

    print("--- Démarrage du traitement et du renommage des segmentations ---")
    print(f"Dossier nnUNet inference: {nnunet_inference_path}")
    print(f"Fichier CSV auxiliaire: {aux_csv_path}")
    print(f"Fichier CSV de log de renommage: {renaming_log_csv_path}")
    print(f"Dossier de sortie des segmentations: {output_seg_folder}")
    print("-" * 60)

    # 1. Charger les fichiers CSV en forçant toutes les colonnes à être du texte (str)
    try:
        # LA MODIFICATION EST ICI : ajout de dtype=str
        df_aux = pd.read_csv(aux_csv_path, encoding='utf-8-sig', dtype=str)
        df_renaming_log = pd.read_csv(renaming_log_csv_path, encoding='utf-8-sig', dtype=str)
        print("Chargement des CSV réussi (toutes les colonnes en tant que texte).")
    except Exception as e:
        print(f"Erreur critique lors de la lecture des fichiers CSV: {e}")
        return

    # --- Fonctions d'extraction d'ID robustes ---

    def get_clean_id(s):
        """
        Extrait l'ID BIDS (ex: '001', 'T001') depuis des formats variés 
        comme 'sub-001', 'sub-T001', ou simplement '001'.
        """
        if pd.isna(s):
            return None
        # La conversion en str() est toujours une bonne pratique, même avec dtype=str
        s = str(s).strip()
        
        match_sub = re.search(r'sub-(T?\d+)', s)
        if match_sub:
            return match_sub.group(1)
            
        match_direct = re.fullmatch(r'T?\d+', s)
        if match_direct:
            return s
            
        return None

    def extract_raw_number_string(s):
        """
        Extrait uniquement les chiffres d'une chaîne pour le nouvel ID.
        """
        if pd.isna(s):
            return None
        s = str(s).strip()
        cleaned = re.sub(r'\D', '', s)
        return cleaned if cleaned else None

    # 2. Créer le mappage `subject_to_bids_map` (Nom de fichier -> ID BIDS)
    subject_to_bids_map = {}
    print("\n--- Création de subject_to_bids_map ---")
    for index, row in df_aux.iterrows():
        # Comme on a utilisé dtype=str, row['subject'] est déjà du texte. .strip() reste utile.
        subject_id = row['subject'].strip()
        bids_id_extracted = get_clean_id(row['bids'])
        
        if bids_id_extracted:
            subject_to_bids_map[subject_id] = bids_id_extracted
        else:
            print(f"AVERTISSEMENT: Impossible de parser l'ID BIDS de '{row['bids']}' pour le sujet '{subject_id}'.")
            
    print(f"subject_to_bids_map créé avec {len(subject_to_bids_map)} entrées.")
    if subject_to_bids_map:
        print("Échantillon:", {k: v for k, v in list(subject_to_bids_map.items())[:5]})
    print("-" * 60)

    # 3. Créer `bids_to_new_id_map` (Ancien ID BIDS -> Nouvel ID)
    bids_to_new_id_map = {}
    print("\n--- Création de bids_to_new_id_map ---")
    
    for index, row in df_renaming_log.iterrows():
        old_bids_id = get_clean_id(row['Old_Subject_Number'])
        new_raw_id = extract_raw_number_string(row['New_Subject_Number'])
        
        if old_bids_id and new_raw_id:
            try:
                bids_to_new_id_map[old_bids_id] = f"{int(new_raw_id):03d}"
            except (ValueError, TypeError):
                print(f"AVERTISSEMENT: Impossible de convertir le nouvel ID '{new_raw_id}' en entier. Ignoré.")
                
    print(f"bids_to_new_id_map créé avec {len(bids_to_new_id_map)} entrées.")
    if bids_to_new_id_map:
        print("Échantillon:", {k: v for k, v in list(bids_to_new_id_map.items())[:5]})
    print("-" * 60)

    # 4. Itérer, mapper et renommer les fichiers
    os.makedirs(output_seg_folder, exist_ok=True)
    print(f"\n--- Début du traitement des fichiers dans: {nnunet_inference_path} ---")

    processed_count = 0
    skipped_count = 0

    for filename in os.listdir(nnunet_inference_path):
        if filename.startswith('AEye_') and filename.endswith('.nii.gz'):
            original_file_path = os.path.join(nnunet_inference_path, filename)
            match_file = re.match(r'AEye_(\d+)\.nii\.gz', filename)
            
            if not match_file:
                continue

            original_subject_id = match_file.group(1)

            bids_id = subject_to_bids_map.get(original_subject_id)
            if bids_id is None:
                print(f"IGNORÉ: '{filename}' (ID: {original_subject_id}) - Pas de correspondance dans {os.path.basename(aux_csv_path)}.")
                skipped_count += 1
                continue

            new_desired_id = bids_to_new_id_map.get(bids_id)
            if new_desired_id is None:
                print(f"IGNORÉ: '{filename}' (ID BIDS: {bids_id}) - Pas de correspondance dans {os.path.basename(renaming_log_csv_path)}.")
                skipped_count += 1
                continue

            new_seg_filename = f"sub-{new_desired_id}_seg.nii.gz"
            destination_file_path = os.path.join(output_seg_folder, new_seg_filename)
            try:
                shutil.copy2(original_file_path, destination_file_path)
                print(f"✅ PROCESSÉ: '{filename}' -> '{new_seg_filename}'")
                processed_count += 1
            except Exception as e:
                print(f"❌ ERREUR: Impossible de copier '{filename}': {e}.")
                skipped_count += 1

    print("-" * 60)
    print("Traitement terminé.")
    print(f"👍 {processed_count} fichiers traités avec succès.")
    print(f"🤔 {skipped_count} fichiers ignorés.")
    print("--- Fin du script ---")


# --- Définissez vos chemins ici ---
nnunet_inference_folder = '/Users/cyriltelley/Library/CloudStorage/OneDrive-SharedLibraries-HESSO/Franceschiello Benedetta - 1_Cyril_Telley/data/nnUNet_inference_non_labeled_dataset'
aux_csv_file = '/Users/cyriltelley/Library/CloudStorage/OneDrive-SharedLibraries-HESSO/Franceschiello Benedetta - 1_Cyril_Telley/data/df_aux_1210.csv'
renaming_log_csv = '/Users/cyriltelley/Desktop/MSE/Second_semester/PA-MReye/Codes/MREyeQC_PA/data/excluded_subjects_imgs/renaming_log.csv'
output_segmentation_folder = '/Users/cyriltelley/Desktop/MSE/Second_semester/PA-MReye/Codes/MREyeQC_PA/data/seg'

# Exécuter la fonction de traitement
process_and_rename_segmentations(
    nnunet_inference_folder,
    aux_csv_file,
    renaming_log_csv,
    output_segmentation_folder
)