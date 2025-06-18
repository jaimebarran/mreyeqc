import pandas as pd
import os

def raccourcir_fichier_csv(nom_fichier_entree, colonne_a_couper):
    """
    Lit un fichier CSV, supprime une colonne spécifiée et toutes les colonnes
    qui la suivent, puis sauvegarde le résultat dans un nouveau fichier.

    Args:
        nom_fichier_entree (str): Nom du fichier CSV d'entrée.
        colonne_a_couper (str): Nom de la colonne à partir de laquelle couper (incluse).
    """
    try:
        # 1. S'assurer que le fichier d'entrée existe
        if not os.path.exists(nom_fichier_entree):
            print(f"❌ ERREUR : Le fichier '{nom_fichier_entree}' est introuvable.")
            print("Veuillez vous assurer que ce script est dans le même dossier que votre fichier CSV.")
            return

        # 2. Lire le fichier CSV
        df = pd.read_csv(nom_fichier_entree)
        print(f"Fichier '{nom_fichier_entree}' lu avec succès.")

        # 3. Vérifier si la colonne à couper existe
        if colonne_a_couper not in df.columns:
            print(f"❌ ERREUR : La colonne '{colonne_a_couper}' n'existe pas dans le fichier.")
            return

        # 4. Trouver la position de la colonne et ne garder que celles d'avant
        index_colonne = df.columns.get_loc(colonne_a_couper)
        df_raccourci = df.iloc[:, :index_colonne]
        print(f"Colonnes à partir de '{colonne_a_couper}' (incluse) supprimées.")

        # 5. Définir le nom du fichier de sortie
        base, ext = os.path.splitext(nom_fichier_entree)
        nom_fichier_sortie = f"{base}_shortened{ext}"

        # 6. Sauvegarder le nouveau fichier CSV
        df_raccourci.to_csv(nom_fichier_sortie, index=False)
        print(f"\n✅ Succès ! Le nouveau fichier a été sauvegardé ici : {nom_fichier_sortie}")

    except Exception as e:
        print(f"Une erreur inattendue est survenue : {e}")


# --- Configuration ---

# Le nom de votre fichier d'entrée (doit être dans le même dossier que le script)
fichier_source = 'IQA_main_v3.csv'

# Le nom de la colonne à partir de laquelle tout supprimer
colonne_source = 'seg_snr_total'


# --- Lancement du script ---
if __name__ == "__main__":
    raccourcir_fichier_csv(fichier_source, colonne_source)