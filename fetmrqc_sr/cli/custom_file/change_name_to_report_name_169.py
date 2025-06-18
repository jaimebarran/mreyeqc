import pandas as pd
import re # For regular expressions

def update_iqa_names_flexible_aux(iqa_csv_path,
                                  aux_csv_path,
                                  aux_key_column,
                                  aux_value_column,
                                  output_csv_path=None):
    """
    Updates the 'name' column in the IQA CSV file based on mappings
    from an auxiliary CSV file, by extracting a common subject ID.

    - IQA 'name' is assumed to be like 'sub-001_T1w' or similar BIDS format.
    - The script extracts the 'sub-XXX' part from IQA 'name'.
    - This extracted ID is then matched against the 'aux_key_column' in the aux_csv_file.
    - If matched, the IQA 'name' is replaced by the content of 'aux_value_column'
      from the aux_csv_file.
    - The column in IQA remains named 'name'.

    Args:
        iqa_csv_path (str): Path to the IQA CSV file.
        aux_csv_path (str): Path to the auxiliary CSV file.
        aux_key_column (str): Name of the column in aux_csv_path that contains
                              the subject identifiers (e.g., 'sub-001') to match against.
        aux_value_column (str): Name of the column in aux_csv_path that contains
                                the new names to update IQA with.
        output_csv_path (str, optional): Path to save the updated IQA CSV.
                                         If None, iqa_csv_path is overwritten.

    Returns:
        pandas.DataFrame or None: The updated IQA DataFrame if successful, else None.
    """
    print(f"🔄 Loading IQA file from: {iqa_csv_path}")
    try:
        df_iqa = pd.read_csv(iqa_csv_path)
    except FileNotFoundError:
        print(f"❌ Error: IQA file not found at {iqa_csv_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading IQA file: {e}")
        return None

    print(f"🔄 Loading auxiliary mapping file from: {aux_csv_path}")
    try:
        df_aux = pd.read_csv(aux_csv_path)
    except FileNotFoundError:
        print(f"❌ Error: Auxiliary file not found at {aux_csv_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading auxiliary file: {e}")
        return None

    # --- Validate required columns ---
    if 'name' not in df_iqa.columns:
        print(f"❌ Error: 'name' column not found in IQA file: {iqa_csv_path}")
        print(f"Available columns in IQA: {df_iqa.columns.tolist()}")
        return None
    
    required_aux_cols = [aux_key_column, aux_value_column]
    if not all(col in df_aux.columns for col in required_aux_cols):
        print(f"❌ Error: Auxiliary file {aux_csv_path} is missing one or more required columns: {required_aux_cols}.")
        print(f"Available columns in auxiliary file: {df_aux.columns.tolist()}")
        return None

    print("✅ Files loaded successfully.")
    print(f"Original IQA DataFrame shape: {df_iqa.shape}")
    print(f"Auxiliary DataFrame shape: {df_aux.shape} (defines the subjects for potential update)")

    # --- Prepare mapping from auxiliary file ---
    # Drop duplicates in the auxiliary key column, keeping the first occurrence.
    unique_aux_mappings = df_aux.drop_duplicates(subset=[aux_key_column], keep='first').copy()
    
    # Clean key and value columns from the aux file
    unique_aux_mappings.loc[:, 'aux_key_cleaned'] = unique_aux_mappings[aux_key_column].astype(str).str.strip()
    unique_aux_mappings.loc[:, 'aux_value_cleaned'] = unique_aux_mappings[aux_value_column].astype(str).str.strip()

    name_update_map = pd.Series(
        unique_aux_mappings['aux_value_cleaned'].values,
        index=unique_aux_mappings['aux_key_cleaned']
    )
    
    if len(name_update_map) < len(df_aux):
        print(f"⚠️ Warning: Duplicate '{aux_key_column}' entries found in {aux_csv_path}. Kept first occurrence for mapping. Mappings available: {len(name_update_map)}")

    # --- Apply mapping to update 'name' column in IQA DataFrame ---
    print(f"🔄 Updating 'name' column in IQA data by extracting 'sub-XXX' and matching against '{aux_key_column}' from aux file...")
    original_iqa_names = df_iqa['name'].copy() # For comparison and reporting
    
    # Extract the 'sub-XXX' part from IQA's 'name' column
    iqa_subject_ids_for_map = df_iqa['name'].astype(str).str.extract(r'(sub-\d+)')[0].str.strip()
    
    # Get the new names based on the map using the extracted subject IDs.
    mapped_new_names = iqa_subject_ids_for_map.map(name_update_map)
    
    # Update the 'name' column in df_iqa
    df_iqa['name'] = mapped_new_names.combine_first(original_iqa_names)

    # --- Report changes ---
    num_changed = (df_iqa['name'] != original_iqa_names).sum()
    print(f"📊 Number of names updated in IQA file: {num_changed}")
    
    if 0 < num_changed <= 10:
        print("🔍 Example of changes (original vs updated for matched rows):")
        changes_df = pd.DataFrame({
            'original_name': original_iqa_names[df_iqa['name'] != original_iqa_names],
            'updated_name': df_iqa['name'][df_iqa['name'] != original_iqa_names]
        })
        print(changes_df.head())
    elif num_changed > 10:
        print("🔍 More than 10 names were updated. Check the output file for all changes.")
    
    print(f"Updated IQA DataFrame shape: {df_iqa.shape}")

    # --- Save the updated DataFrame ---
    if output_csv_path is None:
        output_csv_path = iqa_csv_path
        print(f"📝 Output path not specified, will overwrite original IQA file: {output_csv_path}")
    else:
        print(f"📝 Saving updated IQA data to: {output_csv_path}")

    try:
        df_iqa.to_csv(output_csv_path, index=False)
        print(f"✅ Successfully saved updated IQA data to {output_csv_path}")
    except Exception as e:
        print(f"❌ Error saving updated IQA data: {e}")
        return None

    return df_iqa

# --- How to use the function for your new case (subjects 84-169) ---

# 1. Define the path to your main IQA.csv file
iqa_file_path = "/Users/cyriltelley/Desktop/MSE/Second_semester/PA-MReye/Codes/MREyeQC_PA/data/IQA.csv" # Please verify this path

# 2. Define the path to your NEW auxiliary file (df_aux_1210.csv)
aux_file_path_1210 = "/Users/cyriltelley/Library/CloudStorage/OneDrive-SharedLibraries-HESSO/Franceschiello Benedetta - 1_Cyril_Telley/data/df_aux_1210.csv"

# 3. Define the column names in df_aux_1210.csv to be used
aux_key_col_for_1210 = "bids"   # This column in df_aux_1210.csv contains 'sub-XXX'
aux_value_col_for_1210 = "report" # This column in df_aux_1210.csv contains the new names

# 4. Define where to save the updated IQA file.
#    It's highly recommended to save to a NEW file if you've already run the script for subjects 1-83,
#    so you don't overwrite those previous changes unless this IQA file is the result of that first update.
#    If iqa_file_path is already the output from the first script run, then overwriting it might be intended.
output_updated_iqa_path_v3 = "/Users/cyriltelley/Desktop/MSE/Second_semester/PA-MReye/Codes/MREyeQC_PA/data/IQA_names_updated_v3.csv"
#    To overwrite the file specified in iqa_file_path:
#    output_updated_iqa_path_v3 = iqa_file_path
#    or
#    output_updated_iqa_path_v3 = None


# 5. Run the function

iqa_file_path == "/Users/cyriltelley/Desktop/MSE/Second_semester/PA-MReye/Codes/MREyeQC_PA/data/IQA_names_updated_v2.csv"
updated_iqa_df_v3 = update_iqa_names_flexible_aux(
    iqa_csv_path=iqa_file_path,
    aux_csv_path=aux_file_path_1210,
    aux_key_column=aux_key_col_for_1210,
    aux_value_column=aux_value_col_for_1210,
    output_csv_path=output_updated_iqa_path_v3
)
if updated_iqa_df_v3 is not None:
    print("\n🎉 Name update process (for subjects defined in df_aux_1210.csv) completed.")
else:
    print("\n🙁 Name update process (for subjects defined in df_aux_1210.csv) failed.")