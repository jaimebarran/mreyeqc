import pandas as pd
import re # For regular expressions

def update_iqa_names_extract_id(iqa_csv_path, aux_csv_path, output_csv_path=None):
    """
    Updates the 'name' column in the IQA CSV file based on mappings
    from an auxiliary CSV file, by extracting a common subject ID.

    - IQA 'name' is like 'sub-001_T1w'.
    - Aux 'bids_name' is like 'sub-001'.
    The script extracts 'sub-001' from IQA 'name' to match with Aux 'bids_name'.
    If matched, IQA 'name' is replaced by Aux 'report_eye_name'.
    The column in IQA remains named 'name'.

    Args:
        iqa_csv_path (str): Path to the IQA CSV file.
        aux_csv_path (str): Path to the auxiliary CSV file containing
                            'bids_name' and 'report_eye_name' columns.
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
    
    required_aux_cols = ['bids_name', 'report_eye_name']
    if not all(col in df_aux.columns for col in required_aux_cols):
        print(f"❌ Error: Auxiliary file {aux_csv_path} is missing one or more required columns: {required_aux_cols}.")
        print(f"Available columns in auxiliary file: {df_aux.columns.tolist()}")
        return None

    print("✅ Files loaded successfully.")
    print(f"Original IQA DataFrame shape: {df_iqa.shape}")
    print(f"Auxiliary DataFrame shape: {df_aux.shape} (defines the subjects to update)")

    # --- Prepare mapping from auxiliary file ---
    # Drop duplicates in bids_name from aux file, keeping the first occurrence.
    unique_aux_mappings = df_aux.drop_duplicates(subset=['bids_name'], keep='first').copy()
    
    # Clean bids_name (e.g., 'sub-001') and report_eye_name from aux file
    unique_aux_mappings.loc[:, 'bids_name_cleaned'] = unique_aux_mappings['bids_name'].astype(str).str.strip()
    unique_aux_mappings.loc[:, 'report_eye_name_cleaned'] = unique_aux_mappings['report_eye_name'].astype(str).str.strip()

    name_update_map = pd.Series(
        unique_aux_mappings['report_eye_name_cleaned'].values,
        index=unique_aux_mappings['bids_name_cleaned']
    )
    
    if len(name_update_map) < len(df_aux):
        print(f"⚠️ Warning: Duplicate 'bids_name' entries found in {aux_csv_path}. Kept first occurrence for mapping. Mappings available: {len(name_update_map)}")

    # --- Apply mapping to update 'name' column in IQA DataFrame ---
    print("🔄 Updating 'name' column in IQA data by extracting and matching subject IDs (e.g., 'sub-001')...")
    original_iqa_names = df_iqa['name'].copy() # For comparison and reporting
    
    # Extract the 'sub-XXX' part from IQA's 'name' column (e.g., from 'sub-001_T1w' extract 'sub-001')
    # The regex (sub-\d+) captures "sub-" followed by one or more digits.
    # .str.extract() returns a DataFrame, so we take the first column [0].
    # Ensure it's treated as string and stripped, in case of leading/trailing spaces in extracted part.
    iqa_subject_ids_for_map = df_iqa['name'].astype(str).str.extract(r'(sub-\d+)')[0].str.strip()
    
    # Get the new names based on the map using the extracted subject IDs.
    # Non-matching IDs or IDs not found in the map will result in NaN.
    mapped_new_names = iqa_subject_ids_for_map.map(name_update_map)
    
    # Update the 'name' column in df_iqa:
    # If mapped_new_names has a value (is not NaN), use it.
    # Otherwise (if no match in aux file or ID couldn't be extracted), keep the original df_iqa['name'].
    df_iqa['name'] = mapped_new_names.combine_first(original_iqa_names)

    # --- Report changes ---
    num_changed = (df_iqa['name'] != original_iqa_names).sum()
    print(f"📊 Number of names updated in IQA file: {num_changed}")
    
    if num_changed > 0 and num_changed <= 10 : # Show examples only if a few changes occurred
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
        output_csv_path = iqa_csv_path # Overwrite original file
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

# --- How to use the function ---

# 1. Define the path to your main IQA.csv file
iqa_file_path = "/Users/cyriltelley/Desktop/MSE/Second_semester/PA-MReye/Codes/MREyeQC_PA/data/IQA.csv" # Please verify this path

# 2. Define the path to your auxiliary file (df_aux_83.csv)
aux_file_path_83 = "/Users/cyriltelley/Library/CloudStorage/OneDrive-SharedLibraries-HESSO/Franceschiello Benedetta - 1_Cyril_Telley/data/df_aux_83.csv"

# 3. Define where to save the updated IQA file.
#    To save to a new file (recommended for safety):
output_updated_iqa_path = "/Users/cyriltelley/Desktop/MSE/Second_semester/PA-MReye/Codes/MREyeQC_PA/data/IQA_names_updated_v2.csv" # Changed version for this example
#    To overwrite the original IQA.csv file:
#    output_updated_iqa_path = iqa_file_path
#    or
#    output_updated_iqa_path = None


# 4. Run the function
#    Make sure the IQA file path is correct before running.
if iqa_file_path == "/Users/cyriltelley/Desktop/MSE/Second_semester/PA-MReye/Codes/MREyeQC_PA/data/IQA.csv": # Basic check for placeholder
    updated_iqa_df = update_iqa_names_extract_id(iqa_file_path, aux_file_path_83, output_updated_iqa_path)
    if updated_iqa_df is not None:
        print("\n🎉 Name update process (with ID extraction) completed.")
        # print("Preview of updated IQA data (first 5 rows):")
        # print(updated_iqa_df.head())
    else:
        print("\n🙁 Name update process (with ID extraction) failed.")
else:
    print("\n🛑 Please verify the 'iqa_file_path' variable in the script before running.")