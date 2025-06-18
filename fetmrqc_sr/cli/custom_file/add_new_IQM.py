import pandas as pd
import re # For regular expressions - though not strictly needed for this version's core matching

def merge_iqa_ship_direct_corrected(iqa_csv_path, ship_tsv_path, output_csv_path=None):
    """
    Merges columns from SHIP1210.tsv (from 'cjv' column onwards) directly into IQA.csv.

    Matching logic based on user clarification:
    - IQA.csv 'name' column (e.g., 'sub-001_T1w') is used as the key.
    - SHIP1210.tsv 'bids_name' column (e.g., 'sub-001_T1w') is used as the key.
    The columns are matched directly after string cleaning.

    Args:
        iqa_csv_path (str): Full path to the IQA.csv file.
        ship_tsv_path (str): Full path to the SHIP1210.tsv file.
        output_csv_path (str, optional): Full path to save the merged CSV.
                                         If None, iqa_csv_path is overwritten.
    Returns:
        pandas.DataFrame or None: The merged DataFrame if successful, otherwise None.
    """
    print("🔄 Starting direct data merging process (IQA <-> SHIP) with direct key matching...")
    print(f"IQA.csv: {iqa_csv_path}")
    print(f"SHIP data file (TSV): {ship_tsv_path}")
    print("-" * 50)

    # --- 1. Load Data ---
    try:
        df_iqa = pd.read_csv(iqa_csv_path)
        df_ship = pd.read_csv(ship_tsv_path, sep='\t', low_memory=False)
    except FileNotFoundError as e:
        print(f"❌ Error: Input file not found. {e}")
        return None
    except pd.errors.EmptyDataError as e:
        print(f"❌ Error: Input file is empty. {e}")
        return None
    except Exception as e:
        print(f"❌ An unexpected error occurred while reading files: {e}")
        return None

    print("✅ Initial data loaded successfully.")
    print(f"IQA.csv original shape: {df_iqa.shape}")
    print(f"SHIP ({ship_tsv_path.split('/')[-1]}) original shape: {df_ship.shape}")
    print("-" * 50)

    # --- 1b. Validate Essential Columns ---
    # Based on clarification: IQA 'name' is 'sub-001_T1w', SHIP 'bids_name' is 'sub-001_T1w'
    required_cols_iqa = ["name"]
    required_cols_ship = ["bids_name", "cjv"]

    if 'name' not in df_iqa.columns:
        print(f"❌ Error: IQA.csv is missing 'name' column. Found: {df_iqa.columns.tolist()}")
        return None
    if not all(col in df_ship.columns for col in required_cols_ship):
        print(f"❌ Error: SHIP file ({ship_tsv_path}) is missing one or more required columns (bids_name, cjv). Found: {df_ship.columns.tolist()}")
        return None
    
    print("✅ Essential columns validated.")
    print("-" * 50)

    # --- 2. Prepare Keys for Merging (Clean and Standardize) ---
    iqa_match_id_col = 'iqa_temp_match_id'  # Temporary key column for IQA
    ship_match_id_col = 'ship_temp_match_id' # Temporary key column for SHIP

    # For IQA: Use 'name' column (e.g., 'sub-001_T1w') directly as the basis for the key
    df_iqa[iqa_match_id_col] = df_iqa['name'].astype(str).str.strip()
    
    # For SHIP: Use 'bids_name' column (e.g., 'sub-001_T1w') directly as the basis for the key
    # **CORRECTION**: No extraction, direct use of bids_name.
    print(f"🛠️ Processing SHIP 'bids_name': Using column directly to create merge key '{ship_match_id_col}'.")
    df_ship[ship_match_id_col] = df_ship['bids_name'].astype(str).str.strip()
    
    print("✅ Key columns prepared for merging.")
    print(f"  IQA merge key ('{iqa_match_id_col}' from 'name'): e.g., 'sub-001_T1w'")
    print(f"  SHIP merge key ('{ship_match_id_col}' from 'bids_name'): e.g., 'sub-001_T1w'")
    print("-" * 50)

    # --- 3. Prepare SHIP1210.tsv Data to be Added ---
    # This includes the 'ship_match_id_col' and columns from 'cjv' onwards.
    try:
        cjv_col_index = df_ship.columns.get_loc('cjv')
    except KeyError:
        print(f"❌ Critical Error: 'cjv' column not found in SHIP1210.tsv ({ship_tsv_path}).")
        if iqa_match_id_col in df_iqa.columns: df_iqa.drop(columns=[iqa_match_id_col], inplace=True, errors='ignore')
        if ship_match_id_col in df_ship.columns: df_ship.drop(columns=[ship_match_id_col], inplace=True, errors='ignore')
        return None
        
    ship_payload_cols_slice = df_ship.columns[cjv_col_index:].tolist()
    
    cols_for_ship_subset = [ship_match_id_col]
    cols_for_ship_subset.extend([col for col in ship_payload_cols_slice if col != ship_match_id_col])
    
    missing_cols_in_df_ship = [col for col in cols_for_ship_subset if col not in df_ship.columns]
    if missing_cols_in_df_ship:
        print(f"❌ Error: The following columns selected for SHIP subset were not found in df_ship: {missing_cols_in_df_ship}")
        print(f"Available df_ship columns: {df_ship.columns.tolist()}")
        if iqa_match_id_col in df_iqa.columns: df_iqa.drop(columns=[iqa_match_id_col], inplace=True, errors='ignore')
        if ship_match_id_col in df_ship.columns: df_ship.drop(columns=[ship_match_id_col], inplace=True, errors='ignore')
        return None

    df_ship_subset = df_ship[cols_for_ship_subset].copy()
    print(f"SHIP data subset created with shape: {df_ship_subset.shape}")
    if ship_match_id_col not in df_ship_subset.columns:
         print(f"⚠️ Critical Warning: Merge Key '{ship_match_id_col}' is unexpectedly NOT in df_ship_subset columns.")
    print("-" * 50)
    
    # --- 4. Merge IQA with SHIP Subset ---
    print(f"🔄 Merging IQA with SHIP data subset on: IQA['{iqa_match_id_col}'] <-> SHIP['{ship_match_id_col}']")
    final_df = pd.merge(
        df_iqa,
        df_ship_subset,
        left_on=iqa_match_id_col,
        right_on=ship_match_id_col,
        how='left',
        suffixes=('_iqa', '_ship')
    )
    print(f"Shape after final merge with SHIP data: {final_df.shape}")

    check_ship_col = next((col for col in df_ship_subset.columns if col != ship_match_id_col and col in final_df.columns), None)
    if check_ship_col:
        merged_data_count = final_df[check_ship_col].notna().sum()
        total_iqa_rows = len(df_iqa) # Use original df_iqa length for base count
        print(f"ℹ️ Info: {merged_data_count} out of {total_iqa_rows} IQA rows received data for SHIP column '{check_ship_col}'.")
        if merged_data_count < total_iqa_rows:
             print(f"   ({total_iqa_rows - merged_data_count} IQA rows did not find a match or matched SHIP row had NaN values for this column).")
    print("-" * 50)

    # --- 5. Clean Up Temporary Key Columns ---
    temp_key_cols_to_drop = [iqa_match_id_col]
    if ship_match_id_col in final_df.columns:
        temp_key_cols_to_drop.append(ship_match_id_col)
    
    cols_to_drop_present = [col for col in temp_key_cols_to_drop if col in final_df.columns]
    if cols_to_drop_present:
        final_df.drop(columns=cols_to_drop_present, inplace=True)
        print(f"✅ Temporary key columns ({cols_to_drop_present}) dropped. Final DataFrame shape: {final_df.shape}")
    else:
        print("ℹ️ No temporary key columns needed to be dropped from final_df.")
    print("-" * 50)

    # --- 6. Save the Merged DataFrame ---
    if output_csv_path is None:
        output_csv_path = iqa_csv_path
        print(f"📝 Output path not specified, will overwrite original IQA file: {output_csv_path}")
    else:
        print(f"📝 Saving merged data to: {output_csv_path}")
    
    try:
        final_df.to_csv(output_csv_path, index=False)
        print(f"✅ Successfully merged data and saved to: {output_csv_path}")
    except Exception as e:
        print(f"❌ Error saving the merged DataFrame to {output_csv_path}: {e}")
        return None

    return final_df

# --- Define your file paths (USER NEEDS TO VERIFY/UPDATE THESE) ---
iqa_file = "/Users/cyriltelley/Desktop/MSE/Second_semester/PA-MReye/Codes/MREyeQC_PA/data/IQA_main_v3.csv"
ship_file = "/Users/cyriltelley/Library/CloudStorage/OneDrive-SharedLibraries-HESSO/Franceschiello Benedetta - 1_Cyril_Telley/QC/mriqc-learn/datasets/SHIP1210.tsv"

# Define where to save the output.
output_file = "/Users/cyriltelley/Desktop/MSE/Second_semester/PA-MReye/Codes/MREyeQC_PA/data/IQA_merged_SHIP_direct_corrected.csv" # Changed output filename

# Run the function (ensure paths are correct)
if __name__ == "__main__":
    # This is an example of how to run the script.
    # Make sure to use your actual file paths.
    if iqa_file == "/Users/cyriltelley/Desktop/MSE/Second_semester/PA-MReye/Codes/MREyeQC_PA/data/IQA_main_v3.csv" and \
       ship_file == "/Users/cyriltelley/Library/CloudStorage/OneDrive-SharedLibraries-HESSO/Franceschiello Benedetta - 1_Cyril_Telley/QC/mriqc-learn/datasets/SHIP1210.tsv":
        
        # To prevent accidental overwrite if you run this multiple times without changing output_file
        # you might want to check if output_file already exists or use a dynamic name.
        # For this example, it will overwrite if output_file is the same.

        merged_dataframe = merge_iqa_ship_direct_corrected(
            iqa_csv_path=iqa_file,
            ship_tsv_path=ship_file,
            output_csv_path=output_file
        )

        if merged_dataframe is not None:
            print("\n🎉 Direct merge process (with corrected direct key matching) completed successfully.")
            # print("Preview of the first 5 rows of the merged data:")
            # print(merged_dataframe.head())
            # print(f"\nMerged data saved to: {output_file}")
        else:
            print("\n🙁 Direct merge process (with corrected direct key matching) failed.")
    else:
        print("\n🛑 Please verify the 'iqa_file' and 'ship_file' paths in the script before running.")