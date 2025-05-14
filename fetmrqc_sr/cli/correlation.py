# Re-run the script to ensure full output, especially correlations.
import pandas as pd
import numpy as np # Import numpy for handling potential NaN issues if necessary

# Load the IQA.csv file
iqa_file_path = "fetmrqc_sr/data/IQA.csv"
#bids_file_path = "fetmrqc_sr/data/bids_csv.csv"

try:
    df_iqa_full = pd.read_csv(iqa_file_path)
    print("--- IQA.csv Full Columns (First 5) ---")
    print(df_iqa_full.columns[:5]) # Show only first 5 to keep it brief here
    # It seems IQA.csv is actually the bids_csv.csv with IQM columns appended.
    # Let's check its columns more thoroughly.
    # If it already contains 'rating' and 'sub', and also IQMs,
    # then it might be the only file we need for correlation.

    # Let's assume 'IQA.csv' is the primary file containing both IQMs and ratings.
    # If not, the merge logic would be needed.
    # The previous log output showed IQA.csv having rating columns.
    print(f"'rating' column in IQA.csv: {'rating' in df_iqa_full.columns}")
    print(f"'sub' column in IQA.csv: {'sub' in df_iqa_full.columns}")

    df_to_analyze = df_iqa_full.copy()

    # --- Correlation Analysis ---
    rating_column = 'rating' # This is numeric based on previous checks.

    # Identify potential IQM columns:
    # These are columns that are numeric and are not identifiers or rating-related text columns.
    # Columns from bids_csv that are not IQMs:
    # 'name', 'sub', 'ses', 'run', 'rating', 'rating_text', 'rating1', 'rating1_text', 'rating2', 'rating2_text',
    # 'blur', 'blur_text', 'noise', 'noise_text', 'motion', 'motion_text', 'bgair', 'bgair_text',
    # 'artifacts', 'nselected', 'selected_slices', 'comments', 'time_sec', 'timestamp', 'dataset',
    # 'im', 'mask', 'ratings_json', 'seg', 'seg_proba'
    # We want to correlate 'rating' with the *other* numeric columns that are likely IQMs.

    # Explicitly list known non-IQM columns that might be in df_to_analyze
    # (especially if df_to_analyze is df_iqa_full which seems to be an augmented bids_csv)
    non_iqm_cols = [
        'sub', 'ses', 'run', 'rating', 'rating_text', 'rating1', 'rating1_text',
        'rating2', 'rating2_text', 'blur', 'blur_text', 'noise', 'noise_text',
        'motion', 'motion_text', 'bgair', 'bgair_text', 'nselected', 'time_sec',
        # also remove text/path columns if they exist from the original bids_csv part
        'name', 'artifacts', 'selected_slices', 'comments', 'timestamp', 'dataset',
        'im', 'mask', 'ratings_json', 'seg', 'seg_proba'
    ]
    # Filter out non_iqm_cols that might not be present, to avoid errors
    non_iqm_cols = [col for col in non_iqm_cols if col in df_to_analyze.columns]

    potential_iqm_cols = [col for col in df_to_analyze.columns if col not in non_iqm_cols]
    
    # Ensure IQM columns are numeric and rating column exists and is numeric
    numeric_iqm_cols = [col for col in potential_iqm_cols if pd.api.types.is_numeric_dtype(df_to_analyze[col])]

    if not pd.api.types.is_numeric_dtype(df_to_analyze[rating_column]):
        print(f"Rating column '{rating_column}' is not numeric. Cannot perform correlation.")
    elif not numeric_iqm_cols:
        print("No numeric IQM columns found for correlation analysis.")
    else:
        print(f"\nIdentified {len(numeric_iqm_cols)} potential numeric IQM columns for correlation.")
        
        # Create a DataFrame for correlation, handling potential all-NaN columns
        cols_for_corr = numeric_iqm_cols + [rating_column]
        corr_df = df_to_analyze[cols_for_corr].copy()

        # Drop columns that are entirely NaN, as they cause issues with .corr()
        corr_df.dropna(axis=1, how='all', inplace=True)
        
        # Re-select numeric IQM columns from the cleaned corr_df
        cleaned_numeric_iqm_cols = [col for col in numeric_iqm_cols if col in corr_df.columns]

        if not cleaned_numeric_iqm_cols:
            print("All potential IQM columns were NaN. No correlation possible.")
        else:
            correlations = corr_df.corr()[rating_column].drop(rating_column) # Drop self-correlation
            
            # Sort by absolute correlation value to see strongest relationships
            sorted_correlations = correlations.abs().sort_values(ascending=False)
            
            print(f"\n\n--- Top IQMs Correlated with '{rating_column}' (Absolute Values) ---")
            # Get back the original correlation values (with sign) for the top ones
            top_correlations = correlations.loc[sorted_correlations.index].dropna()
            print(top_correlations.head(20)) # Show top 20

            # Also print correlations with original signs for context
            print(f"\n\n--- All IQM Correlations with '{rating_column}' (Sorted by Absolute Value) ---")
            print(correlations.loc[sorted_correlations.index].dropna())

except FileNotFoundError as e:
    print(f"Error: A file was not found. Details: {e}")
    # If IQA.csv is not found, there's nothing to analyze.
except Exception as e:
    print(f"An error occurred during data processing: {e}")
    import traceback
    traceback.print_exc()