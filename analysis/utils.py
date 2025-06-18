# In analysis/utils.py

import numpy as np
import pandas as pd
# from sklearn.preprocessing import RobustScaler, StandardScaler # Keep if pick_scaler uses them

def load_and_format_data(dataset_path, first_iqm_column_name, move_iqms=False): # Renamed parameters for clarity
    """Load and format a dataset from a CSV file.
    Separates metadata (train_y) from IQM features (train_x) based on first_iqm_column_name.
    Handles NaN/Inf values in features and removes constant feature columns.
    """
    df = pd.read_csv(dataset_path)

    all_columns = df.columns.tolist()
    try:
        first_iqm_index = all_columns.index(first_iqm_column_name)
    except ValueError:
        error_msg = (
            f"Critical Error: The specified 'first_iqm_column_name' ('{first_iqm_column_name}') "
            f"was not found in the CSV file: {dataset_path}.\n"
            f"Please ensure '{first_iqm_column_name}' is a valid column name in your CSV.\n"
            f"Available columns are: {all_columns}"
        )
        raise ValueError(error_msg)

    if move_iqms:
        # This logic for 'move_iqms' uses a hardcoded slice [-14:].
        # If you use this, ensure it's appropriate for your CSV structure.
        print("Warning: 'move_iqms=True' logic uses a hardcoded slice [-14:] which might be fragile.")
        # Be very careful with this reordering logic if used.
        df = df[all_columns[:first_iqm_index] + all_columns[-14:] + all_columns[first_iqm_index:-14]]
        # Update columns list and re-find index if columns were moved
        all_columns = df.columns.tolist()
        try:
            first_iqm_index = all_columns.index(first_iqm_column_name)
        except ValueError:
             raise ValueError(
                f"The 'first_iqm_column_name' ('{first_iqm_column_name}') is no longer "
                f"found after 'move_iqms' reordering. Current columns: {all_columns}")


    train_y = df[all_columns[:first_iqm_index]].copy()  # Metadata (columns before first IQM)
    train_x = df[all_columns[first_iqm_index:]].copy()   # IQM features (columns from first IQM onwards)

    # Cast columns containing "nan" in their name (likely boolean flags for NaN indicators)
    # These are expected to be 0.0 or 1.0 after this cast.
    cols_nan_flags = train_x.columns[train_x.columns.str.contains("nan")]
    if not cols_nan_flags.empty:
        train_x[cols_nan_flags] = train_x[cols_nan_flags].astype(bool).astype(float)

    # Remove rows with any NaN values in any of the feature columns (train_x)
    initial_feature_rows = len(train_x)
    # Keep track of original indices before dropping NaNs from train_x
    original_indices = train_x.index 
    nan_in_features_rows = train_x.isnull().any(axis=1)
    if np.any(nan_in_features_rows):
        train_x = train_x[~nan_in_features_rows]
        # Use the filtered index from train_x to also filter train_y correctly
        train_y = train_y.loc[train_x.index] 
        print(f"Removed {initial_feature_rows - len(train_x)} rows due to NaN values in IQM features.")

    # Remove rows with any Inf values in any of the feature columns (train_x)
    initial_feature_rows = len(train_x) # Update row count
    # Create boolean Series for Inf values properly aligned with current train_x index
    inf_in_features_rows = (train_x == np.inf).any(axis=1) | (train_x == -np.inf).any(axis=1)
    if np.any(inf_in_features_rows):
        train_x = train_x[~inf_in_features_rows]
        # Use the filtered index from train_x to also filter train_y correctly
        train_y = train_y.loc[train_x.index]
        print(f"Removed {initial_feature_rows - len(train_x)} rows due to Inf values in IQM features.")

    # --- All lines related to "site", "rec", "site_rec", and "feta" site filtering are now REMOVED ---

    # Reset indices for clean DataFrames after all filtering
    train_x = train_x.reset_index(drop=True)
    train_y = train_y.reset_index(drop=True)
    
    # Drop constant feature columns from train_x
    if not train_x.empty:
        numeric_train_x = train_x.select_dtypes(include=np.number)
        if not numeric_train_x.empty:
            # skipna=False is important for var() if NaNs might still exist (though they shouldn't here)
            non_constant_numeric_cols = numeric_train_x.loc[:, numeric_train_x.var(skipna=False) != 0].columns 
            # Keep all non-numeric columns (e.g., _nan flags if they are objects/bools)
            # and only the non-constant numeric columns.
            non_numeric_cols = train_x.select_dtypes(exclude=np.number).columns
            cols_to_keep = non_constant_numeric_cols.union(non_numeric_cols)
            train_x = train_x[cols_to_keep]
        else:
            print("Warning: No numeric columns found in train_x to check for constant variance.")
            # If train_x only had non-numeric columns, it remains unchanged by this step.
    
    return train_x, train_y

# Your other functions (pick_scaler, add_group) would follow here.
# Ensure their definitions are also up-to-date with how you intend to use them.
# For example, make sure `add_group` uses the `group_column_name` you pass from the notebook.

def pick_scaler(scaler_name_str): # Renamed parameter for clarity
    # from fetal_brain_qc.qc_evaluation.preprocess import (...) # Check if these custom scalers are actually available and needed
    from sklearn.preprocessing import RobustScaler, StandardScaler # Ensure this is imported

    if scaler_name_str is None or scaler_name_str.lower() in ("none", "passthrough", "passthroughscaler"):
        print("No scaler will be applied.")
        return None # Signifies no scaling
    elif scaler_name_str.lower() == "standardscaler":
        return StandardScaler()
    elif scaler_name_str.lower() == "robustscaler":
        return RobustScaler()
    # Add other standard sklearn scalers if needed
    # elif scaler_name_str.lower() == "groupstandardscaler":
    #     print("Warning: GroupStandardScaler might require specific library (fetal_brain_qc). Using StandardScaler as fallback.")
    #     return StandardScaler() 
    # elif scaler_name_str.lower() == "grouprobustscaler":
    #     print("Warning: GroupRobustScaler might require specific library (fetal_brain_qc). Using RobustScaler as fallback.")
    #     return RobustScaler()
    else:
        raise ValueError(f"Unknown scaler choice: {scaler_name_str}")


def add_group(df_reference, group_col_name): # Renamed parameters for clarity
    """
    Returns the specified group column as a Series from the reference DataFrame.
    'group_col_name' should be the actual name of the column in df_reference
    that you want to use for grouping.
    """
    if group_col_name is None or group_col_name.lower() == "none":
        print("No group column specified for add_group, returning None for groups.")
        return None 
    elif group_col_name in df_reference.columns:
        return df_reference[group_col_name].copy() # Return a copy of the Series
    else:
        # Raise an error or return None with a warning if the column isn't found
        print(f"Warning: Group column '{group_col_name}' not found in the DataFrame for add_group. Available columns: {df_reference.columns.tolist()}")
        return None