import pandas as pd
import re
import numpy as np # Import numpy for handling potential NaN values if needed

# Load the dataframe.
df = pd.read_csv('/Users/cyriltelley/Desktop/MSE/Second_semester/PA-MReye/Codes/fetmrqc_sr/fetmrqc_sr/data/cyril/bids_csv.csv')

# Display the first 5 rows and the columns and their types
print("Original DataFrame:")
print(df.head().to_markdown(index=False, numalign="left", stralign="left"))
print("\nOriginal DataFrame Information:")
# Check the data type of the 'sub' column and show unique values to understand potential issues
print(df.info())
print("\nData type of 'sub' column:", df['sub'].dtype)
print("Unique values in 'sub' column (first 10):", df['sub'].unique()[:10])


# Function to extract subject ID (e.g., '001' from 'sub-001')
# Updated to handle non-string inputs
def extract_sub_id(sub_input):
    # Convert input to string, handle potential NaN or other types
    sub_string = str(sub_input) if pd.notna(sub_input) else ""
    match = re.search(r'sub-(\d+)', sub_string)
    if match:
        return match.group(1)
    return None # Return None if no match or input was NaN/empty

# Apply the function to create a new column with just the subject ID
df['sub_id'] = df['sub'].apply(extract_sub_id)

# Check if any sub_id is None and show those rows
invalid_sub_id_rows = df[df['sub_id'].isnull()]
if not invalid_sub_id_rows.empty:
    print(f"\nWarning: Could not extract subject ID for {len(invalid_sub_id_rows)} rows. Paths will not be updated for these rows.")
    # Optionally print the problematic rows:
    # print(invalid_sub_id_rows[['sub', 'seg', 'seg_proba']].to_markdown(index=False, numalign="left", stralign="left"))


# Function to update the 'seg' path
def update_seg_path(row):
  sub_id = row['sub_id']
  if sub_id: # Only update if sub_id was extracted
      # Use an f-string to format the path correctly
      return f"/Users/cyriltelley/Desktop/MSE/Second_semester/PA-MReye/Codes/mriqc-learn/data/data_cyril/sub-{sub_id}/sub-{sub_id}_T1w.nii.gz"
  else:
      return row['seg'] # Keep original path if sub_id is None

# Function to update the 'seg_proba' path
def update_seg_proba_path(row):
    sub_id = row['sub_id']
    if sub_id: # Only update if sub_id was extracted
        # Use an f-string to format the path correctly
        return f"/Users/cyriltelley/Desktop/MSE/Second_semester/PA-MReye/Codes/mriqc-learn/data/data_cyril/sub-{sub_id}/sub-{sub_id}_T1w.npz"
    else:
        return row['seg_proba'] # Keep original path if sub_id is None

# Apply the functions row-wise to update the columns
df['seg'] = df.apply(update_seg_path, axis=1)
df['seg_proba'] = df.apply(update_seg_proba_path, axis=1)

# Remove the temporary 'sub_id' column
df = df.drop(columns=['sub_id'])

# Display the first 5 rows of the modified DataFrame
print("\nModified DataFrame (first 5 rows):")
print(df.head().to_markdown(index=False, numalign="left", stralign="left"))

# Save the modified DataFrame to a new CSV file
output_filename = 'bids_csv_updated.csv'
df.to_csv(output_filename, index=False)

print(f"\nModified DataFrame saved to '{output_filename}'")

# Display the first 5 rows of the saved file to confirm
# df_saved = pd.read_csv(output_filename)
# print(f"\nFirst 5 rows of the saved file '{output_filename}':")
# print(df_saved.head().to_markdown(index=False, numalign="left", stralign="left"))