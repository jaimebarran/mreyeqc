import os
import pandas as pd

def generate_and_add_paths(rating_csv_path, output_csv_path, base_im_path, base_mask_path, base_seg_path):
    """
    Reads a rating CSV file, generates BIDS-style paths for image, mask, and segmentation
    files based on subject IDs, and adds these paths as new columns ('im', 'mask', 'seg')
    to the DataFrame. The modified DataFrame is then saved.

    Args:
        rating_csv_path (str): Path to the input rating CSV file.
                               This CSV must contain a 'sub' column for subject IDs.
        output_csv_path (str): Path where the modified CSV file will be saved.
        base_im_path (str): The base directory path for the image files.
        base_mask_path (str): The base directory path for the mask files.
        base_seg_path (str): The base directory path for the segmentation files.
                             (Script will append '/mask/sub-XXX_mask.nii.gz' to this)
    """
    try:
        df = pd.read_csv(rating_csv_path)
        print(f"Successfully read '{rating_csv_path}'. Initial shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: The file '{rating_csv_path}' was not found.")
        return
    except Exception as e:
        print(f"Error reading CSV '{rating_csv_path}': {e}")
        return

    if 'sub' not in df.columns:
        print("Error: 'sub' column (for subject ID) not found in the CSV.")
        return

    # Store generated paths in temporary lists
    image_paths_list = []
    mask_paths_list = []
    seg_paths_list = []

    for index, row in df.iterrows():
        sub_id = row['sub']
        # Format subject ID to be like sub-001, sub-002, etc.
        padded_id = f"sub-{str(sub_id).zfill(3)}"

        # --- Image Path Construction ---
        im_filename = f"{padded_id}_T1w.nii.gz"
        image_p = os.path.join(base_im_path, padded_id, "anat", im_filename)
        image_paths_list.append(image_p)

        # --- Mask Path Construction ---
        mask_filename = f"{padded_id}_mask.nii.gz"
        mask_p = os.path.join(base_mask_path, mask_filename)
        mask_paths_list.append(mask_p)

        # --- Segmentation Path Construction ---
        # (User's example: base_seg_path/mask/sub-XXX_mask.nii.gz)
        seg_filename = f"{padded_id}_seg.nii.gz"
        seg_p = os.path.join(base_seg_path, seg_filename)
        seg_paths_list.append(seg_p)

    # Add the new paths as columns. This will overwrite existing columns
    # named 'im', 'mask', 'seg', or create them if they don't exist.
    df['im'] = image_paths_list
    df['mask'] = mask_paths_list
    df['seg'] = seg_paths_list
    
    print(f"\nAdded/Updated 'im', 'mask', and 'seg' columns with generated paths.")

    # Display the head of the modified DataFrame
    print("\n--- Head of the modified DataFrame ---")
    print(df[['sub', 'im', 'mask', 'seg']].head().to_string())

    # Save the modified DataFrame
    try:
        df.to_csv(output_csv_path, index=False)
        print(f"\nSuccessfully saved the modified DataFrame to '{output_csv_path}'")
    except Exception as e:
        print(f"\nError saving modified CSV to '{output_csv_path}': {e}")

# --- Configuration ---
# !! IMPORTANT !! User needs to set these paths appropriately.

# 1. Path to your input rating CSV file (this is your 'bids_csv.csv')
INPUT_RATING_CSV = "/Users/cyriltelley/Desktop/MSE/Second_semester/PA-MReye/Codes/fetmrqc_sr/data/bids_csv.csv" # Assuming it's in the same directory

# 2. Path for the output CSV with the added/updated path columns
OUTPUT_MODIFIED_CSV = "/Users/cyriltelley/Desktop/MSE/Second_semester/PA-MReye/Codes/fetmrqc_sr/data/bids_csv_with_generated_paths.csv"

# 3. Base paths for constructing the full file paths.
#    Replace these with your actual system paths.
#    Example for image base path (root of BIDS-like image dataset):
#    From CSV: /Users/cyriltelley/Desktop/MSE/Second_semester/PA-MReye/Codes/mreyeqc/data/samples_v3_bids
BASE_IMAGE_PATH = "/Users/cyriltelley/Desktop/MSE/Second_semester/PA-MReye/Codes/fetmrqc_sr/data/samples_v3_bids" 

#    Example for mask base path (directory containing sub-XXX_mask.nii.gz files):
#    From CSV: /Users/cyriltelley/Desktop/MSE/Second_semester/PA-MReye/Codes/mreyeqc/data/samples_v3_bids/derivatives/masks
BASE_MASK_PATH = "/Users/cyriltelley/Desktop/MSE/Second_semester/PA-MReye/Codes/fetmrqc_sr/data/mask"   

#    Example for segmentation base path (path before the "/mask/sub-XXX..." part from your example):
#    User's example for seg file: (.../fetmrqc_sr/data/mask/sub-001_mask.nii.gz)
#    This implies base_seg_path should be like: /Users/cyriltelley/Desktop/MSE/Second_semester/PA-MReye/Codes/fetmrqc_sr/data
BASE_SEG_PATH = "/Users/cyriltelley/Desktop/MSE/Second_semester/PA-MReye/Codes/fetmrqc_sr/data/seg"     

if __name__ == "__main__":
    # Basic check if paths have been changed from placeholders
    if BASE_IMAGE_PATH == "/your/actual/base_im_path" or \
       BASE_MASK_PATH == "/your/actual/base_mask_path" or \
       BASE_SEG_PATH == "/your/actual/base_seg_path":
        print("WARNING: You are using placeholder base paths for image, mask, or seg.")
        print("Please update BASE_IMAGE_PATH, BASE_MASK_PATH, and BASE_SEG_PATH in the script with your actual paths before running.")
        print("Execution will proceed with placeholders, which might not be what you want for real data.\n")
    
    generate_and_add_paths(
        INPUT_RATING_CSV,
        OUTPUT_MODIFIED_CSV,
        BASE_IMAGE_PATH,
        BASE_MASK_PATH,
        BASE_SEG_PATH
    )