import os
import nibabel as nib
import numpy as np

def create_binary_mask(segmentation_path, output_mask_path):
    """
    Loads a multi-label segmentation NIfTI file, converts it to a binary mask
    (all non-zero voxels become 1), and saves it as a new NIfTI file.

    Args:
        segmentation_path (str): Path to the input multi-label segmentation NIfTI file.
        output_mask_path (str): Path to save the output binary mask NIfTI file.
    """
    try:
        # Load the segmentation image
        seg_img = nib.load(segmentation_path)
        seg_data = seg_img.get_fdata()

        # Create a binary mask: set all non-zero voxels to 1, others to 0.
        # Ensure the output data type is appropriate for a mask (e.g., uint8).
        binary_mask_data = (seg_data > 0).astype(np.uint8)

        # Create a new NIfTI image for the binary mask.
        # It's crucial to preserve the original affine and header information
        # so the mask aligns correctly with the original image.
        binary_mask_img = nib.Nifti1Image(binary_mask_data, seg_img.affine, seg_img.header)

        # Ensure the output directory exists.
        # os.path.dirname will get the directory part of the output_mask_path.
        output_dir = os.path.dirname(output_mask_path)
        if output_dir and not os.path.exists(output_dir): # Check if output_dir is not empty
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")

        # Save the binary mask image
        nib.save(binary_mask_img, output_mask_path)
        print(f"Successfully created binary mask: {output_mask_path}")

    except FileNotFoundError:
        print(f"Error: Segmentation file not found at {segmentation_path}")
    except Exception as e:
        print(f"An error occurred processing {segmentation_path}: {e}")

def main():
    """
    Main function to configure paths and iterate through subjects.
    """
    print("--- Binary Mask Creation Script ---")

    # --- User Input for Paths and Patterns ---
    # Try to determine a sensible default base project directory.
    # This assumes the script might be run from the project root or a subfolder.
    # Adjust 'base_project_dir' if this assumption is incorrect for your setup.
    try:
        # If script is in fetmrqc_sr/some_subfolder/this_script.py, this goes to fetmrqc_sr/
        base_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    except NameError: # __file__ is not defined if running in some interactive environments
        base_project_dir = os.getcwd()
        print(f"Warning: Could not automatically determine project root. Using current working directory: {base_project_dir}")
        print("If this is incorrect, please modify 'base_project_dir' in the script or provide full paths below.")


    default_input_seg_dir = os.path.join(base_project_dir, "data", "segmentations_to_binarize") # Suggest a specific input folder
    input_seg_base_dir = input(f"Enter the full path to your multi-label segmentation directory (e.g., where sub-001_mask.nii.gz is) [{default_input_seg_dir}]: ") or default_input_seg_dir
    if not os.path.isdir(input_seg_base_dir):
        print(f"Error: Input directory '{input_seg_base_dir}' not found. Please create it and place your segmentations there or provide a valid path.")
        return

    default_input_seg_pattern = "sub-{subj_id}_seg.nii.gz"
    input_seg_pattern = input(f"Enter the file name pattern for input segmentations (use '{{subj_id}}' as placeholder for e.g. 001) [{default_input_seg_pattern}]: ") or default_input_seg_pattern

    # Output directory as per your request
    default_output_mask_dir = os.path.join(base_project_dir, "fetmrqc_sr", "data", "mask")
    output_mask_base_dir = input(f"Enter the full path for the output binary mask directory [{default_output_mask_dir}]: ") or default_output_mask_dir

    # Output file name pattern as per your request
    default_output_mask_pattern = "sub-{subj_id}_mask.nii.gz"
    output_mask_pattern = input(f"Enter the file name pattern for output masks (use '{{subj_id}}' as placeholder) [{default_output_mask_pattern}]: ") or default_output_mask_pattern
    # --- End User Input ---

    print(f"\n--- Configuration Summary ---")
    print(f"Input Segmentation Directory: {input_seg_base_dir}")
    print(f"Input Segmentation Pattern: {input_seg_pattern.format(subj_id='XXX')}") # Show example
    print(f"Output Mask Directory: {output_mask_base_dir}")
    print(f"Output Mask Pattern: {output_mask_pattern.format(subj_id='XXX')}") # Show example
    print(f"----------------------------\n")

    # Create the main output directory if it doesn't exist
    if not os.path.exists(output_mask_base_dir):
        os.makedirs(output_mask_base_dir)
        print(f"Created base output directory: {output_mask_base_dir}")

    # Iterate through subjects 001 to 083
    for i in range(1, 170):
        subj_id_formatted = f"{i:03d}"  # Formats as "001", "002", ..., "083"

        # Construct full input and output file paths
        segmentation_file_name = input_seg_pattern.format(subj_id=subj_id_formatted)
        output_mask_file_name = output_mask_pattern.format(subj_id=subj_id_formatted)

        segmentation_path = os.path.join(input_seg_base_dir, segmentation_file_name)
        output_mask_path = os.path.join(output_mask_base_dir, output_mask_file_name)

        print(f"\nProcessing subject {subj_id_formatted}...")
        print(f"  Input segmentation: {segmentation_path}")
        print(f"  Output binary mask: {output_mask_path}")
        create_binary_mask(segmentation_path, output_mask_path)

    print("\n--- All processing complete. ---")

if __name__ == "__main__":
    main()
