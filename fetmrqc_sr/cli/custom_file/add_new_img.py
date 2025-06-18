import os
import re
import csv
import shutil  # Imported for file copying operations

def save_with_new_names(source_base_path, output_base_path, start_number=84, log_filename="mapping_log.csv"):
    """
    Copies folders and files from a source directory to a new destination,
    renaming them sequentially in the process. It logs the mapping of old
    to new names in a CSV file.

    Args:
        source_base_path (str): The path to the directory containing the original 'sub-XXX' folders.
        output_base_path (str): The path to the directory where the renamed folders and files will be saved.
        start_number (int): The starting number for the new names.
        log_filename (str): The name of the CSV file to save the mapping log.
    """

    # Ensure the main output directory exists. If not, create it.
    os.makedirs(output_base_path, exist_ok=True)
    print(f"Output will be saved in: {os.path.abspath(output_base_path)}")

    # Get a list of all items in the source directory
    try:
        items = sorted(os.listdir(source_base_path))
    except FileNotFoundError:
        print(f"Error: Source directory not found at '{source_base_path}'. Please check the path. Exiting.")
        return

    # Filter for directories that match the 'sub-XXX' pattern
    subject_folders = [
        item for item in items
        if os.path.isdir(os.path.join(source_base_path, item)) and re.match(r'sub-\d+', item)
    ]

    if not subject_folders:
        print(f"No 'sub-XXX' folders found in '{source_base_path}'. Exiting.")
        return

    # Prepare for CSV logging with more descriptive headers
    log_data = [['Old_Folder_Name', 'New_Folder_Name', 'Old_Subject_Number', 'New_Subject_Number']]
    current_new_number = start_number

    print("\n--- Starting Copy and Rename Process ---")

    # Iterate through each valid source folder
    for old_folder_name in subject_folders:
        match = re.search(r'sub-(\d+)', old_folder_name)
        if not match:
            print(f"Warning: Could not parse subject number from folder '{old_folder_name}'. Skipping.")
            continue

        old_sub_number = match.group(1)
        new_sub_number_str = f"{current_new_number:03d}" # Format new number with leading zeros

        # Define the new folder name and its full path in the destination
        new_folder_name = f"sub-{new_sub_number_str}"
        new_folder_path = os.path.join(output_base_path, new_folder_name)

        print(f"Mapping: '{old_folder_name}'  ->  '{new_folder_name}'")
        log_data.append([old_folder_name, new_folder_name, old_sub_number, new_sub_number_str])

        # Define the source and destination paths for the 'anat' subdirectory
        source_anat_path = os.path.join(source_base_path, old_folder_name, 'anat')
        dest_anat_path = os.path.join(new_folder_path, 'anat')

        # Proceed only if the source 'anat' directory exists
        if os.path.exists(source_anat_path) and os.path.isdir(source_anat_path):
            # Create the destination directory structure (e.g., data/img/sub-084/anat)
            os.makedirs(dest_anat_path, exist_ok=True)
            print(f"  Created directory: {dest_anat_path}")

            # Process each file within the source 'anat' directory
            for filename in os.listdir(source_anat_path):
                # Construct the new filename by replacing the old subject ID with the new one
                new_filename = re.sub(f'sub-{old_sub_number}', f'sub-{new_sub_number_str}', filename)

                source_file_path = os.path.join(source_anat_path, filename)
                dest_file_path = os.path.join(dest_anat_path, new_filename)

                # Copy the file from the source to the destination with its new name
                try:
                    shutil.copy2(source_file_path, dest_file_path)
                    print(f"    Copied and renamed: '{filename}' -> '{new_filename}'")
                except Exception as e:
                    print(f"    ERROR copying file {source_file_path} to {dest_file_path}: {e}")
        else:
            print(f"  Warning: 'anat' sub-directory not found in '{old_folder_name}'. No files were copied for this subject.")

        current_new_number += 1
        print("-" * 25)

    # Save the log to a CSV file in the main output directory
    log_file_path = os.path.join(output_base_path, log_filename)
    try:
        with open(log_file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerows(log_data)
        print(f"\nProcessing complete. Mapping log saved to: {log_file_path}")
    except IOError as e:
        print(f"ERROR writing CSV log file '{log_file_path}': {e}")


# --- Configuration ---
# Set the path to your original, unsorted data
source_directory_path = "data/excluded_subjects_imgs"

# Set the path where the newly named files and folders will be saved
output_directory_path = "data/img"

# Run the function to copy and rename the data
save_with_new_names(
    source_base_path=source_directory_path,
    output_base_path=output_directory_path,
    start_number=84
)