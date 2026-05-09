import csv
import os
import shutil

# Configuration
csv_file = 'bia-matches.csv'
output_folder = 'bia-matched'
source_folders = ['night-photos', 'night-photos-all']
prefix = 'bia-'

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

with open(csv_file, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    
    count = 0
    for row in reader:
        # Check if skipped is NOT True (case-insensitive check)
        if row['skipped'].strip().lower() != 'true':
            filename = row['night_photo']
            
            # If the filename in CSV already has 'bia-', 
            # we need to know the original source filename to find it
            source_filename = filename
            if filename.startswith(prefix):
                source_filename = filename[len(prefix):]
            
            # Look for the file in the two source directories
            found = False
            for folder in source_folders:
                source_path = os.path.join(folder, source_filename)
                
                if os.path.exists(source_path):
                    # Define the new name (ensuring it has bia- prefix)
                    new_name = filename if filename.startswith(prefix) else f"{prefix}{filename}"
                    dest_path = os.path.join(output_folder, new_name)
                    
                    # Copy the file
                    shutil.copy2(source_path, dest_path)
                    print(f"Copied: {source_filename} -> {dest_path}")
                    found = True
                    count += 1
                    break
            
            if not found:
                print(f"Warning: Could not find {source_filename} in source folders.")

print(f"\nFinished! Total photos copied to '{output_folder}': {count}")