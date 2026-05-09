import pandas as pd
from PIL import Image
import os

def visualize_day_night_pairs(csv_filename, night_base_dir, day_base_dir, output_dir):
    # Load the CSV data
    df = pd.read_csv(csv_filename)
    
    # Filter for rows where 'skipped' is False
    matches = df[df['skipped'] == False].copy()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    for index, row in matches.iterrows():
        # Construct full file paths
        night_path = os.path.join(night_base_dir, row['night_photo'])
        day_path = os.path.join(day_base_dir, row['day_image'])
        
        if not os.path.exists(night_path) or not os.path.exists(day_path):
            print(f"Skipping {row['night_photo']}: One of the images was not found.")
            continue

        try:
            # Open images
            night_img = Image.open(night_path)
            day_img = Image.open(day_path)

            # Standardize width (matching day image width to night image width)
            target_width = night_img.width
            aspect_ratio = day_img.height / day_img.width
            new_day_height = int(target_width * aspect_ratio)
            day_img_resized = day_img.resize((target_width, new_day_height), Image.Resampling.LANCZOS)

            # Create a new blank image for the vertical stack
            total_height = night_img.height + day_img_resized.height
            combined_img = Image.new('RGB', (target_width, total_height))

            # Paste images: Day on top, Night on bottom
            combined_img.paste(day_img_resized, (0, 0))
            combined_img.paste(night_img, (0, day_img_resized.height))

            # Save the result
            output_filename = f"stacked_{row['night_photo']}"
            combined_img.save(os.path.join(output_dir, output_filename))
            print(f"Successfully processed: {output_filename}")

        except Exception as e:
            print(f"Error processing {row['night_photo']}: {e}")

# --- Configuration ---
CSV_FILE = 'bia-matches.csv' 
NIGHT_DIR = 'bia-matched'
DAY_DIR = 'urban-mosaic/washington-square'
OUTPUT_DIR = 'visualized_pairs'

# Run the function
visualize_day_night_pairs(CSV_FILE, NIGHT_DIR, DAY_DIR, OUTPUT_DIR)