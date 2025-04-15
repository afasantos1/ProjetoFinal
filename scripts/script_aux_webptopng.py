import os
from PIL import Image

# Specify the folder path containing .webp images
folder_path = "../alt_imgs"

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith('.webp'):
        # Open the .webp image
        webp_path = os.path.join(folder_path, filename)
        img = Image.open(webp_path)

        # Convert to PNG and define new filename
        png_filename = f"{os.path.splitext(filename)[0]}.png"
        png_path = os.path.join(folder_path, png_filename)

        # Save as PNG
        img.save(png_path, 'PNG')
        print(f"Converted: {webp_path} -> {png_path}")

print("Conversion complete!")
