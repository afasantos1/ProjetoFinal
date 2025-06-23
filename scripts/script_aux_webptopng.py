import os
from PIL import Image

# Specify the folder path containing images
folder_path = "../imagens_treino/FR2007 treaty"

# Define which extensions to convert
to_convert = ('.webp', '.jpg', '.jpeg')

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    ext = os.path.splitext(filename)[1].lower()
    if ext in to_convert:
        # Open the image
        src_path = os.path.join(folder_path, filename)
        img = Image.open(src_path)

        # Define the new .png filename
        png_filename = f"{os.path.splitext(filename)[0]}.png"
        png_path = os.path.join(folder_path, png_filename)

        # Save as PNG
        img.save(png_path, 'PNG')
        print(f"Converted: {src_path} -> {png_path}")

print("Conversion complete!")
