import os
from PIL import Image, ImageEnhance

# Function to adjust brightness of an image
def adjust_brightness(image, factor):
    # Enhance brightness (factor < 1 darkens, > 1 lightens)
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

# Specify the folder path containing images
folder_path = "../imagens_treino/Belgium_2009/replicas" 

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
        # Open the original image
        img_path = os.path.join(folder_path, filename)
        original_img = Image.open(img_path)

        # Make a darker version
        darker_img = adjust_brightness(original_img, 0.7)
        darker_filename = f"{os.path.splitext(filename)[0]}_darker.png"
        darker_path = os.path.join(folder_path, darker_filename)
        darker_img.save(darker_path)
        print(f"Saved: {darker_path}")

        # Make a lighter version
        lighter_img = adjust_brightness(original_img, 1.3)
        lighter_filename = f"{os.path.splitext(filename)[0]}_lighter.png"
        lighter_path = os.path.join(folder_path, lighter_filename)
        lighter_img.save(lighter_path)
        print(f"Saved: {lighter_path}")

print("Image processing complete!")

