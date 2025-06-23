import os
from PIL import Image
import numpy as np

# Function to add noise to an image
def add_noise(image, noise_level):
    # Convert image to numpy array
    img_array = np.array(image)
    # Generate random noise
    noise = np.random.normal(0, noise_level, img_array.shape)
    # Add noise and clip values to valid range (0-255)
    noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

# Specify the folder path containing images
folder_path = "../imagens_treino/Belgium_2009/replicas"  # Replace with your folder path

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
        # Open the original image
        img_path = os.path.join(folder_path, filename)
        original_img = Image.open(img_path)

        # Add light noise (e.g., noise level 25)
        noisy_img_light = add_noise(original_img, 25)
        light_noise_filename = f"{os.path.splitext(filename)[0]}_light_noise.png"
        light_noise_path = os.path.join(folder_path, light_noise_filename)
        noisy_img_light.save(light_noise_path)
        print(f"Saved: {light_noise_path}")

        # Add heavier noise (e.g., noise level 50)
        noisy_img_heavy = add_noise(original_img, 50)
        heavy_noise_filename = f"{os.path.splitext(filename)[0]}_heavy_noise.png"
        heavy_noise_path = os.path.join(folder_path, heavy_noise_filename)
        noisy_img_heavy.save(heavy_noise_path)
        print(f"Saved: {heavy_noise_path}")

print("Image processing complete!")
