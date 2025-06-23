import os
from PIL import Image, ImageEnhance
import numpy as np

def rotate_image(image, angle):
    return image.rotate(angle, expand=True)

def adjust_brightness(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def add_noise(image, noise_level):
    img_array = np.array(image)
    noise = np.random.normal(0, noise_level, img_array.shape)
    noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_array)

def save_image(image, path):
    image.save(path)
    print(f"Saved: {path}")

def process_images(input_folder, output_folder, rotation_angle=15):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(valid_extensions):
            base_name = os.path.splitext(filename)[0]
            image_path = os.path.join(input_folder, filename)

            try:
                original = Image.open(image_path).convert('RGBA')
                variants = {}

                # Step 1: Original and rotated
                variants["original"] = original
                variants["left"] = rotate_image(original, rotation_angle)
                variants["right"] = rotate_image(original, -rotation_angle)

                # Step 2: Brightness on original and rotated
                new_variants = {}
                for name, img in variants.items():
                    new_variants[f"{name}_darker"] = adjust_brightness(img, 0.7)
                    new_variants[f"{name}_lighter"] = adjust_brightness(img, 1.3)
                variants.update(new_variants)

                # Step 3: Save all base images (before noise)
                for name, img in variants.items():
                    save_image(img, os.path.join(output_folder, f"{base_name}_{name}.png"))

                # Step 4: Add noise to each variant
                for name, img in variants.items():
                    light_noise = add_noise(img, 25)
                    heavy_noise = add_noise(img, 50)
                    save_image(light_noise, os.path.join(output_folder, f"{base_name}_{name}_light_noise.png"))
                    save_image(heavy_noise, os.path.join(output_folder, f"{base_name}_{name}_heavy_noise.png"))

            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")

if __name__ == "__main__":
    input_dir = "../imagens_treino/PT2016-2/resized"
    output_dir = "../imagens_treino/PT2016-2/resized-replicas"
    process_images(input_dir, output_dir)
    print("✅ All image augmentations complete!")
