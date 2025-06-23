import os
from skimage import io

# Set the directory containing your images
image_dir = '../alt_imgs'  # change this to your directory

def main():
    # Get list of image files in the directory (you can adjust the file extensions as needed)
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]

    if not image_files:
        print("No image files found in the directory.")
        return

    print(f"Found {len(image_files)} images in {image_dir}.\n")
    
    # Loop through each image file
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        try:
            image = io.imread(img_path)
            # The shape is usually in the order (height, width, channels)
            if image.ndim == 2:
                print(f"{img_file}: Grayscale image with shape {image.shape} (1 channel)")
            else:
                print(f"{img_file}: Image shape {image.shape} (channels: {image.shape[2]})")
        except Exception as e:
            print(f"Error reading {img_file}: {e}")

if __name__ == '__main__':
    main()
