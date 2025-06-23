from PIL import Image
import os

def resize_images(input_folder, output_folder, target_size):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get all PNG files from input folder
    png_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.png')]
    
    if not png_files:
        print("No PNG files found in the input folder!")
        return
    
    # Process each image
    for filename in png_files:
        try:
            # Open the image
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            
            # Convert to RGBA if not already (preserves transparency)
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            # Calculate the scaling factor to maintain aspect ratio
            width, height = img.size
            target_width, target_height = target_size
            
            # Calculate aspect ratios
            aspect = width / height
            target_aspect = target_width / target_height
            
            # Resize while maintaining aspect ratio
            if aspect > target_aspect:
                # Image is wider than target
                new_width = target_width
                new_height = int(target_width / aspect)
            else:
                # Image is taller than target
                new_height = target_height
                new_width = int(target_height * aspect)
            
            # Resize the image
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create a new blank image with target size and paste resized image in center
            new_img = Image.new('RGBA', target_size, (0, 0, 0, 0))
            paste_x = (target_width - new_width) // 2
            paste_y = (target_height - new_height) // 2
            new_img.paste(img, (paste_x, paste_y))
            
            # Save the new image
            output_path = os.path.join(output_folder, filename)
            new_img.save(output_path, 'PNG')
            # print(f"Processed: {filename}")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Set your parameters here
    INPUT_FOLDER = "../imagens_treino/PT2016-2/"    # Folder with original PNGs
    OUTPUT_FOLDER = "../imagens_treino/PT2016-2/" # Folder where resized PNGs will be saved
    TARGET_SIZE = (500, 500)         # Desired width and height in pixels
    
    resize_images(INPUT_FOLDER, OUTPUT_FOLDER, TARGET_SIZE)
    print("Done!")