from PIL import Image
import os

def replicate_images(input_folder, output_folder, rotation_angle=15):

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Supported image extensions (added .webp)
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    
    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(valid_extensions):
            # Open the original image
            image_path = os.path.join(input_folder, filename)
            original_img = Image.open(image_path)
            
            # Get the base filename without extension
            base_name = os.path.splitext(filename)[0]
            
            # Save original image
            original_output = os.path.join(output_folder, f"{base_name}_original.png")
            original_img.save(original_output, 'PNG', quality=95)
            
            # Create and save left rotation
            left_rotated = original_img.rotate(rotation_angle, expand=True)
            left_output = os.path.join(output_folder, f"{base_name}_left.png")
            left_rotated.save(left_output, 'PNG', quality=95)
            
            # Create and save right rotation
            right_rotated = original_img.rotate(-rotation_angle, expand=True)
            right_output = os.path.join(output_folder, f"{base_name}_right.png")
            right_rotated.save(right_output, 'PNG', quality=95)
            
            # Close the original image
            original_img.close()
            
            print(f"Processed: {filename} - Created 3 versions")

# Example usage
if __name__ == "__main__":
    # Specify your input and output folders
    input_dir = "../imagens_treino/Belgium_2009/originals"    # Folder containing original images
    output_dir = "../imagens_treino/Belgium_2009/replicas"  # Folder where replicated images will be saved
    
    try:
        replicate_images(input_dir, output_dir, rotation_angle=15)
        print("Image replication completed successfully!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
