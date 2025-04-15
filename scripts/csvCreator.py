import os
import csv

def extract_info_from_filename(filename):
    # Get the part before the first underscore (prefix)
    prefix = filename.split('_')[0]
    
    # Get the full picture name
    picture_name = filename
    
    # Get the year, which is a 4-character string starting from the 4th position in the prefix
    year = prefix[3:7]  # Year is the 4 characters starting from the 4th position
    
    return prefix, picture_name, year

def process_images(folder_path, output_csv):
    prefix_to_id = {}  # Dictionary to map prefix to sequential ID
    next_id = 0  # Start with ID 0 for the first group
    
    # Open a CSV file for writing
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header
        writer.writerow(['id', 'picture_name', 'year_of_release'])
        
        # Iterate through the files in the provided folder
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):  # Check for image files
                prefix, picture_name, year = extract_info_from_filename(filename)
                
                # If this prefix hasn't been encountered before, assign it a new ID
                if prefix not in prefix_to_id:
                    prefix_to_id[prefix] = next_id
                    next_id += 1
                
                # Get the ID for the current image's prefix
                image_id = prefix_to_id[prefix]
                
                # Write the data to the CSV file
                writer.writerow([image_id, picture_name, year])
                
    print(f"CSV file '{output_csv}' has been created successfully.")

# Example usage
folder_path = '../alt_imgs'  # Replace with the actual folder path
output_csv = '../alt_imgs/dados_teste.csv'  # Desired output CSV file name
process_images(folder_path, output_csv)
