import os
import csv

def list_files_to_csv(folder_path, output_csv='file_list.csv'):
    # Get all files (not directories) in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Write to CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id','Filename'])  # header
        for file in files:
            writer.writerow([372, file])

    print(f"CSV file '{output_csv}' created with {len(files)} file(s).")

if __name__ == '__main__':
    # Replace with your target folder path
    folder = input("Enter the folder path: ")
    list_files_to_csv(folder)
