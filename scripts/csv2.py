import os
import pandas as pd
from datetime import datetime

def get_file_info(directory):
    files_data = []
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        if os.path.isfile(file_path):
            file_size = os.path.getsize(file_path)
            creation_time = datetime.fromtimestamp(os.path.getctime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
            
            files_data.append({
                "File Name": filename,
                "Size (bytes)": file_size,
                "Creation Date": creation_time
            })
    
    return files_data

def save_to_csv(directory, output_file="/file_list.csv"):
    files_data = get_file_info(directory)
    df = pd.DataFrame(files_data)
    df.to_csv(directory + output_file, index=False)
    print(f"File list saved to {output_file}")

if __name__ == "__main__":
    directory = input("Enter the directory path: ")
    save_to_csv(directory)
