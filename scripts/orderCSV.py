import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('output_images.csv')

# Extract the header separately
header = df.columns

# Sort the DataFrame by the first column (index 0)
df_sorted = df.sort_values(by=df.columns[0])

# Save the sorted DataFrame to a new CSV file, including the header
df_sorted.to_csv('image_data.csv', index=False, header=True)
