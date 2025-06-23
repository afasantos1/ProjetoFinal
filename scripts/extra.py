import pandas as pd

# Load the CSV file
df = pd.read_csv('output_images.csv')

# Subtract 1 from the 'class' column (adjust column name if different)
df['id'] = df['id'] - 1

# Save the modified DataFrame to a new CSV file (optional)
df.to_csv('output_images.csv', index=False)

# Or if you're using it directly, you can now use df['class'] with 0-indexed labels
