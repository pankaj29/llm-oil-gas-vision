#%%
import os
import pandas as pd

def list_png_files(directory_path):
    """
    List all PNG files in the specified directory and return a DataFrame.
    
    Parameters:
        directory_path (str): Path to the folder containing PNG files.

    Returns:
        pd.DataFrame: DataFrame containing file names and their full paths.
    """
    # Get all PNG files in the directory
    png_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.png')]

    # Create a DataFrame
    df = pd.DataFrame({"Image_Path": png_files, "Full Path": [os.path.join(directory_path, f) for f in png_files]})
    df["File Name"]=df["Image_Path"].apply(lambda x:"images/"+x)
    del df["Full Path"]
    return df

# Example usage (Update the path as per your directory)
directory_path = r"C:\Users\panka\OneDrive\Documents\GitHub\llm-oil-gas-vision\Data\images"
png_files_df = list_png_files(directory_path)
output_path=r"C:\Users\panka\OneDrive\Documents\GitHub\llm-oil-gas-vision\Data\list_all_images.csv"
png_files_df.to_csv(output_path,index=-False)

#%%
# Update the column name for filtering
# Load the datasets
test_df = pd.read_csv(r"C:\Users\panka\OneDrive\Documents\GitHub\llm-oil-gas-vision\Data\test.csv")
train_df = pd.read_csv(r"C:\Users\panka\OneDrive\Documents\GitHub\llm-oil-gas-vision\Data\train.csv")
val_df = pd.read_csv(r"C:\Users\panka\OneDrive\Documents\GitHub\llm-oil-gas-vision\Data\val.csv")
list_all_images_df = pd.read_csv(r"C:\Users\panka\OneDrive\Documents\GitHub\llm-oil-gas-vision\Data\list_all_images.csv")

# Print the lengths before filtering
before_lengths = {
    "Test Before": len(test_df),
    "Train Before": len(train_df),
    "Validation Before": len(val_df)
}

print(before_lengths)
# Extract the list of valid image paths from list_all_images.csv
valid_images = set(list_all_images_df["Image_Path"].astype(str))

# Filter datasets to keep only rows where 'Image_Path' exists in valid_images
test_filtered = test_df[test_df["Image_Path"].astype(str).isin(valid_images)]
train_filtered = train_df[train_df["Image_Path"].astype(str).isin(valid_images)]
val_filtered = val_df[val_df["Image_Path"].astype(str).isin(valid_images)]


after_lengths = {
    "Test After": len(test_filtered),
    "Train After": len(train_filtered),
    "Validation After": len(val_filtered)
}

print(after_lengths)
#%%
# Save the filtered datasets
test_filtered_path = r"C:\Users\panka\OneDrive\Documents\GitHub\llm-oil-gas-vision\Data\test.csv"
train_filtered_path = r"C:\Users\panka\OneDrive\Documents\GitHub\llm-oil-gas-vision\Data\train.csv"
val_filtered_path = r"C:\Users\panka\OneDrive\Documents\GitHub\llm-oil-gas-vision\Data\val.csv"


test_filtered.to_csv(test_filtered_path, index=False)
train_filtered.to_csv(train_filtered_path, index=False)
val_filtered.to_csv(val_filtered_path, index=False)

#%%
