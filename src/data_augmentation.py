#%% Import Libraries#%% Import Libraries
#%% Import Libraries
#%% Import Libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
import pandas as pd
import numpy as np
import os
from collections import Counter

# Paths to datasets
train_csv = 'Data/train.csv'
val_csv = 'Data/val.csv'
test_csv = 'Data/test.csv'
image_dir = 'Data'

# Output directory for augmented images
augmented_dir = 'Data/augmented_images/images'
os.makedirs(augmented_dir, exist_ok=True)

# Load datasets
def load_datasets(train_csv, val_csv, test_csv, image_dir):
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    for df in [train_df, val_df, test_df]:
        df['Target'] = df['Target'].astype(str)
        df['Image_Path'] = df['Image_Path'].apply(lambda x: os.path.join(image_dir, x))

    print(f"Train class distribution:\n{train_df['Target'].value_counts()}")
    print(f"Validation class distribution:\n{val_df['Target'].value_counts()}")
    print(f"Test class distribution:\n{test_df['Target'].value_counts()}")

    return train_df, val_df, test_df

train_df, val_df, test_df = load_datasets(train_csv, val_csv, test_csv, image_dir)

# Augmentation settings - Only flipping horizontally or vertically
augmentor = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True
)

# Function to augment and save images
def augment_and_save(df, label, target_count, save_dir):
    current_images = df[df['Target'] == label]
    current_count = len(current_images)
    needed_count = target_count - current_count

    print(f"Augmenting {needed_count} images for class {label}")
    
    i = 0
    for idx, row in current_images.iterrows():
        if i >= needed_count:
            break
        img = load_img(row['Image_Path'])
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        for batch in augmentor.flow(img_array, batch_size=1):
            save_path = os.path.join(save_dir, f"aug_{label}_{i}_{os.path.basename(row['Image_Path'])}")
            array_to_img(batch[0]).save(save_path)
            i += 1
            if i >= needed_count:
                break

# Balance classes for train, val, and test
max_class_count = 2000  # Limiting to 2000 images per class

# Augmenting train data
for label in train_df['Target'].unique():
    augment_and_save(train_df, label, max_class_count, augmented_dir)

# Augmenting validation data
for label in val_df['Target'].unique():
    augment_and_save(val_df, label, max_class_count, augmented_dir)

# Augmenting test data
for label in test_df['Target'].unique():
    augment_and_save(test_df, label, max_class_count, augmented_dir)

# Function to create balanced dataframe from original and augmented images
def create_balanced_dataframe(original_df, augmented_dir):
    balanced_data = []
    for label in original_df['Target'].unique():
        augmented_images = [os.path.join(augmented_dir, img) for img in os.listdir(augmented_dir) if f"aug_{label}_" in img]
        
        original_images = original_df[original_df['Target'] == label]['Image_Path'].tolist()
        total_images = original_images + augmented_images[:max_class_count - len(original_images)]

        # Ensure only 2000 images are saved per class
        total_images = total_images[:max_class_count]

        for img_path in total_images:
            balanced_data.append({'Image_Path': img_path, 'Target': label})
    
    return pd.DataFrame(balanced_data)

# Create balanced datasets
train_balanced_df = create_balanced_dataframe(train_df, augmented_dir)
val_balanced_df = create_balanced_dataframe(val_df, augmented_dir)
test_balanced_df = create_balanced_dataframe(test_df, augmented_dir)

# Save balanced datasets
train_balanced_df.to_csv('Data/balanced_train.csv', index=False)
val_balanced_df.to_csv('Data/balanced_val.csv', index=False)
test_balanced_df.to_csv('Data/balanced_test.csv', index=False)

print("Balanced datasets with 2000 images per class created and saved.")
#%%

#%%