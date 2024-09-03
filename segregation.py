import os
import shutil
import random

# Define paths
source_dir = r'C:\Users\rathn\OneDrive\Documents\dataset-resized\dataset-resized\glass'
train_dir = r'C:\Users\rathn\OneDrive\Documents\dataset-resized\dataset-resized\train\glass'
val_dir = r'C:\Users\rathn\OneDrive\Documents\dataset-resized\dataset-resized\validation\glass'

# Create directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# List all files in the source directory
files = os.listdir(source_dir)

# Shuffle the list of files
random.shuffle(files)

# Define the split ratio
split_ratio = 0.8  # 80% for training, 20% for validation

# Calculate split index
split_index = int(len(files) * split_ratio)

# Split files into training and validation
train_files = files[:split_index]
val_files = files[split_index:]

# Move files to training directory
for file in train_files:
    shutil.move(os.path.join(source_dir, file), os.path.join(train_dir, file))

# Move files to validation directory
for file in val_files:
    shutil.move(os.path.join(source_dir, file), os.path.join(val_dir, file))

print(f"Moved {len(train_files)} files to training directory.")
print(f"Moved {len(val_files)} files to validation directory.")
