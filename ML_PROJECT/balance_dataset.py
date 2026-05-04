import os
import random
import shutil

BASE_DIR = os.path.expanduser("~/Desktop/ml_project/ML_PROJECT/data/train")

benign_dir = os.path.join(BASE_DIR, "benign")
malignant_dir = os.path.join(BASE_DIR, "malignant")

benign_files = os.listdir(benign_dir)
malignant_files = os.listdir(malignant_dir)

print("Before Balancing:")
print("Benign:", len(benign_files))
print("Malignant:", len(malignant_files))

# Find smaller class size
min_count = min(len(benign_files), len(malignant_files))

random.seed(42)

# Randomly select files to KEEP
benign_keep = set(random.sample(benign_files, min_count))
malignant_keep = set(random.sample(malignant_files, min_count))

# Remove extra benign
for file in benign_files:
    if file not in benign_keep:
        os.remove(os.path.join(benign_dir, file))

# Remove extra malignant
for file in malignant_files:
    if file not in malignant_keep:
        os.remove(os.path.join(malignant_dir, file))

print("\nAfter Balancing:")
print("Benign:", len(os.listdir(benign_dir)))
print("Malignant:", len(os.listdir(malignant_dir)))

print("\nTrain dataset successfully balanced.")