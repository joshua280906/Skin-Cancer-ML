import os
import shutil
import random
import pandas as pd
from tqdm import tqdm

# =====================================
# PATH CONFIGURATION (CORRECTED)
# =====================================

SOURCE_BASE = os.path.expanduser("~/Desktop/ml_project")
DEST_BASE = os.path.expanduser("~/Desktop/ml_project/ML_PROJECT/data")

TRAIN_DIR = os.path.join(DEST_BASE, "train")
VAL_DIR = os.path.join(DEST_BASE, "val")
TEST_DIR = os.path.join(DEST_BASE, "test")

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

random.seed(42)

print("\n========== STARTING DATASET SPLIT ==========\n")

# =====================================
# FIND METADATA FILE
# =====================================

print("Searching for HAM10000_metadata.csv ...")

METADATA_PATH = None
for root, dirs, files in os.walk(SOURCE_BASE):
    for file in files:
        if file == "HAM10000_metadata.csv":
            METADATA_PATH = os.path.join(root, file)
            break

if METADATA_PATH is None:
    print("ERROR: HAM10000_metadata.csv not found!")
    exit()

print("Found metadata at:", METADATA_PATH)
print("Loading metadata...")

df = pd.read_csv(METADATA_PATH)

print("Metadata loaded successfully!")
print("Total records:", len(df))

# =====================================
# FIND ALL IMAGE FOLDERS
# =====================================

print("\nScanning for image folders...")

IMAGE_FOLDERS = []

for root, dirs, files in os.walk(SOURCE_BASE):
    for file in files:
        if file.endswith(".jpg"):
            IMAGE_FOLDERS.append(root)
            break

IMAGE_FOLDERS = list(set(IMAGE_FOLDERS))

if len(IMAGE_FOLDERS) == 0:
    print("ERROR: No image folders found!")
    exit()

print("Found image folders:")
for folder in IMAGE_FOLDERS:
    print("  ->", folder)

# =====================================
# MAP TO BINARY CLASSES
# =====================================

MALIGNANT_CLASSES = ["mel"]

df["binary_label"] = df["dx"].apply(
    lambda x: "malignant" if x in MALIGNANT_CLASSES else "benign"
)

print("\nClass Distribution:")
print(df["binary_label"].value_counts())

# =====================================
# SPLIT DATA
# =====================================

train_list, val_list, test_list = [], [], []

for label in ["benign", "malignant"]:

    subset = df[df["binary_label"] == label].sample(frac=1, random_state=42)
    total = len(subset)

    train_end = int(TRAIN_SPLIT * total)
    val_end = train_end + int(VAL_SPLIT * total)

    train_list.append(subset[:train_end])
    val_list.append(subset[train_end:val_end])
    test_list.append(subset[val_end:])

train_df = pd.concat(train_list)
val_df = pd.concat(val_list)
test_df = pd.concat(test_list)

print("\nSplit Summary:")
print("Train:", len(train_df))
print("Val  :", len(val_df))
print("Test :", len(test_df))

# =====================================
# FUNCTION TO FIND IMAGE
# =====================================

def find_image(filename):
    for folder in IMAGE_FOLDERS:
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            return path
    return None

# =====================================
# COPY FILES
# =====================================

def process_split(dataframe, split_name, split_dir):

    print(f"\nProcessing {split_name} set...")

    for label in ["benign", "malignant"]:
        subset = dataframe[dataframe["binary_label"] == label]

        print(f"Copying {label} images ({len(subset)})...")

        for _, row in tqdm(subset.iterrows(), total=len(subset)):

            filename = row["image_id"] + ".jpg"
            src_path = find_image(filename)

            if src_path is None:
                continue

            dst_path = os.path.join(split_dir, label, filename)

            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)

    print(f"{split_name} completed.")

# =====================================
# EXECUTE
# =====================================

process_split(train_df, "TRAIN", TRAIN_DIR)
process_split(val_df, "VALIDATION", VAL_DIR)
process_split(test_df, "TEST", TEST_DIR)

print("\n========== DATASET PREPARATION COMPLETE ==========\n")