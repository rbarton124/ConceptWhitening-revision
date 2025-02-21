import os
import json
import argparse
import shutil
import random
from collections import defaultdict
from pycocotools.coco import COCO
from PIL import Image, ImageDraw

"""
COCO Dataset Processing Script

This script processes the COCO 2017 dataset by:
- Organizing images into `train`, `val`, and `test` directories.
- Grouping images into **supersets (parts mode)** (e.g., "vehicles", "animals", etc.).
- Grouping images into **fine-grained classes (attributes mode)** (e.g., "car", "dog", "chair", etc.).
- Automatically extracting a fraction of test images for validation.

Usage Examples:
---------------
1. Basic usage with default paths:
   ```
   python prepare_COCO_concepts.py --coco_dir /path/to/COCO 
   ```

2. Custom output directories and validation fraction:
   ```
   python prepare_COCO_concepts.py --coco_dir /path/to/COCO \
                                   --output_main main_dataset \
                                   --output_concept concept_dataset \
                                   --val_fraction 0.3
   ```

Arguments:
----------
- `--coco_dir`: Path to COCO dataset directory (should contain `annotations` and `train2017` images).
- `--output_main`: Directory where the main dataset will be stored, relative to coco_dir
- `--output_concept`: Directory where the concept dataset will be stored, relative to coco_dir
- `--val_fraction`: Fraction of test images to move to validation set.

Output Structure:
-----------------
After running the script, the dataset will be organized as follows:

```
main_dataset/
    train/
    val/
    test/

concept_dataset/
    concept_train/
        vehicles/
        animals/
        ...
    concept_test/
        vehicles/
        animals/
        ...
```
"""

# Define COCO Supersets
COCO_SUPERSETS = {
    "vehicles": ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"],
    "traffic_objects": ["traffic light", "fire hydrant", "stop sign", "parking meter", "bench"],
    "animals": ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
    "clothing": ["backpack", "umbrella", "handbag", "tie", "suitcase"],
    "sports": ["frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket"],
    "kitchen": ["bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl"],
    "food": ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"],
    "furniture": ["chair", "couch", "potted plant", "bed", "dining table", "toilet"],
    "electronics": ["TV", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator"],
    "miscellaneous": ["book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
}

# Reverse mapping for quick lookup
CATEGORY_TO_SUPERSET = {}
for supercat, classes in COCO_SUPERSETS.items():
    for cls in classes:
        CATEGORY_TO_SUPERSET[cls] = supercat

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate Main and Concept Datasets from COCO 2017")
    parser.add_argument("--coco_dir", required=True, help="Path to COCO dataset (where annotations and images are stored)")
    parser.add_argument("--output_main", default="main_data", help="Folder to store main dataset")
    parser.add_argument("--output_concept", default="concept_data", help="Folder to store concept dataset")
    parser.add_argument("--val_fraction", type=float, default=0.5, help="Fraction of test images that become val")
    return parser.parse_args()

def load_coco_annotations(coco_dir):
    """Loads COCO dataset annotations."""
    annotation_file = os.path.join(coco_dir, "annotations", "instances_train2017.json")
    return COCO(annotation_file)

def get_category_mapping(coco):
    """Returns a mapping from category IDs to category names."""
    categories = coco.loadCats(coco.getCatIds())
    return {cat["id"]: cat["name"] for cat in categories}

def create_directory_structure(base_dir, categories):
    """Creates the necessary directory structure for the dataset."""
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(base_dir, split), exist_ok=True)
    for supercat in COCO_SUPERSETS.keys():
        os.makedirs(os.path.join(base_dir, "concept_train", supercat), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "concept_test", supercat), exist_ok=True)
    for category in categories.values():
        os.makedirs(os.path.join(base_dir, "concept_train", category), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "concept_test", category), exist_ok=True)

def copy_and_organize_images(image_data, split_set, output_main, output_concept, coco_dir, split_name):
    """Copies images into train/val/test directories and organizes concept datasets."""
    image_dir = os.path.join(coco_dir, "train2017")  # or "val2017" if needed

    for img_id in split_set:
        if img_id not in image_data:
            continue

        img_info, cat_name, supercat_name = image_data[img_id][0]
        img_filename = img_info["file_name"]
        src_path = os.path.join(image_dir, img_filename)

        # Ensure target directory exists
        target_dir = os.path.join(output_main, split_name)
        os.makedirs(target_dir, exist_ok=True)

        dst_path = os.path.join(target_dir, img_filename)
        shutil.copy(src_path, dst_path)  # Copy to main dataset

        # Ensure concept dataset directories exist
        concept_train_dir = os.path.join(output_concept, "concept_train", supercat_name)
        concept_test_dir = os.path.join(output_concept, "concept_test", supercat_name)
        os.makedirs(concept_train_dir, exist_ok=True)
        os.makedirs(concept_test_dir, exist_ok=True)

        # Copy to concept dataset (superset and original category)
        sup_dst = os.path.join(concept_train_dir if split_name == "train" else concept_test_dir, img_filename)
        shutil.copy(src_path, sup_dst)

        cat_train_dir = os.path.join(output_concept, "concept_train", cat_name)
        cat_test_dir = os.path.join(output_concept, "concept_test", cat_name)
        os.makedirs(cat_train_dir, exist_ok=True)
        os.makedirs(cat_test_dir, exist_ok=True)

        cat_dst = os.path.join(cat_train_dir if split_name == "train" else cat_test_dir, img_filename)
        shutil.copy(src_path, cat_dst)

def split_dataset(image_ids, val_fraction):
    """Splits dataset into validation and test sets."""
    random.shuffle(image_ids)
    val_size = int(len(image_ids) * val_fraction)
    return image_ids[:val_size], image_ids[val_size:]

def process_images(args, coco, category_mapping):
    """Processes images and assigns them to train, val, and test sets."""
    image_data = defaultdict(list)
    
    for ann in coco.loadAnns(coco.getAnnIds()):
        img_info = coco.loadImgs(ann["image_id"])[0]
        cat_name = category_mapping[ann["category_id"]]
        supercat_name = CATEGORY_TO_SUPERSET.get(cat_name, "miscellaneous")
        
        image_data[ann["image_id"]].append((img_info, cat_name, supercat_name))
    
    train_images = [img["id"] for img in coco.loadImgs(coco.getImgIds())]
    val_images, test_images = split_dataset(train_images, args.val_fraction)
    
    return image_data, train_images, val_images, test_images

def main():
    """Main function that orchestrates the dataset organization."""
    args = parse_args()
    args.output_concept = os.path.join(args.coco_dir, args.output_concept)
    args.output_main = os.path.join(args.coco_dir, args.output_main)

    print("[Info] Loading COCO dataset...")
    coco = load_coco_annotations(args.coco_dir)
    category_mapping = get_category_mapping(coco)
    
    print("[Info] Creating directory structure...")
    create_directory_structure(args.output_concept, category_mapping)
    os.makedirs(args.output_main, exist_ok=True)
    
    print("[Info] Processing images...")
    image_data, train_images, val_images, test_images = process_images(args, coco, category_mapping)
    
    print("[Info] Copying images to train/val/test...")
    copy_and_organize_images(image_data, train_images, args.output_main, args.output_concept, args.coco_dir, "train")
    copy_and_organize_images(image_data, val_images, args.output_main, args.output_concept, args.coco_dir, "val")
    copy_and_organize_images(image_data, test_images, args.output_main, args.output_concept, args.coco_dir, "test")
    
    print("[Done] COCO dataset processing complete.")

if __name__ == "__main__":
    main()
