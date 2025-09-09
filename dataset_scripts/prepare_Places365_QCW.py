import os
import shutil
import argparse
import random
from tqdm import tqdm

# Constants
TRAIN_SAMPLES_PER_CLASS = 150
VAL_SAMPLES_PER_CLASS = 20
TEST_SAMPLES_PER_CLASS = 40

def log(msg, level, verbose):
    """Print message if verbosity level is high enough."""
    if verbose >= level:
        print(msg)

def read_class_list(filepath):
    """Read Places365 categories file."""
    categories = []
    with open(filepath, 'r') as f:
        for line in f:
            category = line.strip().split(' ')[0][3:]  # Remove leading /a/
            category = category.replace('/', ':')  # Replace slashes with colons
            categories.append(category)
    return categories

def read_val_file(val_file):
    """Read Places365 validation file."""
    img_to_class = {}
    with open(val_file, 'r') as f:
        for line in f:
            img_name, class_idx = line.strip().split()
            img_to_class[img_name] = int(class_idx)
    return img_to_class

def get_category_to_index_map(categories_file):
    """Create mapping from indices to category names."""
    category_to_idx = {}
    with open(categories_file, 'r') as f:
        for line in f:
            category_raw, category_idx = line.strip().split(' ')
            category_idx = int(category_idx)
            category = category_raw[3:]
            category = category.replace('/', ':')
            category_to_idx[category] = category_idx
    return {v: k for k, v in category_to_idx.items()}

def setup_directory(dir_path, categories, verbose):
    """Set up directory structure for dataset split."""
    if os.path.exists(dir_path):
        log(f"Clearing output directory: {dir_path}", 1, verbose)
        shutil.rmtree(dir_path)

    log(f"Creating output directory: {dir_path}", 1, verbose)
    os.makedirs(dir_path, exist_ok=True)

    # Create category directories
    for category in categories:
        os.makedirs(os.path.join(dir_path, category), exist_ok=True)

def copy_images(src_path, dst_path, images, verbose):
    """Copy selected images from source to destination."""
    for img in images:
        src_img_path = os.path.join(src_path, img)
        dst_img_path = os.path.join(dst_path, img)
        shutil.copy(src_img_path, dst_img_path)
        log(f"Copied: {src_img_path} -> {dst_img_path}", 2, verbose)

def organize_train_split(src_dir, dst_dir, categories, samples_per_class, verbose):
    """Organize training split of the dataset."""
    setup_directory(dst_dir, categories, verbose)

    for category in tqdm(categories, disable=verbose < 1):
        src_category_dir = os.path.join(src_dir, category[0], *(category.split(':')))
        if not os.path.exists(src_category_dir):
            log(f"Warning: Category directory not found: {src_category_dir}", 1, verbose)
            continue

        # Get and sample images
        images = [f for f in os.listdir(src_category_dir) 
                 if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(images) < samples_per_class:
            log(f"Warning: Not enough images for {category}. Found {len(images)}, needed {samples_per_class}", 1, verbose)
            selected_images = images
        else:
            selected_images = random.sample(images, samples_per_class)

        # Copy selected images
        dst_category_dir = os.path.join(dst_dir, category)
        copy_images(src_category_dir, dst_category_dir, selected_images, verbose)

def organize_val_split(val_dir, dst_dir, categories, samples_per_class, val_txt_file, categories_file, verbose):
    """Organize validation/test split using places365_val.txt"""
    setup_directory(dst_dir, categories, verbose)

    # Load validation data
    img_to_class = read_val_file(val_txt_file)
    idx_to_category = get_category_to_index_map(categories_file)

    # Group images by category
    category_images = {}
    for img_name, class_idx in img_to_class.items():
        category = idx_to_category[class_idx]
        category_images.setdefault(category, []).append(img_name)

    # Process each category
    for category in tqdm(categories, disable=verbose < 1):
        if category not in category_images:
            log(f"Warning: No images found for category {category}", 1, verbose)
            continue

        available_images = category_images[category]
        if len(available_images) < samples_per_class:
            log(f"Warning: Not enough images for {category}. Found {len(available_images)}, needed {samples_per_class}", 1, verbose)
            selected_images = available_images
        else:
            selected_images = random.sample(available_images, samples_per_class)

        # Copy selected images
        dst_category_dir = os.path.join(dst_dir, category)
        copy_images(val_dir, dst_category_dir, selected_images, verbose)

def organize_dataset(train_dir, val_dir, categories_file, val_txt_file, target_root, dataset_name, verbose):
    """Main function to organize Places365 dataset."""
    log("Reading categories...", 1, verbose)
    categories = read_class_list(categories_file)

    dataset_root = os.path.join(target_root, dataset_name)
    main_root = os.path.join(dataset_root, "main_dataset")

    # Create train split
    log("\nCreating train split...", 1, verbose)
    train_dir_out = os.path.join(main_root, "train")
    organize_train_split(train_dir, train_dir_out, categories, TRAIN_SAMPLES_PER_CLASS, verbose)

    # Create validation split
    log("\nCreating validation split...", 1, verbose)
    val_dir_out = os.path.join(main_root, "val")
    organize_val_split(val_dir, val_dir_out, categories, VAL_SAMPLES_PER_CLASS, 
                       val_txt_file, categories_file, verbose)

    # Create test split
    log("\nCreating test split...", 1, verbose)
    test_dir_out = os.path.join(main_root, "test")
    organize_val_split(val_dir, test_dir_out, categories, TEST_SAMPLES_PER_CLASS,
                       val_txt_file, categories_file, verbose)

    log("\nDone organizing Places365 dataset.", 1, verbose)

def main():
    parser = argparse.ArgumentParser(description="Organize Places365 images into main train/val/test folders.")
    parser.add_argument("--train_dir", type=str, required=True, 
                        help="Path to Places365 train directory")
    parser.add_argument("--val_dir", type=str, required=True, 
                        help="Path to Places365 validation directory")
    parser.add_argument("--categories", type=str, required=True,
                        help="Path to categories list file (categories_places365.txt)")
    parser.add_argument("--val_txt", type=str, required=True,
                        help="Path to places365_val.txt file")
    parser.add_argument("--target_root", type=str, required=True,
                        help="Directory where output will be stored")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Top-level name for the organized dataset folder")
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2],
                        help="Verbosity level: 0=silent, 1=progress (default), 2=all messages")

    args = parser.parse_args()
    organize_dataset(args.train_dir, args.val_dir, args.categories, args.val_txt,
                     args.target_root, args.dataset_name, args.verbose)

if __name__ == "__main__":
    main()