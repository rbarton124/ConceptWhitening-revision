import os
import shutil
import argparse
import random
import json
from pycocotools.coco import COCO
from tqdm import tqdm


TOTAL_SAMPLES = 30000
TRAIN_SAMPLES = int(0.7 * TOTAL_SAMPLES)
VAL_SAMPLES = int(0.1 * TOTAL_SAMPLES)
TEST_SAMPLES = int(0.2 * TOTAL_SAMPLES)


def log(msg, level, verbose):
    if verbose >= level:
        print(msg)


def split_dataset(image_ids, seed=42):
    random.seed(seed)
    random.shuffle(image_ids)
    n = len(image_ids)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    return image_ids[:train_end], image_ids[train_end:val_end], image_ids[val_end:]


def organize_by_category(
    coco, bboxes, image_ids, image_dir, output_dir, concept_dir, cat_id_to_name, cat_id_to_supercat, img_id_to_cats, verbose=1, group_by_supercat=False
):
    if os.path.exists(output_dir):
        log(f"Clearing output directory: {output_dir}", 1, verbose)
        shutil.rmtree(output_dir)

    log(f"Creating output directory: {output_dir}", 1, verbose)
    os.makedirs(output_dir, exist_ok=True)
    log(f"Output directory created.", 1, verbose)

    iterable = tqdm(image_ids) if verbose >= 1 else image_ids
    for img_id in iterable:
        img_info = coco.loadImgs(img_id)[0]
        filename = img_info['file_name']
        src_path = os.path.join(image_dir, filename)

        if not os.path.exists(src_path):
            log(f"Missing image: {src_path}", 2, verbose)
            continue

        if img_id not in img_id_to_cats:
            continue

        for cat_id in img_id_to_cats[img_id]:
            category = cat_id_to_name[cat_id]

            if group_by_supercat:
                supercategory = cat_id_to_supercat[cat_id]
                dst_dir = os.path.join(output_dir, supercategory, category)
            else:
                dst_dir = os.path.join(output_dir, category)

            os.makedirs(dst_dir, exist_ok=True)
            dst_path = os.path.join(dst_dir, filename)

            if not os.path.exists(dst_path):
                shutil.copy(src_path, dst_path)
                log(f"Copied: {src_path} -> {dst_path}", 2, verbose)

        if group_by_supercat:
            anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
            for ann in anns:
                cat_id = ann['category_id']
                category = cat_id_to_name[cat_id]
                supercategory = cat_id_to_supercat[cat_id]

                dst_dir = os.path.join(output_dir, supercategory, category)
                dst_path = os.path.join(dst_dir, filename)
                rel_path = os.path.relpath(dst_path, concept_dir)
                x, y, width, height = ann['bbox']
                bboxes[rel_path] = [x, y, x + width, y + height]


def organize_coco(json_path, image_dir, target_root, dataset_name, verbose=1):
    log(f"Loading COCO annotations from {json_path}...", 1, verbose)
    coco = COCO(json_path)

    bboxes = {}

    cat_id_to_name = {}
    cat_id_to_supercat = {}
    for cat in coco.loadCats(coco.getCatIds()):
        cat_id_to_name[cat['id']] = cat['name']
        cat_id_to_supercat[cat['id']] = cat['supercategory']

    log("Mapping image IDs to category IDs...", 1, verbose)
    img_id_to_cats = {}
    for ann in coco.loadAnns(coco.getAnnIds()):
        img_id = ann['image_id']
        cat_id = ann['category_id']
        img_id_to_cats.setdefault(img_id, set()).add(cat_id)

    all_img_ids = list(img_id_to_cats.keys())

    sampled_img_ids = random.sample(all_img_ids, TOTAL_SAMPLES)
    train_ids = sampled_img_ids[:TRAIN_SAMPLES]
    val_ids = sampled_img_ids[TRAIN_SAMPLES:TRAIN_SAMPLES + VAL_SAMPLES]
    test_ids = sampled_img_ids[TRAIN_SAMPLES + VAL_SAMPLES:TOTAL_SAMPLES]

    dataset_root = os.path.join(target_root, dataset_name)
    main_root = os.path.join(dataset_root, "main_dataset")
    concept_root = os.path.join(dataset_root, "concept_dataset")

    log("\nCreating train split in 'main_dataset'...", 1, verbose)
    organize_by_category(coco, bboxes, train_ids, image_dir, os.path.join(main_root, "train"), concept_root,
                         cat_id_to_name, cat_id_to_supercat, img_id_to_cats, verbose, group_by_supercat=False)
    log("\nCreating validation split in 'main_dataset'...", 1, verbose)
    organize_by_category(coco, bboxes, val_ids, image_dir, os.path.join(main_root, "val"), concept_root,
                         cat_id_to_name, cat_id_to_supercat, img_id_to_cats, verbose, group_by_supercat=False)
    log("\nCreating test split in 'main_dataset'...", 1, verbose)
    organize_by_category(coco, bboxes, test_ids, image_dir, os.path.join(main_root, "test"), concept_root,
                         cat_id_to_name, cat_id_to_supercat, img_id_to_cats, verbose, group_by_supercat=False)

    log("\nCreating train concept datasets...", 1, verbose)
    organize_by_category(coco, bboxes, train_ids, image_dir, os.path.join(concept_root, "concept_train"), concept_root,
                         cat_id_to_name, cat_id_to_supercat, img_id_to_cats, verbose, group_by_supercat=True)
    log("\nCreating validation concept datasets...", 1, verbose)
    organize_by_category(coco, bboxes, val_ids, image_dir, os.path.join(concept_root, "concept_val"), concept_root,
                         cat_id_to_name, cat_id_to_supercat, img_id_to_cats, verbose, group_by_supercat=True)
    
    log("\nExtracting bounding boxes for the full main dataset...", 1, verbose)

    output_path = os.path.join(concept_root, "bboxes.json")
    with open(output_path, 'w') as f:
        json.dump(bboxes, f, indent=2)
    log(f"Saved bboxes to {output_path}", 1, verbose)

    log("\nDone organizing COCO dataset.", 1, verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize COCO images into main/concept train/val/test folders.")
    parser.add_argument("--json", type=str, required=True, help="Path to COCO annotation file (e.g., instances_val2017.json)")
    parser.add_argument("--img_dir", type=str, required=True, help="Path to directory with COCO images (e.g., val2017/)")
    parser.add_argument("--target_root", type=str, required=True, help="Directory where output will be stored")
    parser.add_argument("--dataset_name", type=str, required=True, help="Top-level name for the organized dataset folder")
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2], help="Verbosity level: 0=silent, 1=progress (default), 2=all messages")

    args = parser.parse_args()
    organize_coco(args.json, args.img_dir, args.target_root, args.dataset_name, args.verbose)
