import argparse
import json
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Pool, cpu_count

def parse_arguments():
    parser = argparse.ArgumentParser(description='Crop COCO images by bounding boxes (parallelized).')
    parser.add_argument('-coco-path', type=str, default='data/coco',
                        help='Path to COCO directory (containing train2017, val2017, annotations, etc.)')
    parser.add_argument('-concept-path', type=str, default='data_256',
                        help='Output path for concept dataset')
    parser.add_argument('--min-size', type=int, default=30,
                        help='Minimum width/height of a bounding box to consider')
    parser.add_argument('--workers', type=int, default=0,
                        help='Number of processes to use (default: 0 means use all CPUs)')
    return parser.parse_args()

def build_image_annotation_map(annotations, imageid2filename, min_size, label2conceptname):
    """
    Build a dictionary mapping:
      image_id -> list of (xmin, ymin, xmax, ymax, concept_name, ann_id)
    Ensures we only open each image once, then crop multiple bounding boxes if needed.
    """
    from collections import defaultdict
    image_map = defaultdict(list)

    for ann in annotations:
        bbox = ann['bbox']  # [x, y, w, h]
        w, h = bbox[2], bbox[3]
        # Filter out bounding boxes that are too small
        if w < min_size or h < min_size:
            continue

        # Convert to (xmin, ymin, xmax, ymax)
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[0] + w
        ymax = bbox[1] + h

        image_id = ann['image_id']
        concept_id = ann['category_id']
        concept_name = label2conceptname[concept_id]
        ann_id = ann['id']

        image_map[image_id].append((xmin, ymin, xmax, ymax, concept_name, ann_id))

    return image_map

def ensure_concept_dirs(label2conceptname, output_dir, is_train=False):
    """
    Ensure each concept directory exists.
      - If is_train=True => double-path structure: e.g. output_dir/concept/concept/
      - Else => single-path structure: e.g. output_dir/concept/
    """
    for _, concept in label2conceptname.items():
        if is_train:
            # e.g. concept_train/car/car/
            (output_dir / concept / concept).mkdir(parents=True, exist_ok=True)
        else:
            # e.g. concept_test/car/
            (output_dir / concept).mkdir(parents=True, exist_ok=True)

def crop_worker(job):
    """
    Worker function that processes a single image.
    job = (image_id, bboxes, image_path, is_train, output_dir)
    bboxes = list of (xmin, ymin, xmax, ymax, concept_name, ann_id)
    Returns a dictionary: concept_name -> # of crops produced
    """
    image_id, bboxes, image_path, is_train, output_dir = job
    local_counts = defaultdict(int)

    # Attempt to open image
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        # Could not open or some error => skip
        return local_counts

    # For each bounding box
    for (xmin, ymin, xmax, ymax, concept_name, ann_id) in bboxes:
        # Crop
        cropped = img.crop((xmin, ymin, xmax, ymax))

        # Decide subfolder
        if is_train:
            # e.g. concept_train/car/car/ann_id.jpg
            out_subdir = output_dir / concept_name / concept_name
        else:
            # e.g. concept_test/car/ann_id.jpg
            out_subdir = output_dir / concept_name

        out_file = out_subdir / f"{ann_id}.jpg"
        try:
            cropped.save(out_file)
            local_counts[concept_name] += 1
        except Exception:
            # Possibly disk write error, skip
            pass

    return local_counts

def crop_images_bbox(image_dir, output_dir, anno_path,
                     min_size=30, is_train=False, n_workers=0):
    """
    Crop bounding boxes from images in image_dir based on annotation file.
    - If is_train=True => double path: concept_train/concept/concept/*.jpg
    - Else => single path: concept_test/concept/*.jpg
    - n_workers: number of parallel processes (0 => use cpu_count()).
    """
    with open(anno_path, 'r') as f:
        anno_data = json.load(f)

    # 1) Build image_id -> filename mapping
    imageid2filename = {img['id']: img['file_name'] for img in anno_data['images']}

    # 2) Build category_id -> concept_name mapping
    label2conceptname = {cat['id']: cat['name'].replace(' ', '_')
                         for cat in anno_data['categories']}

    # 3) Ensure concept directories exist
    ensure_concept_dirs(label2conceptname, output_dir, is_train=is_train)

    # 4) Build dictionary mapping image_id -> bounding boxes
    image_map = build_image_annotation_map(
        anno_data['annotations'], imageid2filename,
        min_size=min_size, label2conceptname=label2conceptname
    )

    # Gather all tasks
    jobs = []
    for image_id, bboxes in image_map.items():
        image_path = image_dir / imageid2filename[image_id]
        if not image_path.is_file():
            continue
        jobs.append((image_id, bboxes, image_path, is_train, output_dir))

    # Prepare for parallel
    concept_counter = defaultdict(int)
    n_workers = n_workers if n_workers > 0 else cpu_count()

    print(f"\nCropping from '{image_dir.name}' with {n_workers} worker(s)...")

    # Use multiprocessing.Pool to process images in parallel
    with Pool(processes=n_workers) as pool:
        # imap_unordered => returns results as they come
        for local_counts in tqdm(pool.imap_unordered(crop_worker, jobs, chunksize=50),
                                 total=len(jobs), desc=f"Cropping {image_dir.name}", unit="img"):
            # Merge results
            for cpt_name, cnt in local_counts.items():
                concept_counter[cpt_name] += cnt

    # Print final stats
    print(f"\nFinished cropping from {image_dir.name}.")
    print(f"Total images with bounding boxes: {len(jobs)}")
    for cpt_name, count in sorted(concept_counter.items()):
        print(f"  Concept '{cpt_name}': {count} crops")

def main():
    args = parse_arguments()
    coco_path = Path(args.coco_path)
    concept_path = Path(args.concept_path)
    concept_path.mkdir(parents=True, exist_ok=True)

    # We'll create concept_train/ and concept_test/ inside concept_path
    concept_train_dir = concept_path / 'concept_train'
    concept_val_dir   = concept_path / 'concept_test'
    concept_train_dir.mkdir(parents=True, exist_ok=True)
    concept_val_dir.mkdir(parents=True, exist_ok=True)

    train_dir = coco_path / 'train2017'
    val_dir   = coco_path / 'val2017'
    train_anno_path = coco_path / 'annotations' / 'instances_train2017.json'
    val_anno_path   = coco_path / 'annotations' / 'instances_val2017.json'

    # 1) Crop for VAL set => single-path
    crop_images_bbox(
        image_dir=val_dir,
        output_dir=concept_val_dir,
        anno_path=val_anno_path,
        min_size=args.min_size,
        is_train=False,   # concept_test => single path
        n_workers=args.workers
    )

    # 2) Crop for TRAIN set => double-path
    crop_images_bbox(
        image_dir=train_dir,
        output_dir=concept_train_dir,
        anno_path=train_anno_path,
        min_size=args.min_size,
        is_train=True,    # concept_train => double path
        n_workers=args.workers
    )

    print("\nAll done!")

if __name__ == '__main__':
    main()
