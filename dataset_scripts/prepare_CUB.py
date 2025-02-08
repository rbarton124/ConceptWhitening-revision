#!/usr/bin/env python3
"""
Unified Script to Generate Main and Concept Datasets from CUB,
supporting both parts and attributes as concepts, and allowing
the user to either draw bounding boxes or crop the image.

It also now creates a "val" set by moving a fraction of the test images
(ensuring an even distribution across classes).

Usage Examples:
---------------
1) Parts Mode, Cropping a 224x224 box around each part location:
   python cub_dataset_setup.py \
     --cub_dir /path/to/CUB_200_2011 \
     --output_main main_data \
     --output_concept concept_data \
     --concepts wing,beak,tail \
     --mode parts \
     --draw_or_crop crop \
     --crop_size 224

2) Attributes Mode, Drawing bounding boxes from the official bounding_box.txt:
   python cub_dataset_setup.py \
     --cub_dir /path/to/CUB_200_2011 \
     --output_main main_data \
     --output_concept concept_data \
     --concepts has_wing_color::blue,has_bill_shape::curved \
     --mode attributes \
     --draw_or_crop draw

Explanation:
------------
- `--mode` sets whether the `--concepts` refer to part names or attribute names.
- `--draw_or_crop` can be:
    * "draw" = Draw a rectangle on the full image
    * "crop" = Physically crop the image to that rectangle
    * "none" = Do not alter the image, just copy it
- `--crop_size` determines the *size of the square region* around a part location if you are in
  "parts" mode with cropping. If using the dataset's official bounding boxes for attributes,
  we can either draw or crop them directly.

Additionally, we create a 'val' set out of some fraction of the test images,
evenly distributed across classes.
"""

import os
import argparse
import pandas as pd
import shutil
from PIL import Image, ImageDraw
import random
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Main and Concept Datasets from CUB")
    parser.add_argument("--cub_dir", required=True,
                        help="Path to the unzipped CUB-200-2011 dataset")
    parser.add_argument("--output_main", default="main_data",
                        help="Folder to store main dataset")
    parser.add_argument("--output_concept", default="concept_data",
                        help="Folder to store concept dataset")
    parser.add_argument("--concepts", required=True,
                        help="Comma-separated list of part or attribute names to extract")
    parser.add_argument("--mode", choices=["parts", "attributes"], default="parts",
                        help="Whether to interpret the `--concepts` as parts or attributes.")
    parser.add_argument("--crop_size", type=int, default=224,
                        help="Size of the region around the part location, if using `crop`.")
    parser.add_argument("--draw_or_crop", choices=["draw", "crop", "none"], default="draw",
                        help="How to handle bounding boxes: 'draw' = draw, 'crop' = physically crop, 'none' = do nothing.")
    parser.add_argument("--val_fraction", type=float, default=0.5,
                        help="Fraction of each class's test images that become val.")
    return parser.parse_args()

########################################
# HELPER FUNCTIONS FOR LOADING DATA
########################################

def load_parts_file(parts_txt):
    """
    Custom loader for parts/parts.txt
    Format: <part_id> <part_name...>
    """
    rows = []
    with open(parts_txt, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 2:
                continue
            part_id = int(tokens[0])
            part_name = " ".join(tokens[1:])
            rows.append([part_id, part_name])
    return pd.DataFrame(rows, columns=["part_id", "part_name"])

def load_annotations(cub_dir):
    """
    Load base annotation files: images.txt, train_test_split.txt, image_class_labels.txt, bounding_boxes.txt.
    Returns a DataFrame with columns [image_id, path, is_train, class, x, y, width, height].
    """
    images_path = os.path.join(cub_dir, "images.txt")
    split_path = os.path.join(cub_dir, "train_test_split.txt")
    class_path = os.path.join(cub_dir, "image_class_labels.txt")
    bbox_path = os.path.join(cub_dir, "bounding_boxes.txt")

    # 1) images.txt
    images_df = pd.read_csv(images_path, sep=" ", header=None, names=["image_id", "path"])
    # 2) train_test_split.txt
    split_df = pd.read_csv(split_path, sep=" ", header=None, names=["image_id", "is_train"])
    # 3) image_class_labels.txt
    class_df = pd.read_csv(class_path, sep=" ", header=None, names=["image_id", "class"])
    # Merge them
    df = images_df.merge(split_df, on="image_id").merge(class_df, on="image_id")
    # 4) bounding_boxes
    bbox_df = pd.read_csv(bbox_path, sep=" ", header=None,
                          names=["image_id", "x", "y", "width", "height"])
    df = df.merge(bbox_df, on="image_id", how="left")
    
    return df

def load_parts_data(cub_dir):
    """
    Load part info from parts/parts.txt, parts/part_locs.txt,
    and only keep visible=1. Returns a DF with columns:
    [image_id, part_id, part_x, part_y, visible, part_name].
    """
    parts_txt = os.path.join(cub_dir, "parts", "parts.txt")
    partlocs_txt = os.path.join(cub_dir, "parts", "part_locs.txt")
    
    if not (os.path.isfile(parts_txt) and os.path.isfile(partlocs_txt)):
        print("[Warning] parts file not found, returning empty.")
        return pd.DataFrame([])
    
    parts_df = load_parts_file(parts_txt)
    locs_df = pd.read_csv(partlocs_txt, sep=" ", header=None,
                          names=["image_id", "part_id", "part_x", "part_y", "visible"])
    locs_df = locs_df[locs_df["visible"] == 1].copy()
    # Merge with part names
    locs_df = locs_df.merge(parts_df, on="part_id", how="left")
    return locs_df

def load_attribute_data(cub_dir):
    """
    Load attribute info from attributes/attributes.txt and
    attributes/image_attribute_labels.txt. Return a DataFrame with columns:
    [image_id, attribute_id, is_present, certainty_id, time, attribute_name].
    
    We'll only keep rows with is_present==1 and certainty_id>=3.
    """
    attributes_txt = os.path.join(cub_dir, "attributes", "attributes.txt")
    img_attr_labels = os.path.join(cub_dir, "attributes", "image_attribute_labels.txt")
    
    if not (os.path.isfile(attributes_txt) and os.path.isfile(img_attr_labels)):
        print("[Warning] attributes file not found, returning empty DataFrame.")
        return pd.DataFrame([])
    
    # Load attribute names
    rows = []
    with open(attributes_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = line.split(" ", 1)
            if len(tokens) < 2:
                continue
            a_id = int(tokens[0])
            a_name = tokens[1]
            rows.append([a_id, a_name])
    attr_df = pd.DataFrame(rows, columns=["attribute_id", "attribute_name"])
    
    # Load image_attribute_labels
    try:
        df_attrlbl = pd.read_csv(img_attr_labels, sep=" ", header=None,
                                 names=["image_id", "attribute_id", "is_present", "certainty_id", "time"],
                                 on_bad_lines='skip')
    except TypeError:
        # older pandas versions
        df_attrlbl = pd.read_csv(img_attr_labels, sep=" ", header=None,
                                 names=["image_id", "attribute_id", "is_present", "certainty_id", "time"],
                                 error_bad_lines=False, warn_bad_lines=True)
    
    df_attrlbl = df_attrlbl[df_attrlbl["is_present"] == 1].copy()
    df_attrlbl = df_attrlbl[df_attrlbl["certainty_id"] >= 3]
    
    merged = df_attrlbl.merge(attr_df, on="attribute_id", how="left")
    return merged

########################################
# SPLIT LOGIC
########################################

def create_main_dataset(df, cub_dir, output_dir):
    """
    Build the main dataset structure:
      output_dir/train/<class_name>
      output_dir/test/<class_name>
      output_dir/val/<class_name>
    We'll first copy "train" images, then "test" images. 
    But we also want to create a val set from some fraction of test images.
    """
    # Make sure train/val/test directories exist
    for sub in ["train", "val", "test"]:
        subdir = os.path.join(output_dir, sub)
        os.makedirs(subdir, exist_ok=True)
    
    # We'll do a two-pass approach:
    #  1) gather test images by class
    #  2) sample val_fraction from each class, assign them to val, the rest remain test
    pass

def gather_test_images_by_class(df):
    """
    Return a dictionary: class_id -> list of row indices in df that belong to test set for that class.
    """
    test_images_by_class = defaultdict(list)
    for i, row in df.iterrows():
        if row["is_train"] == 0:  # test
            c_id = row["class"]
            test_images_by_class[c_id].append(i)
    return test_images_by_class

def split_test_into_val(df, val_fraction=0.3, seed=42):
    """
    From the test partition, sample a fraction of images per class
    to become val. We'll alter df in place, setting is_train=2 for val rows.
    
    val_fraction=0.3 means 30% of the test images in each class become val.
    """
    random.seed(seed)
    test_by_class = gather_test_images_by_class(df)
    for c_id, indices in test_by_class.items():
        size = len(indices)
        val_size = int(round(size * val_fraction))
        if val_size <= 0 or size == 0:
            continue
        # shuffle the indices
        random.shuffle(indices)
        val_inds = indices[:val_size]
        # mark those as val (use is_train=2 to differentiate)
        for vi in val_inds:
            df.at[vi, "is_train"] = 2

def copy_images_to_split(df, cub_dir, output_dir):
    """
    Actually copy the images to output_dir/{train,val,test}/<class_name>.
    We interpret is_train=1 => train, 2 => val, 0 => test.
    """
    for _, row in df.iterrows():
        if row["is_train"] == 1:
            split="train"
        elif row["is_train"] == 2:
            split="val"
        else:
            split="test"
        
        class_name = str(row["class"]).zfill(3)
        dst_class_dir = os.path.join(output_dir, split, class_name)
        os.makedirs(dst_class_dir, exist_ok=True)
        
        src = os.path.join(cub_dir, "images", row["path"])
        dst = os.path.join(dst_class_dir, os.path.basename(row["path"]))
        if not os.path.exists(dst):
            shutil.copy(src, dst)

########################################
# CONCEPT FOLDER CREATION
########################################

def create_concept_dirs(concept_list, output_dir):
    """
    Build subdirectories for concept_train/<concept>/<concept>
    and concept_test/<concept> for each concept.
    Returns a dict: concept_subdirs[concept] = (train_dir, test_dir)
    """
    c_train_dir = os.path.join(output_dir, "concept_train")
    c_test_dir = os.path.join(output_dir, "concept_test")
    os.makedirs(c_train_dir, exist_ok=True)
    os.makedirs(c_test_dir, exist_ok=True)
    
    concept_subdirs={}
    for cpt in concept_list:
        train_sub = os.path.join(c_train_dir, cpt, cpt)
        test_sub = os.path.join(c_test_dir, cpt)
        os.makedirs(train_sub, exist_ok=True)
        os.makedirs(test_sub, exist_ok=True)
        concept_subdirs[cpt] = (train_sub, test_sub)
    return concept_subdirs

########################################
# PART CONCEPT DATASET
########################################

def create_part_concept_dataset(args, cub_df, parts_df, concept_list):
    """
    For each row in parts_df that matches a user-chosen part concept, 
    we either draw or crop around the part location in the image.
    """
    concept_subdirs = create_concept_dirs(concept_list, args.output_concept)
    
    # Merge parts info with base cub_df
    merged = cub_df.merge(parts_df, on="image_id", how="left")
    
    for _, row in merged.iterrows():
        if pd.isna(row.get("part_name")) or pd.isna(row.get("part_x")):
            continue
        part_name = row["part_name"].strip().lower()
        if part_name not in concept_list:
            continue
        
        # Decide train or test
        # is_train=1 => train, is_train=2 => val, is_train=0 => test
        # For concept data, we do not currently create 'concept_val', only concept_train and concept_test
        # so treat 'val' images like 'train' for concept data
        if row["is_train"] in [1, 2]:
            split="train"
        else:
            split="test"
        
        subdirs = concept_subdirs[part_name]
        (train_subdir, test_subdir) = subdirs
        
        src_img = os.path.join(args.cub_dir, "images", row["path"])
        try:
            im = Image.open(src_img).convert("RGB")
        except Exception as e:
            print(f"[Warning] Could not open {src_img}: {e}")
            continue
        
        out_filename = f"{row['image_id']}_{part_name}.jpg"
        if split=="train":
            dest = os.path.join(train_subdir, out_filename)
        else:
            dest = os.path.join(test_subdir, out_filename)
        
        # bounding box around part_x, part_y of size crop_size
        half = args.crop_size // 2
        px, py = int(row["part_x"]), int(row["part_y"])
        box = (px - half, py - half, px + half, py + half)
        
        if args.draw_or_crop=="draw":
            draw = ImageDraw.Draw(im)
            draw.rectangle(box, outline="red", width=3)
            im.save(dest)
        elif args.draw_or_crop=="crop":
            box = _clip_box_to_image(box, im.width, im.height)
            cropped = im.crop(box)
            cropped = cropped.resize((args.crop_size,args.crop_size))
            cropped.save(dest)
        else:
            # "none" => just copy the original image
            im.save(dest)

########################################
# ATTRIBUTE CONCEPT DATASET
########################################

def create_attribute_concept_dataset(args, cub_df, attr_df, concept_list):
    """
    For each row in attr_df that matches a user-chosen attribute, 
    we either draw or crop around the official bounding box from bounding_boxes.txt,
    or do nothing. 
    """
    concept_subdirs = create_concept_dirs(concept_list, args.output_concept)
    
    attr_df["attribute_name"] = attr_df["attribute_name"].str.lower()
    relevant = attr_df[attr_df["attribute_name"].isin(concept_list)].copy()
    
    merged = relevant.merge(cub_df, on="image_id", how="left")
    
    for _, row in merged.iterrows():
        attr_name = row["attribute_name"]
        if pd.isna(attr_name):
            continue
        
        if row["is_train"] in [1,2]:
            split="train"
        else:
            split="test"
        
        subdirs = concept_subdirs[attr_name]
        (train_subdir, test_subdir)=subdirs
        
        src_img = os.path.join(args.cub_dir, "images", row["path"])
        try:
            im = Image.open(src_img).convert("RGB")
        except Exception as e:
            print(f"[Warning] Could not open {src_img}: {e}")
            continue
        
        out_filename = f"{row['image_id']}_{attr_name}.jpg"
        if split=="train":
            dest = os.path.join(train_subdir, out_filename)
        else:
            dest = os.path.join(test_subdir, out_filename)
        
        x1 = int(row["x"])
        y1 = int(row["y"])
        x2 = int(x1 + row["width"])
        y2 = int(y1 + row["height"])
        
        if args.draw_or_crop=="draw":
            draw = ImageDraw.Draw(im)
            draw.rectangle((x1,y1,x2,y2), outline="blue", width=3)
            im.save(dest)
        elif args.draw_or_crop=="crop":
            box = (x1, y1, x2, y2)
            box = _clip_box_to_image(box, im.width, im.height)
            cropped = im.crop(box)
            cropped = cropped.resize((args.crop_size,args.crop_size))
            cropped.save(dest)
        else:
            im.save(dest)

########################################
# UTILITY 
########################################

def _clip_box_to_image(box, img_w, img_h):
    """
    Clip the bounding box (left, top, right, bottom)
    so it doesn't go out of image bounds.
    """
    left, top, right, bottom = box
    left = max(left, 0)
    top = max(top, 0)
    right = min(right, img_w)
    bottom = min(bottom, img_h)
    return (left, top, right, bottom)

########################################
# MAIN
########################################

def main():
    args = parse_args()

    print("[Info] Loading base CUB annotations...")
    cub_df = load_annotations(args.cub_dir)

    # Step 1: We want to refine the "is_train" column to create a val set from test images
    print("[Info] Splitting test images into val set. val_fraction=%.2f" % args.val_fraction)
    # We'll do it in-place:
    split_test_into_val(cub_df, val_fraction=args.val_fraction, seed=42)

    # Step 2: Actually copy images to main_data
    print("[Info] Creating main dataset (train/val/test) structure in", args.output_main)
    copy_images_to_split(cub_df, args.cub_dir, args.output_main)
    
    # Step 3: Build concept dataset
    concept_list = [c.strip().lower() for c in args.concepts.split(',')]
    
    if args.mode=="parts":
        print("[Info] Mode=parts => building part-based concept dataset.")
        parts_df = load_parts_data(args.cub_dir)
        create_part_concept_dataset(args, cub_df, parts_df, concept_list)
    elif args.mode=="attributes":
        print("[Info] Mode=attributes => building attribute-based concept dataset.")
        attr_df = load_attribute_data(args.cub_dir)
        create_attribute_concept_dataset(args, cub_df, attr_df, concept_list)
    else:
        print("[Error] unknown mode:", args.mode)
    
    print("[Done] Dataset creation complete.")

#########################
# Splitting test => val
#########################

def gather_test_images_by_class(df):
    """
    Return a dictionary: class_id -> list of row indices in df that belong to test set for that class.
    We'll interpret is_train=0 as 'test' for the original CUB labeling.
    """
    from collections import defaultdict
    test_images_by_class = defaultdict(list)
    for i, row in df.iterrows():
        if row["is_train"] == 0:  # test
            c_id = row["class"]
            test_images_by_class[c_id].append(i)
    return test_images_by_class

def split_test_into_val(df, val_fraction=0.3, seed=42):
    """
    From the test partition, sample a fraction of images per class
    to become val. We'll alter df in place, setting is_train=2 for val rows.
    
    val_fraction=0.3 means 30% of the test images in each class become val.
    
    We do not alter the training images (is_train=1). Only test => val.
    """
    import random
    random.seed(seed)
    test_by_class = gather_test_images_by_class(df)
    for c_id, indices in test_by_class.items():
        size = len(indices)
        if size == 0:
            continue
        val_size = int(round(size * val_fraction))
        if val_size <= 0:
            continue
        random.shuffle(indices)
        val_inds = indices[:val_size]
        for vi in val_inds:
            df.at[vi, "is_train"] = 2  # mark as val

def copy_images_to_split(df, cub_dir, output_dir):
    """
    After the test->val splitting, we have:
       is_train=1 => train
       is_train=2 => val
       is_train=0 => test
    We copy images to output_dir/{train,val,test}/<class_name>/image.jpg
    """
    for sub in ["train", "val", "test"]:
        subdir = os.path.join(output_dir, sub)
        os.makedirs(subdir, exist_ok=True)
    
    for _, row in df.iterrows():
        if row["is_train"] == 1:
            split="train"
        elif row["is_train"] == 2:
            split="val"
        else:
            split="test"
        
        class_name = str(row["class"]).zfill(3)
        dst_class_dir = os.path.join(output_dir, split, class_name)
        os.makedirs(dst_class_dir, exist_ok=True)
        
        src = os.path.join(cub_dir, "images", row["path"])
        dst = os.path.join(dst_class_dir, os.path.basename(row["path"]))
        if not os.path.exists(dst):
            shutil.copy(src, dst)

if __name__ == "__main__":
    main()