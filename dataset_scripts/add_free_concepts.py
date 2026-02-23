#!/usr/bin/env python3
"""
Utility script to add free (unlabeled) concept folders to an existing QCW concept dataset.

For each high-level concept (e.g., "wing", "beak"), this creates N free subconcept folders
and populates them with all images from that high-level concept (mixed from all labeled subconcepts).

This enables the winner-takes-all mechanism to discover unlabeled subconcepts within each high-level concept.

Usage:
    python add_free_concepts.py \
        --concept_dir /path/to/concept_dataset \
        --n_free_per_hl 2 \
        --exclude_concepts general \
        --dry_run  # To preview without actually copying files
"""

import os
import argparse
import shutil
import random
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Add free concept folders to QCW concept dataset")
    parser.add_argument("--concept_dir", required=True,
                       help="Path to concept dataset (should contain concept_train/ and concept_val/)")
    parser.add_argument("--n_free_per_hl", type=int, default=2,
                       help="Number of free subconcept folders to create per high-level concept")
    parser.add_argument("--exclude_concepts", type=str, default="general",
                       help="Comma-separated list of high-level concepts to exclude (default: general)")
    parser.add_argument("--splits", type=str, default="concept_train",
                       help="Comma-separated list of splits to process (e.g., 'concept_train,concept_val')")
    parser.add_argument("--sample_fraction", type=float, default=1.0,
                       help="Fraction of images to include in each free folder (1.0 = all images)")
    parser.add_argument("--dry_run", action="store_true",
                       help="Preview changes without actually creating files")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for sampling")
    return parser.parse_args()


def add_free_concepts_to_split(split_dir, n_free, exclude_concepts, sample_fraction, dry_run, seed):
    """
    Add free concept folders to a single split (e.g., concept_train).
    
    Args:
        split_dir: Path to split directory (e.g., /path/to/concept_train)
        n_free: Number of free folders per HL concept
        exclude_concepts: Set of HL concepts to skip
        sample_fraction: What fraction of images to include (1.0 = all)
        dry_run: If True, don't actually create files
        seed: Random seed
    """
    random.seed(seed)
    
    if not os.path.isdir(split_dir):
        print(f"[Warning] Split directory not found: {split_dir}")
        return
    
    print(f"\nProcessing split: {os.path.basename(split_dir)}")
    print("=" * 50)
    
    stats = {"hl_processed": 0, "hl_skipped": 0, "free_folders_created": 0, "images_copied": 0}
    
    # Iterate over high-level concept folders
    for hl_folder in sorted(os.listdir(split_dir)):
        hl_path = os.path.join(split_dir, hl_folder)
        if not os.path.isdir(hl_path):
            continue
        
        hl_lower = hl_folder.lower()
        
        # Skip excluded concepts
        if hl_lower in exclude_concepts:
            print(f"  Skipping '{hl_folder}' (excluded)")
            stats["hl_skipped"] += 1
            continue
        
        # Skip if already has free folders
        existing_free = [f for f in os.listdir(hl_path) if f.endswith("_free")]
        if existing_free and not dry_run:
            print(f"  Skipping '{hl_folder}' (already has {len(existing_free)} free folders)")
            stats["hl_skipped"] += 1
            continue
        
        # Collect all images from labeled subconcept folders
        all_images = []
        labeled_folders = []
        
        for sc_folder in os.listdir(hl_path):
            if sc_folder.endswith("_free"):
                continue  # Skip existing free folders
            
            sc_path = os.path.join(hl_path, sc_folder)
            if not os.path.isdir(sc_path):
                continue
            
            labeled_folders.append(sc_folder)
            
            # Collect images
            for fname in os.listdir(sc_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(sc_path, fname)
                    all_images.append(img_path)
        
        if not all_images:
            print(f"  Skipping '{hl_folder}' (no images found)")
            stats["hl_skipped"] += 1
            continue
        
        # Sample if requested
        if sample_fraction < 1.0:
            k = int(len(all_images) * sample_fraction)
            all_images = random.sample(all_images, k)
        
        print(f"\n  HL '{hl_folder}':")
        print(f"    Labeled subconcepts: {len(labeled_folders)}")
        print(f"    Total images: {len(all_images)}")
        print(f"    Creating {n_free} free subconcept folders...")
        
        # Create free concept folders
        for i in range(1, n_free + 1):
            free_folder_name = f"{hl_folder}_free_{i}"
            free_folder_path = os.path.join(hl_path, free_folder_name)
            
            if dry_run:
                print(f"      [DRY RUN] Would create: {free_folder_name} with {len(all_images)} images")
                stats["free_folders_created"] += 1
                stats["images_copied"] += len(all_images)
            else:
                os.makedirs(free_folder_path, exist_ok=True)
                
                # Copy all images to this free folder
                copied = 0
                for img_path in tqdm(all_images, desc=f"      {free_folder_name}", leave=False):
                    dest = os.path.join(free_folder_path, os.path.basename(img_path))
                    if not os.path.exists(dest):
                        shutil.copy(img_path, dest)
                        copied += 1
                
                print(f"      ✓ Created {free_folder_name}: {copied} images")
                stats["free_folders_created"] += 1
                stats["images_copied"] += copied
        
        stats["hl_processed"] += 1
    
    print(f"\n{os.path.basename(split_dir)} Summary:")
    print(f"  HL concepts processed: {stats['hl_processed']}")
    print(f"  HL concepts skipped: {stats['hl_skipped']}")
    print(f"  Free folders created: {stats['free_folders_created']}")
    print(f"  Total images copied: {stats['images_copied']}")
    
    return stats


def main():
    args = parse_args()
    
    # Parse exclusions
    exclude_set = set([x.strip().lower() for x in args.exclude_concepts.split(",")])
    
    # Parse splits
    splits = [x.strip() for x in args.splits.split(",")]
    
    print("="*70)
    print(" "*15 + "ADD FREE CONCEPTS TO QCW DATASET")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Concept directory: {args.concept_dir}")
    print(f"  Free folders per HL: {args.n_free_per_hl}")
    print(f"  Excluded concepts: {exclude_set}")
    print(f"  Splits to process: {splits}")
    print(f"  Sample fraction: {args.sample_fraction}")
    print(f"  Dry run: {args.dry_run}")
    print(f"  Seed: {args.seed}\n")
    
    if args.dry_run:
        print("[DRY RUN MODE] No files will be modified.\n")
    
    # Process each split
    total_stats = {"hl_processed": 0, "hl_skipped": 0, "free_folders_created": 0, "images_copied": 0}
    
    for split_name in splits:
        split_dir = os.path.join(args.concept_dir, split_name)
        
        if not os.path.isdir(split_dir):
            print(f"[Warning] Split directory not found: {split_dir}")
            continue
        
        split_stats = add_free_concepts_to_split(
            split_dir,
            args.n_free_per_hl,
            exclude_set,
            args.sample_fraction,
            args.dry_run,
            args.seed
        )
        
        # Aggregate stats
        for key in total_stats:
            total_stats[key] += split_stats[key]
    
    # Final summary
    print("\n" + "="*70)
    print(" "*25 + "FINAL SUMMARY")
    print("="*70)
    print(f"  Total HL concepts processed: {total_stats['hl_processed']}")
    print(f"  Total HL concepts skipped: {total_stats['hl_skipped']}")
    print(f"  Total free folders created: {total_stats['free_folders_created']}")
    print(f"  Total images copied: {total_stats['images_copied']}")
    
    if args.dry_run:
        print(f"\n[DRY RUN COMPLETE] Re-run without --dry_run to actually create files.")
    else:
        print(f"\n[COMPLETE] Free concept folders have been added to {args.concept_dir}")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

