import os
import zipfile
import shutil
import random
from kaggle.api.kaggle_api_extended import KaggleApi

def download_places365(output_dir):
    """
    Downloads and prepares the Places365 dataset from Kaggle into the structure:
      data_256/
        train/...
        val/...
        test/...
    We create 'test/' ourselves by sampling from 'val/'.
    """
    os.makedirs(output_dir, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    dataset = "benjaminkz/places365"
    print(f"Downloading dataset '{dataset}' from Kaggle...")
    api.dataset_download_files(dataset, path=output_dir, unzip=False)
    print("Download complete!")

    zip_path = os.path.join(output_dir, "places365.zip")
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Expected {zip_path} but did not find it.")

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(output_dir, "places365"))
    print("Extraction complete.")

    # Clean up the ZIP
    os.remove(zip_path)
    print("Removed the original ZIP file.")

    extracted_root = os.path.join(output_dir, "places365")
    # Inside "places365", we expect something like:
    #  places365/
    #     train/
    #     val/
    #  etc.

    # If the Kaggle dataset only has train/ and val/ in "places365_standard" or something,
    # adjust accordingly. We'll check for "train" and "val" inside extracted_root:
    train_src = os.path.join(extracted_root, "train")
    val_src   = os.path.join(extracted_root, "val")

    if not os.path.exists(train_src) or not os.path.exists(val_src):
        raise Exception("Could not find 'train' or 'val' folders inside the extracted dataset. "
                        "Check the extracted structure under 'places365'.")

    # Move train/ and val/ into data_256/train and data_256/val
    train_dest = os.path.join(extracted_root, "train")
    val_dest   = os.path.join(extracted_root, "val")

    # Now create a test/ folder by sampling images from val/
    test_dest = os.path.join(extracted_root, "test")
    os.makedirs(test_dest, exist_ok=True)
    create_test_split(val_dest, test_dest, num_per_class=20)

    print("Dataset is reorganized. Ready for training with the CW repo!")


def create_test_split(val_dir, test_dir, num_per_class=20):
    """
    Creates a 'test/' folder by randomly moving num_per_class images
    from each category in 'val/' to 'test/'.
    """
    categories = os.listdir(val_dir)
    for cat in categories:
        cat_path = os.path.join(val_dir, cat)
        if not os.path.isdir(cat_path):
            continue  # skip non-folders

        # Make corresponding category folder in test/
        test_cat_path = os.path.join(test_dir, cat)
        os.makedirs(test_cat_path, exist_ok=True)

        images = os.listdir(cat_path)
        images = [img for img in images if os.path.isfile(os.path.join(cat_path, img))]
        # Shuffle to randomize
        random.shuffle(images)

        # Take up to num_per_class images (handle if fewer exist)
        move_count = min(num_per_class, len(images))
        to_move = images[:move_count]

        for img_name in to_move:
            src_img = os.path.join(cat_path, img_name)
            dst_img = os.path.join(test_cat_path, img_name)
            shutil.move(src_img, dst_img)

    print(f"Moved up to {num_per_class} images from each val/* folder into test/*.")


if __name__ == "__main__":
    output_directory = "data"
    download_places365(output_directory)
