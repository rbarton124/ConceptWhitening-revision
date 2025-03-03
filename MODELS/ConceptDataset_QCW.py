import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset

class ConceptDataset(Dataset):
    """
    A simple dataset that:
    1) Recursively scans the `root_dir` folder.
       - `root_dir` might be `concept_train/` or `concept_val/`.
    2) For each subfolder structure: `root_dir/high_level/sub_concept/...images...`
       - If `high_level` is in `high_level_filter`, we include those images.
    3) Looks up bounding box from `bboxes_file` based on the relative path in the folder.
    4) Builds a subspace_mapping that records sub_concepts per high_level.
    5) Returns (image_tensor, bounding_box, hl_label).
    """
    def __init__(self,
                 root_dir: str,
                 bboxes_file: str,
                 high_level_filter: list[str],
                 transform=None,
                 mode: str = 'train'):
        """
        Args:
          root_dir: Path to 'concept_train/' or 'concept_val/'.
          bboxes_file: Path to 'bboxes.json' with bounding boxes for each image (relative path).
          high_level_filter: List of high-level concept folder names to include (e.g. ["eye","nape","beak"]).
          transform: Torchvision transform (augmentations, normalization).
          mode: "train" or "val" or "test", optional. Could be used if you also store concept_val subdir, etc.
        """
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode

        # Read bounding boxes
        with open(bboxes_file, 'r') as f:
            self.bboxes_dict = json.load(f)  # e.g. { "concept_train/eye/has_eye_color::brown/img.jpg": [x1,y1,x2,y2], ... }

        self.high_level_filter = set([hl.lower() for hl in high_level_filter])  # for easy membership check

        # We'll store:
        #   self.samples: a list of (img_path, bounding_box, hl_label)
        #   self.hl2idx: a dict mapping high_level_name -> int index
        #   self.subspace_mapping: e.g. { "eye": ["has_eye_color::brown","has_eye_color::red", ... ], "nape": [...], ... }

        self.samples = []
        self.hl2idx = {}
        self.subspace_mapping = {}  # e.g. { "eye": set_of_subconcepts, "nape": set_of_subconcepts, ... }

        self._scan_directory()

        # Convert subspace sets to sorted lists and build hl2idx
        sorted_hl = sorted(self.subspace_mapping.keys())
        for i, hl_name in enumerate(sorted_hl):
            self.hl2idx[hl_name] = i
            # turn the set_of_subconcepts into a sorted list
            self.subspace_mapping[hl_name] = sorted(list(self.subspace_mapping[hl_name]))

        # Now we have a stable mapping from HL-> idx and HL-> sub_concepts
        # self.samples is ready

    def _scan_directory(self):
        """
        Recursively walk `root_dir`. For each item:
         `root_dir/high_level/sub_concept/.../image.jpg`
        If `high_level` in self.high_level_filter, we gather bounding box from self.bboxes_dict
        and store the sample.
        We also record sub_concept in subspace_mapping[high_level].
        """
        for high_level_folder in os.listdir(self.root_dir):
            hl_path = os.path.join(self.root_dir, high_level_folder)
            if not os.path.isdir(hl_path):
                continue

            hl_lower = high_level_folder.lower()
            if hl_lower not in self.high_level_filter:
                # skip
                continue

            # We keep track of sub_concepts for this HL
            if hl_lower not in self.subspace_mapping:
                self.subspace_mapping[hl_lower] = set()

            # Now, inside high_level_folder are multiple sub_concept folders
            for sub_concept_folder in os.listdir(hl_path):
                sub_concept_path = os.path.join(hl_path, sub_concept_folder)
                if not os.path.isdir(sub_concept_path):
                    continue

                # record sub_concept name
                self.subspace_mapping[hl_lower].add(sub_concept_folder)

                # gather images inside sub_concept_folder
                for root, dirs, files in os.walk(sub_concept_path):
                    for fname in files:
                        # We can check extension if we want
                        if not fname.lower().endswith(('.jpg','.jpeg','.png','.bmp')):
                            continue

                        rel_path = os.path.relpath(os.path.join(root, fname), start=os.path.dirname(self.root_dir))
                        # e.g. "concept_train/eye/has_eye_color::brown/0001_Black_footed_Albatross_0046.jpg"
                        full_path = os.path.join(root, fname)

                        # bounding box from bboxes_dict
                        bbox = self.bboxes_dict.get(rel_path, None)
                        if bbox is None:
                            # maybe default to [0,0,0,0], or skip
                            bbox = [0,0,0,0]  # or skip sample
                            # continue

                        # We'll store (full_image_path, bounding_box, high_level_name)
                        # We'll assign hl_label later in __getitem__, or build it now
                        self.samples.append((full_path, bbox, hl_lower))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, box_coords, hl_name = self.samples[idx]
        # hl_name => we find the integer label
        hl_label = self.hl2idx[hl_name]
        # load image
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        # region => Torch expects e.g. a Tensor with shape (4,) if we want [x1,y1,x2,y2]
        # We keep them as floats
        region_tensor = torch.tensor(box_coords, dtype=torch.float32)

        return img, region_tensor, hl_label

    def get_num_high_level(self):
        return len(self.hl2idx)

    def get_hl_names(self):
        return sorted(self.hl2idx.keys(), key=lambda x: self.hl2idx[x])

    def __repr__(self):
        return (f"ConceptDataset(\n"
                f"  root_dir={self.root_dir},\n"
                f"  bboxes=<{len(self.bboxes_dict)} entries>,\n"
                f"  high_level_filter={self.high_level_filter},\n"
                f"  samples={len(self.samples)}\n)")

    @property
    def subspace_mapping(self):
        """
        We define a property so external code can get the subspace dictionary easily.
        In this example, subspace_mapping is e.g.
           {
             "eye": ["has_eye_color::brown","has_eye_color::red", ...],
             "nape": [...],
             ...
           }
        Then the training script decides how to turn sub_concepts => dimension indices if it wants to.
        """
        return self._subspace_mapping

    @subspace_mapping.setter
    def subspace_mapping(self, val):
        self._subspace_mapping = val
