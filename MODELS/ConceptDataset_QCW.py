import os
import json
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset

class ConceptDataset(Dataset):
    """
    Each sample is a physically cropped or redacted image (or full image)
    plus the integer label for subconcept.
    """
    def __init__(
        self,
        root_dir: str,
        bboxes_file: str,
        high_level_filter: list[str],
        transform=None,
        crop_mode: str = "crop"
    ):
        """
        Args:
          root_dir: Path like 'concept_train/' or 'concept_val/'.
          bboxes_file: Path to 'bboxes.json' with bounding boxes. 
                       Keys = relative path, value=[x1,y1,x2,y2].
          high_level_filter: Which high-level concepts to include (e.g., ["eye","beak","nape"]).
          transform: Torch transforms for augmentation.
          crop_mode: "crop", "redact", or "none".
             - "crop": physically crop each image to bounding box, then do transforms
             - "redact": fill everything outside the bounding box with black (0), keep the image size
             - "none": do nothing (ignores bounding box)
        """
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.crop_mode = crop_mode.lower().strip()

        with open(bboxes_file, 'r') as f:
            self.bboxes_dict = json.load(f)

        self.high_level_filter = set([hl.lower() for hl in high_level_filter])

        # We'll build:
        #  self.samples -> list of (full_image_path, bbox, hl_name, sc_name)
        #  self.hl2idx -> dict mapping high_level_name -> int
        #  self.sc2idx -> dict mapping subconcept_name -> int
        #  self.subspace_mapping -> {hl_name: list_of_subconcept_indices}
        self.samples = []
        self.hl2idx = {}
        self.sc2idx = {}
        self._subspace_mapping = {}  # store internally

        self._scan_directory()

        # finalize subspace mappings and build indices
        sorted_hls = sorted(self._subspace_mapping.keys())
        for i, hl_name in enumerate(sorted_hls):
            self.hl2idx[hl_name] = i
            # Convert subconcept sets to sorted lists and map them to indices
            sorted_subconcepts = sorted(self._subspace_mapping[hl_name])
            subconcept_indices = []
            
            for sc in sorted_subconcepts:
                if sc not in self.sc2idx:
                    self.sc2idx[sc] = len(self.sc2idx)
                subconcept_indices.append(self.sc2idx[sc])
                
            self._subspace_mapping[hl_name] = subconcept_indices

    def _scan_directory(self):
        """ Walk root_dir, e.g. concept_train/high_level/sub_concept. """
        for high_level_folder in os.listdir(self.root_dir):
            hl_path = os.path.join(self.root_dir, high_level_folder)
            if not os.path.isdir(hl_path):
                continue

            hl_lower = high_level_folder.lower()
            if hl_lower not in self.high_level_filter:
                continue

            if hl_lower not in self._subspace_mapping:
                self._subspace_mapping[hl_lower] = set()

            # sub_concept dirs
            for sc_folder in os.listdir(hl_path):
                sc_path = os.path.join(hl_path, sc_folder)
                if not os.path.isdir(sc_path):
                    continue

                self._subspace_mapping[hl_lower].add(sc_folder)

                for root, dirs, files in os.walk(sc_path):
                    for fname in files:
                        if not fname.lower().endswith(('.jpg','.jpeg','.png','.bmp')):
                            continue
                        full_path = os.path.join(root, fname)
                        rel_path  = os.path.relpath(full_path, start=os.path.dirname(self.root_dir))
                        # bounding box from bboxes_dict if present
                        # can skip if none found or default [0,0,0,0]
                        # We'll store (image_path, bounding_box, hl_name, sc_folder)
                        bbox = self.bboxes_dict.get(rel_path, None)
                        if bbox is None:
                            # user might skip or default
                            bbox = [0,0,0,0]
                        
                        # store sample with both high-level concept and subconcept information
                        self.samples.append((full_path, bbox, hl_lower, sc_folder))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, box_coords, hl_name, sc_name = self.samples[idx]
        hl_label = self.hl2idx[hl_name]
        sc_label = self.sc2idx[sc_name]

        # load image
        img = Image.open(img_path).convert("RGB")
        x1, y1, x2, y2 = box_coords

        if self.crop_mode == "crop":
            # physically crop
            # also be sure to clamp coords
            x1_cl, y1_cl = max(0, x1), max(0, y1)
            x2_cl, y2_cl = min(x2, img.width), min(y2, img.height)
            if x2_cl > x1_cl and y2_cl > y1_cl:
                img = img.crop((x1_cl, y1_cl, x2_cl, y2_cl))
            # else if invalid, we skip or let it continue
        elif self.crop_mode == "redact":
            # black out everything outside the bounding box
            # for simplicity, let's define a quick approach:
            draw = ImageDraw.Draw(img)
            # top region
            draw.rectangle([0,0, img.width, y1], fill=(0,0,0))
            # bottom region
            draw.rectangle([0, y2, img.width, img.height], fill=(0,0,0))
            # left region
            draw.rectangle([0, y1, x1, y2], fill=(0,0,0))
            # right region
            draw.rectangle([x2, y1, img.width, y2], fill=(0,0,0))
            # now the box area is left as is
        # else "none", do nothing

        # Now apply transforms if any
        if self.transform:
            img = self.transform(img)

        # Return image tensor, subconcept label, and high-level concept label
        # This lets us align by subconcept while still tracking high-level concept grouping
        return img, sc_label, hl_label

    @property
    def subspace_mapping(self):
        return self._subspace_mapping
    
    def get_num_subconcepts(self):
        return len(self.sc2idx)
    
    def get_subconcept_names(self):
        return sorted(self.sc2idx.keys(), key=lambda x: self.sc2idx[x])

    def get_num_high_level(self):
        return len(self.hl2idx)

    def get_hl_names(self):
        return sorted(self.hl2idx.keys(), key=lambda x: self.hl2idx[x])
        
    def get_subconcept_to_hl_mapping(self):
        """Returns a mapping from subconcept index to high-level concept index"""
        sc_to_hl = {}
        for hl_name, sc_indices in self._subspace_mapping.items():
            hl_idx = self.hl2idx[hl_name]
            for sc_idx in sc_indices:
                sc_to_hl[sc_idx] = hl_idx
        return sc_to_hl

    def __repr__(self):
        return (f"ConceptDataset(\n"
                f" root_dir={self.root_dir},\n"
                f" bboxes=<{len(self.bboxes_dict)} entries>,\n"
                f" high_level_filter={self.high_level_filter},\n"
                f" samples={len(self.samples)},\n"
                f" subconcepts={len(self.sc2idx)},\n"
                f" mode={self.crop_mode}\n)")
