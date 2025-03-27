import os
import json
from PIL import Image, ImageDraw
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import ImageFilter

class ConceptDataset(Dataset):
    """
    Each sample is a physically cropped or redacted image (or full image)
    plus the integer label for subconcept. Also keeps track of
    which subconcepts are free (unlabeled) vs. labeled.
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
          crop_mode: "crop", "redact", "blur", or "none".
             - "crop": physically crop each image to bounding box, then do transforms
             - "redact": fill everything outside the bounding box with black (0), keep the image size
             - "blur": blur everything outside the bounding box
             - "none": do nothing (ignores bounding box)
        """
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.crop_mode = crop_mode.lower().strip()

        # Load bounding boxes
        with open(bboxes_file, 'r') as f:
            self.bboxes_dict = json.load(f)

        # Convert HL filter to set for easy membership checking
        self.high_level_filter = set([hl.lower() for hl in high_level_filter])

        # We'll build:
        #  self.samples -> list of (full_image_path, bbox, hl_name, sc_name)
        #  self.hl2idx -> dict mapping high_level_name -> int
        #  self.sc2idx -> dict mapping subconcept_name -> int
        #  self._subspace_mapping -> {hl_name: list_of_subconcept_indices}
        #
        #  Additionally, we track free vs. labeled subconcepts in each HL:
        #  self._free_map[hl_name] = set of subconcepts that end in "_free"
        #  self._labeled_map[hl_name] = set of subconcepts that do NOT end in "_free"
        self.samples = []
        self.hl2idx = {}
        self.sc2idx = {}

        self._subspace_mapping = {}  # { hl_name -> set of sc_name }
        self._free_map = {}         # { hl_name -> set of sc_name that are free  }
        self._labeled_map = {}      # { hl_name -> set of sc_name that are labeled }

        self._scan_directory()

        # Finalize subspace mappings and build concept indices
        self._build_indices()

    def _scan_directory(self):
        """
        Walks root_dir, expecting structure:
          root_dir/
            high_level_concept/
               sub_concept_1/
               sub_concept_2/
               ...
        We skip directories whose HL name isn't in self.high_level_filter.
        We store all sub-concepts, but also check if name ends in "_free".
        """
        for high_level_folder in os.listdir(self.root_dir):
            hl_path = os.path.join(self.root_dir, high_level_folder)
            if not os.path.isdir(hl_path):
                continue

            hl_lower = high_level_folder.lower()
            if hl_lower not in self.high_level_filter:
                continue

            if hl_lower not in self._subspace_mapping:
                self._subspace_mapping[hl_lower] = set()
                self._free_map[hl_lower] = set()
                self._labeled_map[hl_lower] = set()

            # sub_concept dirs
            for sc_folder in os.listdir(hl_path):
                sc_path = os.path.join(hl_path, sc_folder)
                if not os.path.isdir(sc_path):
                    continue

                # Check if sc_folder ends with "_free"
                if sc_folder.lower().endswith("_free"):
                    self._free_map[hl_lower].add(sc_folder)
                else:
                    self._labeled_map[hl_lower].add(sc_folder)

                # Add to the universal subspace mapping for that HL
                self._subspace_mapping[hl_lower].add(sc_folder)

                # Collect images
                for root_, dirs_, files_ in os.walk(sc_path):
                    for fname in files_:
                        if not fname.lower().endswith(('.jpg','.jpeg','.png','.bmp')):
                            continue
                        full_path = os.path.join(root_, fname)
                        rel_path  = os.path.relpath(full_path, start=os.path.dirname(self.root_dir))

                        # bounding box from bboxes_dict if present
                        bbox = self.bboxes_dict.get(rel_path, None)
                        if bbox is None:
                            bbox = [0,0,0,0]

                        # store sample
                        self.samples.append((full_path, bbox, hl_lower, sc_folder))

    def _build_indices(self):
        """
        For each HL concept, we have a set of sub-concept names. We sort them,
        assign integer indices, store in self.sc2idx, and then store the resulting
        index list in self._subspace_mapping[hl].
        """
        sorted_hls = sorted(self._subspace_mapping.keys())
        for i, hl_name in enumerate(sorted_hls):
            self.hl2idx[hl_name] = i
            sc_names_list = sorted(self._subspace_mapping[hl_name])
            sc_indices = []
            for sc_name in sc_names_list:
                if sc_name not in self.sc2idx:
                    self.sc2idx[sc_name] = len(self.sc2idx)
                sc_indices.append(self.sc2idx[sc_name])
            # Now replace the set with an ordered list of indices
            self._subspace_mapping[hl_name] = sc_indices

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, box_coords, hl_name, sc_name = self.samples[idx]
        hl_label = self.hl2idx[hl_name]
        sc_label = self.sc2idx[sc_name]

        # Load image
        img = Image.open(img_path).convert("RGB")
        x1, y1, x2, y2 = box_coords

        # Crop / redact / blur logic
        if self.crop_mode == "crop":
            x1_cl, y1_cl = max(0, x1), max(0, y1)
            x2_cl, y2_cl = min(x2, img.width), min(y2, img.height)
            if x2_cl > x1_cl and y2_cl > y1_cl:
                img = img.crop((x1_cl, y1_cl, x2_cl, y2_cl))

        elif self.crop_mode == "blur":
            w, h = img.size
            blur_radius = max(3, int(min(w,h)*0.15))
            blurred_img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            box_region = img.crop((x1, y1, x2, y2))
            blurred_img.paste(box_region, (int(x1), int(y1)))
            img = blurred_img

        elif self.crop_mode.startswith("redact"):
            # e.g. "redact" or "redact_av"
            if self.crop_mode == "redact_av":
                np_img = np.array(img.convert("RGB"))
                mean_color = np_img.reshape(-1,3).mean(axis=0)
                fill_color = tuple(int(c) for c in mean_color)
            else:
                fill_color = (0,0,0)

            draw = ImageDraw.Draw(img)
            draw.rectangle([0, 0, img.width, y1], fill=fill_color)
            draw.rectangle([0, y2, img.width, img.height], fill=fill_color)
            draw.rectangle([0, y1, x1, y2], fill=fill_color)
            draw.rectangle([x2, y1, img.width, y2], fill=fill_color)

        # else: do nothing for "none"

        if self.transform:
            img = self.transform(img)

        # Return (image, subconcept_label, image_path)
        return img, sc_label, img_path

    ############################################################################
    # Properties + Getters
    ############################################################################

    @property
    def subspace_mapping(self):
        """
        Returns { hl_name -> [sc_label, sc_label, ...] }
        (the integer indices of each sub-concept in that subspace).
        """
        return self._subspace_mapping

    def get_num_subconcepts(self):
        return len(self.sc2idx)
    
    def get_subconcept_names(self):
        # Return sub-concept names sorted by their integer index
        return sorted(self.sc2idx.keys(), key=lambda x: self.sc2idx[x])

    def get_num_high_level(self):
        return len(self.hl2idx)

    def get_hl_names(self):
        return sorted(self.hl2idx.keys(), key=lambda x: self.hl2idx[x])
        
    def get_subconcept_to_hl_mapping(self):
        """Returns a dict: subconcept_index -> high_level_index"""
        sc_to_hl = {}
        for hl_name, sc_indices in self._subspace_mapping.items():
            hl_idx = self.hl2idx[hl_name]
            for sc_idx in sc_indices:
                sc_to_hl[sc_idx] = hl_idx
        return sc_to_hl

    def get_subconcept_to_hl_name_mapping(self):
        """Returns a dict: subconcept_index -> high_level_name"""
        sc_to_hl_name = {}
        for hl_name, sc_indices in self._subspace_mapping.items():
            for sc_idx in sc_indices:
                sc_to_hl_name[sc_idx] = hl_name
        return sc_to_hl_name

    ############################################################################
    # New: Distinguish labeled vs. free subconcepts
    ############################################################################

    def is_free_subconcept_name(self, sc_name: str) -> bool:
        """
        Returns True if `sc_name` ends with '_free'.
        """
        return sc_name.lower().endswith("_free")

    def get_hl_free_subconcepts(self, hl_name: str) -> list[int]:
        """
        For a given high-level concept (string), return a list of *subconcept indices*
        that are considered free (end in "_free").
        """
        hl_name = hl_name.lower()
        if hl_name not in self._subspace_mapping:
            return []
        all_sc_indices = self._subspace_mapping[hl_name]  # list of int
        result = []
        for sc_idx in all_sc_indices:
            # figure out the string name from sc_idx
            # recall sc2idx is name->idx, so invert it:
            # we can do a quick search or keep a separate reversed dict
            for sc_name, sc_val in self.sc2idx.items():
                if sc_val == sc_idx:
                    if self.is_free_subconcept_name(sc_name):
                        result.append(sc_idx)
                    break
        return result

    def get_hl_labeled_subconcepts(self, hl_name: str) -> list[int]:
        """
        For a given high-level concept, return a list of subconcept indices
        that are NOT free-labeled (i.e. do not end in "_free").
        """
        hl_name = hl_name.lower()
        if hl_name not in self._subspace_mapping:
            return []
        all_sc_indices = self._subspace_mapping[hl_name]
        result = []
        for sc_idx in all_sc_indices:
            for sc_name, sc_val in self.sc2idx.items():
                if sc_val == sc_idx:
                    if not self.is_free_subconcept_name(sc_name):
                        result.append(sc_idx)
                    break
        return result

    def __repr__(self):
        return (
            f"ConceptDataset(\n"
            f"  root_dir={self.root_dir},\n"
            f"  bboxes=<{len(self.bboxes_dict)} entries>,\n"
            f"  high_level_filter={self.high_level_filter},\n"
            f"  samples={len(self.samples)},\n"
            f"  subconcepts={len(self.sc2idx)},\n"
            f"  mode={self.crop_mode}\n"
            f")"
        )
