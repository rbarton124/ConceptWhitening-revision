import os
import json
import re
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from torch.utils.data import Dataset

# >>> FIX: match "_free", "_free0", "_free12", etc.
_FREE_RE = re.compile(r"_free(\d+)?$", re.IGNORECASE)

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
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.crop_mode = crop_mode.lower().strip()

        with open(bboxes_file, 'r') as f:
            self.bboxes_dict = json.load(f)

        self.high_level_filter = set([hl.lower() for hl in high_level_filter])

        self.samples = []
        self.hl2idx = {}
        self.sc2idx = {}
        self.idx2sc = {}  # >>> FIX: build once for fast reverse lookup

        self._subspace_mapping = {}  # { hl_name -> set(sc_name) initially, later list[int] }
        self._free_map = {}         # { hl_name -> set(sc_name) free  }
        self._labeled_map = {}      # { hl_name -> set(sc_name) labeled }

        self._scan_directory()
        self._build_indices()

    def _scan_directory(self):
        """
        Walks root_dir, expecting structure:
          root_dir/
            high_level_concept/
               sub_concept_1/
               sub_concept_2/
               ...
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

            for sc_folder in os.listdir(hl_path):
                sc_path = os.path.join(hl_path, sc_folder)
                if not os.path.isdir(sc_path):
                    continue

                # >>> FIX: robust free detection
                if _FREE_RE.search(sc_folder) is not None:
                    self._free_map[hl_lower].add(sc_folder)
                else:
                    self._labeled_map[hl_lower].add(sc_folder)

                self._subspace_mapping[hl_lower].add(sc_folder)

                for root_, _, files_ in os.walk(sc_path):
                    for fname in files_:
                        if not fname.lower().endswith(('.jpg','.jpeg','.png','.bmp')):
                            continue
                        full_path = os.path.join(root_, fname)
                        rel_path  = os.path.relpath(full_path, start=os.path.dirname(self.root_dir))

                        bbox = self.bboxes_dict.get(rel_path, None)
                        if bbox is None:
                            bbox = [0, 0, 0, 0]

                        self.samples.append((full_path, bbox, hl_lower, sc_folder))

    def _build_indices(self):
        sorted_hls = sorted(self._subspace_mapping.keys())
        for i, hl_name in enumerate(sorted_hls):
            self.hl2idx[hl_name] = i
            sc_names_list = sorted(self._subspace_mapping[hl_name])

            sc_indices = []
            for sc_name in sc_names_list:
                if sc_name not in self.sc2idx:
                    self.sc2idx[sc_name] = len(self.sc2idx)
                sc_indices.append(self.sc2idx[sc_name])

            self._subspace_mapping[hl_name] = sc_indices

        # >>> FIX: build reverse lookup once
        self.idx2sc = {idx: name for name, idx in self.sc2idx.items()}

    def __len__(self):
        return len(self.samples)

    def _parse_and_clamp_bbox(self, box_coords, img_w: int, img_h: int):
        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in box_coords]
        except Exception:
            x1 = y1 = x2 = y2 = 0

        # clamp into image bounds
        x1 = max(0, min(x1, img_w))
        x2 = max(0, min(x2, img_w))
        y1 = max(0, min(y1, img_h))
        y2 = max(0, min(y2, img_h))

        return x1, y1, x2, y2

    def __getitem__(self, idx):
        img_path, box_coords, hl_name, sc_name = self.samples[idx]
        sc_label = self.sc2idx[sc_name]

        img = Image.open(img_path).convert("RGB")
        x1, y1, x2, y2 = self._parse_and_clamp_bbox(box_coords, img.width, img.height)

        if self.crop_mode == "crop":
            if x2 > x1 and y2 > y1:
                img = img.crop((x1, y1, x2, y2))

        elif self.crop_mode == "blur":
            if x2 > x1 and y2 > y1:
                w, h = img.size
                blur_radius = max(3, int(min(w, h) * 0.15))
                blurred_img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                box_region = img.crop((x1, y1, x2, y2))
                blurred_img.paste(box_region, (int(x1), int(y1)))
                img = blurred_img
            # else: bbox invalid => leave unchanged

        elif self.crop_mode.startswith("redact"):
            if self.crop_mode == "redact_av":
                np_img = np.array(img.convert("RGB"))
                mean_color = np_img.reshape(-1, 3).mean(axis=0)
                fill_color = tuple(int(c) for c in mean_color)
            else:
                fill_color = (0, 0, 0)

            draw = ImageDraw.Draw(img)
            draw.rectangle([0, 0, img.width, y1], fill=fill_color)
            draw.rectangle([0, y2, img.width, img.height], fill=fill_color)
            draw.rectangle([0, y1, x1, y2], fill=fill_color)
            draw.rectangle([x2, y1, img.width, y2], fill=fill_color)

        # else: "none" => do nothing

        if self.transform:
            img = self.transform(img)

        return img, sc_label, img_path

    ############################################################################
    # Properties + Getters
    ############################################################################
    @property
    def subspace_mapping(self):
        """
        Returns { hl_name -> [sc_label, sc_label, ...] }
        """
        return self._subspace_mapping

    def get_num_subconcepts(self):
        return len(self.sc2idx)

    def get_subconcept_names(self):
        # Return sub-concept names sorted by their integer index
        return [self.idx2sc[i] for i in range(len(self.idx2sc))]

    def get_num_high_level(self):
        return len(self.hl2idx)

    def get_hl_names(self):
        return sorted(self.hl2idx.keys(), key=lambda x: self.hl2idx[x])

    def get_subconcept_to_hl_mapping(self):
        sc_to_hl = {}
        for hl_name, sc_indices in self._subspace_mapping.items():
            hl_idx = self.hl2idx[hl_name]
            for sc_idx in sc_indices:
                sc_to_hl[sc_idx] = hl_idx
        return sc_to_hl

    def get_subconcept_to_hl_name_mapping(self):
        sc_to_hl_name = {}
        for hl_name, sc_indices in self._subspace_mapping.items():
            for sc_idx in sc_indices:
                sc_to_hl_name[sc_idx] = hl_name
        return sc_to_hl_name

    ############################################################################
    # Labeled vs free
    ############################################################################
    def is_free_subconcept_name(self, sc_name: str) -> bool:
        return _FREE_RE.search(sc_name) is not None

    def get_hl_free_subconcepts(self, hl_name: str) -> list[int]:
        hl_name = hl_name.lower()
        if hl_name not in self._subspace_mapping:
            return []
        return [sc_idx for sc_idx in self._subspace_mapping[hl_name]
                if self.is_free_subconcept_name(self.idx2sc.get(sc_idx, ""))]

    def get_hl_labeled_subconcepts(self, hl_name: str) -> list[int]:
        hl_name = hl_name.lower()
        if hl_name not in self._subspace_mapping:
            return []
        return [sc_idx for sc_idx in self._subspace_mapping[hl_name]
                if not self.is_free_subconcept_name(self.idx2sc.get(sc_idx, ""))]

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
