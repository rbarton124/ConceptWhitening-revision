import torch
import os
from collections import defaultdict
import re

def apply_per_layer_activation_masks(model, layer_masks):
    """
    Apply a separate activation mask to each QCW layer.

    Args:
        model: ResNetQCW model wrapped in nn.DataParallel
        layer_masks: list of 1D tensors, each of shape [num_features]
    """
    assert len(layer_masks) == len(model.module.cw_layers), "Mismatch in number of masks and number of CW layers."
    for cw_layer, mask in zip(model.module.cw_layers, layer_masks):
        cw_layer.set_activation_mask(mask)


def clear_all_activation_masks(model):
    """
    Clear (reset) activation masks for all QCW layers.
    """
    for cw_layer in model.module.cw_layers:
        cw_layer.clear_activation_mask()


def mask_concepts_old(model, concept_ds, names_to_mask, mask_value=0.0, default_value=1.0):
    """
    Create per-layer activation masks by concept name.

    Supports:
        - Masking individual subconcepts (e.g., "wing_white")
        - Masking entire high-level concepts (e.g., "wing")

    Args:
        model: ResNetQCW model wrapped in nn.DataParallel
        concept_ds: the ConceptDataset object
        names_to_mask: list of names (either subconcepts or high-level concepts)
        mask_value: value to assign to masked axes (typically 0.0)
        default_value: default value for all other axes (typically 1.0)

    Returns:
        list of torch.Tensor masks, one per CW layer
    """
    masked_indices = set()

    for name in names_to_mask:
        if name in concept_ds.sc2idx:
            masked_indices.add(concept_ds.sc2idx[name])
        elif name in concept_ds.subspace_mapping:
            masked_indices.update(concept_ds.subspace_mapping[name])
        else:
            print(f"[Warning] Concept '{name}' not found in either subconcepts or high-level concepts.")

    layer_masks = []
    for cw_layer in model.module.cw_layers:
        mask = torch.full((cw_layer.num_features,), default_value, device=next(cw_layer.parameters()).device)
        for idx in masked_indices:
            if 0 <= idx < mask.shape[0]:
                mask[idx] = mask_value
        layer_masks.append(mask)
    
    print(f"masked concept indices: ", masked_indices)

    return layer_masks

import json
import os

def mask_concepts(model, concept_ds, names_to_mask, mask_value=0.0, default_value=1.0, bird_name=None, json_path=None):
    """
    Create per-layer activation masks by concept name or per-bird nonpresent concepts.

    Supports:
        - Masking individual subconcepts (e.g., "wing_white")
        - Masking entire high-level concepts (e.g., "wing")
        - Masking all nonpresent concepts for a given bird species if names_to_mask == "all_nonpresent"
        - Masking all nonpresent concepts across all species if bird_name == "all"

    Args:
        model: ResNetQCW model wrapped in nn.DataParallel
        concept_ds: the ConceptDataset object
        names_to_mask: list of names or string "all_nonpresent"
        mask_value: value to assign to masked axes (typically 0.0)
        default_value: value for all other axes (typically 1.0)
        bird_name: name of the bird species, required if names_to_mask == "all_nonpresent"
        json_path: path to the nonpresent_concepts.json file

    Returns:
        list of torch.Tensor masks, one per CW layer
    """
    masked_indices = set()

    if isinstance(names_to_mask, str) and names_to_mask == "all_nonpresent":
        print(f"to mask all nonpresent concepts")
        if bird_name is None:
            raise ValueError("bird_name must be provided when names_to_mask is 'all_nonpresent'")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"{json_path} not found.")

        with open(json_path, "r") as f:
            all_nonpresent = json.load(f)
            print(f"loaded bird name to nonpresent concepts json")

        if bird_name == "all":
            names_to_mask = set()
            for bird, concepts in all_nonpresent.items():
                names_to_mask.update(concepts)
            print(f"[Masking ALL] Total unique concepts to mask across all birds: {len(names_to_mask)}")
        else:
            if bird_name not in all_nonpresent:
                print(f"[Warning] Bird name '{bird_name}' not found in {json_path}.")
                names_to_mask = []
            else:
                names_to_mask = all_nonpresent[bird_name]

    # Convert to set to avoid duplicate masking
    for name in set(names_to_mask):
        if name in concept_ds.sc2idx:
            masked_indices.add(concept_ds.sc2idx[name])
        elif name in concept_ds.subspace_mapping:
            masked_indices.update(concept_ds.subspace_mapping[name])
        else:
            print(f"[Warning] Concept '{name}' not found in either subconcepts or high-level concepts.")

    layer_masks = []
    for cw_layer in model.module.cw_layers:
        mask = torch.full((cw_layer.num_features,), default_value, device=next(cw_layer.parameters()).device)
        for idx in masked_indices:
            if 0 <= idx < mask.shape[0]:
                mask[idx] = mask_value
        layer_masks.append(mask)

    print(f"[Mask Applied] Total masked concept indices: {len(masked_indices)}")
    return layer_masks


def extract_bird_name(filename):
    # get filename from first image under test directory
    # files = [f for f in os.listdir(data_test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    # files_full_path = [os.path.join(data_test_dir, f) for f in files]
    # first_image = files_full_path[0]
    # filename = os.path.basename(first_image)
    parts = filename.split('_')
    if len(parts) >= 4:
        bird_name = '_'.join(parts[1:-2])
    # else:
    #     bird_name = filename.rsplit('.', 1)[1]
    # print(f"test bird name: ", bird_name)
    return bird_name

def analyze_concepts(concept_train_root, concept_ds):
    all_sc_names = concept_ds.get_subconcept_names()
    all_birds = set()
    subconcept_files = defaultdict(list)

    # Step 1: Walk through files and group them under each subconcept
    for root, _, files in os.walk(concept_train_root):
        subconcept = os.path.relpath(root, concept_train_root)
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                filename = os.path.basename(file)
                bird_name = extract_bird_name(filename)
                all_birds.add(bird_name)
                subconcept_files[subconcept].append(filename)

    # Step 2: Check if bird_name matches any filename under each subconcept using regex
    zero_subconcepts_by_bird = {}
    for bird in sorted(all_birds):
        zero_subconcepts = []
        # Build regex pattern: e.g., r'\bBlack_Footed_Albatross\b' (word boundary)
        pattern = re.compile(re.escape(bird), re.IGNORECASE)
        for sc_name in all_sc_names:
            found = False
            # print(f"sc name: ", sc_name)
            # find the subconcept directory
            if "bill" in sc_name:
                hl_dir = os.path.join(concept_train_root, f"beak")
            elif "wing" in sc_name:
                hl_dir = os.path.join(concept_train_root, f"wing")
            elif "throat" in sc_name:
                hl_dir = os.path.join(concept_train_root, f"throat")
            else:
                hl_dir = os.path.join(concept_train_root, f"general")
            sc_dir = os.path.join(hl_dir, sc_name)
            # print(f"sc dir: ", sc_dir)
            for root, _, files in os.walk(sc_dir):
                for fname in files:
                    # print(f"fname: ", fname)
                    if fname.lower().endswith(('.jpg','.jpeg','.png')):
                        if re.search(pattern, fname):
                            found = True
                            break
            if not found:
                zero_subconcepts.append(sc_name)
        zero_subconcepts_by_bird[bird] = zero_subconcepts
        print("\nSubconcepts that are nonpresent:")
        print(f"{bird}: {', '.join(zero_subconcepts)}")
