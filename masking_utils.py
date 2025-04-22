import torch

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


def mask_concepts(model, concept_ds, names_to_mask, mask_value=0.0, default_value=1.0):
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

    return layer_masks
