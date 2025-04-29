import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import shutil
from types import SimpleNamespace
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from MODELS.model_resnet_qcw import build_resnet_qcw, get_last_qcw_layer
from MODELS.ConceptDataset_QCW import ConceptDataset

# Model loading stuff
def resume_checkpoint(model, checkpoint_path):
    """Load weights from checkpoint, handling key renaming"""
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        print(f"[Checkpoint] No checkpoint found at {checkpoint_path}")
        return model

    print(f"[Checkpoint] Resuming from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw_sd = ckpt.get("state_dict", ckpt)
    model_sd = model.state_dict()
    renamed_sd = {}

    def rename_key(old_key):
        # remove common prefixes
        if old_key.startswith("module."):
            old_key = old_key[7:]
        if old_key.startswith("model."):
            old_key = old_key[6:]
        # add backbone prefix if needed
        if not old_key.startswith("backbone.") and ("backbone." + old_key in model_sd):
            return "backbone." + old_key
        if old_key.startswith("fc.") and ("backbone.fc" + old_key[2:] in model_sd):
            return "backbone." + old_key
        return old_key

    matched_keys, skipped_keys = [], []
    for ckpt_k, ckpt_v in raw_sd.items():
        new_k = rename_key(ckpt_k)
        if new_k in model_sd and ckpt_v.shape == model_sd[new_k].shape:
            renamed_sd[new_k] = ckpt_v
            matched_keys.append(f"{ckpt_k} -> {new_k}")
        else:
            skipped_keys.append(f"{ckpt_k}")

    print("Loading model...")
    result = model.load_state_dict(renamed_sd, strict=False)
    print("Missing keys:", result.missing_keys)
    print("Unexpected keys:", result.unexpected_keys)
    print("[Checkpoint] Skipped keys:", skipped_keys)

    return model


def load_qcw_model(
    checkpoint_path: str,
    depth: int = 18,
    whitened_layers: list[int] = None,
    act_mode: str = "pool_max",
    subspaces=None,
    num_classes: int = 200,
):
    """Build a QCW model, load checkpoint, wrap in DataParallel"""
    if whitened_layers is None:
        whitened_layers = []
    model = build_resnet_qcw(
        num_classes=num_classes,
        depth=depth,
        whitened_layers=whitened_layers,
        act_mode=act_mode,
        subspaces=subspaces
    )
    model = resume_checkpoint(model, checkpoint_path)
    model = nn.DataParallel(model).cuda()
    model.eval()
    return model


# Concept purity analysis
def compute_concept_purity_info(
    model: nn.Module,
    concept_dataset: ConceptDataset,
    batch_size: int = 64,
    device: str = "cuda"
):
    """Get axis activations and compute AUCs for each concept/subconcept"""
    sc_names = concept_dataset.get_subconcept_names()
    num_concepts = len(sc_names)

    # split labeled and free concepts
    labeled_idxs, free_idxs = [], []
    for i, nm in enumerate(sc_names):
        if concept_dataset.is_free_subconcept_name(nm):
            free_idxs.append(i)
        else:
            labeled_idxs.append(i)
    labeled_set = set(labeled_idxs)

    # hook into the last CW layer
    hook_storage = SimpleNamespace(output=None)
    last_cw = get_last_qcw_layer(model.module)

    def hook_fn(module, inp, out):
        with torch.no_grad():
            feat_avg = out.mean(dim=(2, 3))[:, :num_concepts]
            hook_storage.output = feat_avg.cpu()

    handle = last_cw.register_forward_hook(hook_fn)
    loader = DataLoader(concept_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    all_acts, all_labels = [], []
    for imgs, sc_label, _ in tqdm(loader, desc="[Purity] Gathering feats"):
        imgs = imgs.to(device)
        hook_storage.output = None
        _ = model(imgs)
        all_acts.append(hook_storage.output.numpy())
        all_labels.append(sc_label.numpy())

    handle.remove()
    all_acts = np.concatenate(all_acts, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    def roc_for_axis(pos_mask, axis_idx):
        # skip if all positive or all negative samples
        if pos_mask.sum() == 0 or pos_mask.sum() == len(pos_mask):
            return float("nan")
        return roc_auc_score(pos_mask.astype(int), all_acts[:, axis_idx])

    info_dict = {}

    # for labeled concepts, use matching axis
    for c_idx in labeled_idxs:
        scn = sc_names[c_idx]
        mask = (all_labels == c_idx)
        val = roc_for_axis(mask, c_idx)
        info_dict[scn] = {
            "is_free": False,
            "label_auc": val,
            "best_axis_auc": val,
            "best_axis_idx": c_idx,
            "best_axis_is_labeled": True,
            "unlabeled_axis_auc": None,
            "unlabeled_axis_idx": None
        }

    # for free concepts, find best activating axis
    for fc_idx in free_idxs:
        scn = sc_names[fc_idx]
        mask = (all_labels == fc_idx)
        if not mask.any():
            info_dict[scn] = {
                "is_free": True,
                "label_auc": None,
                "best_axis_auc": float("nan"),
                "best_axis_idx": -1,
                "best_axis_is_labeled": False,
                "unlabeled_axis_auc": None,
                "unlabeled_axis_idx": None
            }
            continue
        sub_acts = all_acts[mask]
        mean_acts = sub_acts.mean(axis=0)
        best_ax = int(np.argmax(mean_acts))
        best_auc = roc_for_axis(mask, best_ax)
        is_lbl = (best_ax in labeled_set)

        unl_auc, unl_idx = None, None
        if is_lbl:
            # If best axis is labeled, attempt a fallback axis
            sorted_ax = np.argsort(mean_acts)[::-1]
            for cand in sorted_ax:
                if cand not in labeled_set:
                    unl_idx = cand
                    unl_auc = roc_for_axis(mask, cand)
                    break

        info_dict[scn] = {
            "is_free": True,
            "label_auc": None,
            "best_axis_auc": best_auc,
            "best_axis_idx": best_ax,
            "best_axis_is_labeled": is_lbl,
            "unlabeled_axis_auc": unl_auc,
            "unlabeled_axis_idx": unl_idx
        }

    return info_dict


def plot_concept_purity(
    concept_info: dict,
    out_path: str = "concept_purity.png"
):
    """Plot bar chart showing concept purity - red=labeled axes, blue=unlabeled"""
    import math
    import matplotlib.patches as mpatches

    data_list = sorted(
        concept_info.items(),
        key=lambda x: x[1]["best_axis_auc"] if not math.isnan(x[1]["best_axis_auc"]) else -999,
        reverse=True
    )

    plt.figure(figsize=(12, 6))
    main_width, second_width = 0.5, 0.3

    def axis_color(is_labeled):
        return "red" if is_labeled else "blue"

    for i, (scn, info) in enumerate(data_list):
        # Potentially 2 bars: best + unlabeled fallback
        bar_specs = []
        best_val = info["best_axis_auc"]
        best_idx = info["best_axis_idx"]
        best_is_lbl = info["best_axis_is_labeled"]
        unl_val = info["unlabeled_axis_auc"]
        unl_idx = info["unlabeled_axis_idx"]

        if not math.isnan(best_val):
            bar_specs.append((best_val, best_idx, best_is_lbl, 0.0))
        if unl_val is not None and not math.isnan(unl_val):
            bar_specs.append((unl_val, unl_idx, False, 0.08))

        # Sort so we draw larger bar first
        bar_specs.sort(key=lambda b: b[0], reverse=True)

        for idxB, (height, axisid, is_lbl, x_offset) in enumerate(bar_specs):
            plt.bar(i + x_offset, height,
                    width=(main_width if idxB == 0 else second_width),
                    color=axis_color(is_lbl), zorder=2)

            label_txt = f"Ax {axisid} ({height:.2f})"
            plt.text(i + x_offset + (0.05 if idxB == 1 else 0),
                     height + 0.01,
                     label_txt,
                     ha="center", va="bottom", fontsize=8, zorder=3)

    plt.xticks(range(len(data_list)), [x[0] for x in data_list], rotation=45, ha="right")
    plt.ylabel("AUC (Concept Purity)")
    plt.title("Concept Purity")
    plt.ylim(0, 1)
    plt.tight_layout()

    # Legend
    labeled_patch = mpatches.Patch(color="red", label="Labeled Axis")
    unlabeled_patch = mpatches.Patch(color="blue", label="Unlabeled Axis")
    plt.legend(handles=[labeled_patch, unlabeled_patch], loc="upper right")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[Info] Saved concept purity chart => {out_path}")

# Top images per concept
def plot_topk_images_for_concept_axes(
    model: nn.Module,
    concept_dataset: ConceptDataset,
    k: int = 10,
    output_dir: str = "./topk_concept_images",
    batch_size: int = 64,
    device: str = "cuda",
    save_transformed: bool = False,
    unnorm_mean: list[float] = None,
    unnorm_std: list[float] = None
):
    """Save images that most activate each concept axis"""
    if unnorm_mean is None:
        unnorm_mean = [0.485, 0.456, 0.406]
    if unnorm_std is None:
        unnorm_std = [0.229, 0.224, 0.225]

    sc_names = concept_dataset.get_subconcept_names()
    sc_count = len(sc_names)
    labeled_idxs, free_idxs = [], []

    for i, scn in enumerate(sc_names):
        if concept_dataset.is_free_subconcept_name(scn):
            free_idxs.append(i)
        else:
            labeled_idxs.append(i)
    labeled_set = set(labeled_idxs)

    # hook into final layer
    hook_storage = SimpleNamespace(output=None)
    last_cw = get_last_qcw_layer(model.module)

    def hook_fn(module, inp, out):
        with torch.no_grad():
            feat_avg = out.mean(dim=(2,3))[:, :sc_count]
            hook_storage.output = feat_avg.cpu()

    h = last_cw.register_forward_hook(hook_fn)
    loader = DataLoader(concept_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    all_acts, all_labels = [], []
    all_imgs = [] if save_transformed else None
    all_paths = [] if not save_transformed else None

    for imgs, sc_label, paths in tqdm(loader, desc="[TopK] Gathering feats"):
        imgs = imgs.to(device)
        hook_storage.output = None
        _ = model(imgs)
        all_acts.append(hook_storage.output.numpy())
        all_labels.append(sc_label.numpy())

        if save_transformed:
            all_imgs.append(imgs.cpu())
        else:
            all_paths.extend(paths)

    h.remove()
    all_acts = np.concatenate(all_acts, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    def unnormalize_img(t):
        # unnormalize back to 0-1 range
        mean_t = torch.tensor(unnorm_mean, dtype=t.dtype).view(-1, 1, 1)
        std_t  = torch.tensor(unnorm_std, dtype=t.dtype).view(-1, 1, 1)
        out_t  = t.clone()
        out_t.mul_(std_t).add_(mean_t).clamp_(0, 1)
        return out_t

    if save_transformed:
        import torchvision
        all_imgs = torch.cat(all_imgs, dim=0)

    os.makedirs(output_dir, exist_ok=True)

    def best_unlabeled_axis(fc_idx):
        mask = (all_labels == fc_idx)
        if not mask.any():
            return None
        sub_acts = all_acts[mask]
        mean_acts = sub_acts.mean(axis=0)
        sorted_ax = np.argsort(mean_acts)[::-1]
        for ax_ in sorted_ax:
            if ax_ not in labeled_set:
                return ax_
        return None

    def copy_topk_for_axis(axis_idx, sc_name):
        axis_scores = all_acts[:, axis_idx]
        sorted_idx = np.argsort(axis_scores)[::-1][:k]
        sc_dir = os.path.join(output_dir, sc_name.replace("/", "_"))
        os.makedirs(sc_dir, exist_ok=True)
        for rank_i, arr_i in enumerate(sorted_idx):
            dst = os.path.join(sc_dir, f"rank_{rank_i+1}.jpg")
            if not save_transformed:
                src = all_paths[arr_i]
                try:
                    shutil.copy2(src, dst)
                except:
                    pass
            else:
                from torchvision.utils import save_image
                img_tensor = unnormalize_img(all_imgs[arr_i])
                save_image(img_tensor, dst)

    # process each subconcept
    for c_idx, sc_name in enumerate(sc_names):
        sc_dir_name = sc_name.replace("_free", "-free")
        if c_idx in labeled_idxs:
            copy_topk_for_axis(c_idx, sc_dir_name)
        else:
            axis_ul = best_unlabeled_axis(c_idx)
            if axis_ul is not None:
                copy_topk_for_axis(axis_ul, sc_dir_name)

    print(f"[Done] Copied top-{k} images per concept axis => {output_dir}")


# Main function
def main():
    """Run concept purity analysis and extract top images"""
    parser = argparse.ArgumentParser(description="QCW Plot Functions (Concept-Only Version)")

    parser.add_argument("--model_checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--concept_dir", default="", help="Folder with 'concept_val' subdir containing concept data.")
    parser.add_argument("--hl_concepts", default="", help="Comma separated high-level concepts for ConceptDataset.")
    parser.add_argument("--bboxes_file", default="", help="Optional bboxes.json for bounding boxes. Use if not in concept dataset")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--depth", type=int, default=18)
    parser.add_argument("--whitened_layers", type=str, default="5")
    parser.add_argument("--act_mode", type=str, default="pool_max")
    parser.add_argument("--num_classes", type=int, default=200)
    parser.add_argument("--vanilla_pretrain", action="store_true")
    parser.add_argument("--use_bn_qcw", action="store_true", help="If using BN->QCW with BN version.")
    parser.add_argument("--image_state", type=str, default="crop", help="(crop|redact|none) for ConceptDataset.")
    parser.add_argument("--run_purity", action="store_true", help="Compute concept purity & plot.")
    parser.add_argument("--topk_images", action="store_true", help="Extract top-k images per concept axis.")
    parser.add_argument("--k", type=int, default=10, help="Number of images to copy per axis.")
    parser.add_argument("--subspaces_json", default="", help="JSON describing subspaces if needed.")
    parser.add_argument("--save_transformed", action="store_true", help="If set, save normalized -> unnormalized images.")
    parser.add_argument("--output_dir", default="qcw_plots", help="Directory for output images/plots.")

    args = parser.parse_args()

    # handle BN->QCW alternative
    if args.use_bn_qcw:
        global build_resnet_qcw, get_last_qcw_layer
        from MODELS.model_resnet_qcw_bn import build_resnet_qcw as bn_build
        from MODELS.model_resnet_qcw_bn import get_last_qcw_layer as bn_get_last
        build_resnet_qcw = bn_build
        get_last_qcw_layer = bn_get_last

    try:
        wl = [int(x) for x in args.whitened_layers.split(",") if x.strip()]
    except ValueError:
        wl = []

    subspaces = None
    if args.subspaces_json and os.path.isfile(args.subspaces_json):
        with open(args.subspaces_json) as f:
            subspaces = json.load(f)

    print("[Info] Loading QCW model...")
    model = load_qcw_model(
        checkpoint_path=args.model_checkpoint,
        depth=args.depth,
        whitened_layers=wl,
        act_mode=args.act_mode,
        subspaces=subspaces,
        num_classes=args.num_classes
    )

    # process concept dataset if provided
    if args.concept_dir and os.path.isdir(args.concept_dir):
        hl_list = [x.strip() for x in args.hl_concepts.split(",")] if args.hl_concepts else []
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        bboxes = args.bboxes_file or os.path.join(args.concept_dir, "bboxes.json")
        concept_ds = ConceptDataset(
            root_dir=os.path.join(args.concept_dir, "concept_val"),
            bboxes_file=bboxes,
            high_level_filter=hl_list,
            transform=transform,
            crop_mode=args.image_state
        )

        # run purity analysis
        if args.run_purity:
            info_dict = compute_concept_purity_info(model, concept_ds, batch_size=args.batch_size)
            out_path = os.path.join(args.output_dir, "concept_purity.png")
            plot_concept_purity(info_dict, out_path)

        # extract top activating images
        if args.topk_images:
            out_dir = os.path.join(args.output_dir, "topk_concept_images")
            plot_topk_images_for_concept_axes(
                model, concept_ds,
                k=args.k,
                output_dir=out_dir,
                batch_size=args.batch_size,
                save_transformed=args.save_transformed
            )

    print("[Done] QCW concept analysis complete.")

if __name__ == "__main__":
    main()