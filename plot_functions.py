import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import shutil
from types import SimpleNamespace

from MODELS.model_resnet_qcw import build_resnet_qcw, get_last_qcw_layer
from MODELS.ConceptDataset_QCW import ConceptDataset

def resume_checkpoint(model, checkpoint_path):
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        print(f"[Checkpoint] No checkpoint found at {checkpoint_path}")
        return

    print(f"[Checkpoint] Resuming from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    raw_sd = ckpt.get("state_dict", ckpt)
    model_sd = model.state_dict()
    renamed_sd = {}

    def rename_key(old_key):
        if old_key.startswith("module."):
            old_key = old_key[len("module."):]
        if old_key.startswith("model."):
            old_key = old_key[len("model."):]
        if not old_key.startswith("backbone.") and ("backbone."+old_key in model_sd):
            return "backbone."+old_key
        if old_key.startswith("fc.") and ("backbone.fc"+old_key[2:] in model_sd):
            return "backbone."+old_key
        return old_key
    
    matched_keys = []
    skipped_keys = []
    
    for ckpt_k, ckpt_v in raw_sd.items():
        new_k = rename_key(ckpt_k)
        if new_k in model_sd:
            if ckpt_v.shape == model_sd[new_k].shape:
                renamed_sd[new_k] = ckpt_v
                matched_keys.append(f"{ckpt_k} -> {new_k}")
            else:
                skipped_keys.append(f"{ckpt_k}: shape {ckpt_v.shape} != {model_sd[new_k].shape}")
        else:
            skipped_keys.append(f"{ckpt_k}: no match found in model")
    
    print("Loading model...")
    result = model.load_state_dict(renamed_sd, strict=False)
    print("Missing keys:", result.missing_keys)
    print("Unexpected keys:", result.unexpected_keys)
    print("[Checkpoint] Skipped keys from checkpoint:")
    for sk in skipped_keys:
        print("   ", sk)

    return model

def tensor_to_pil(
    tensor: torch.Tensor, 
    mean: list[float] = [0.485, 0.456, 0.406],
    std: list[float]  = [0.229, 0.224, 0.225]
):
    """
    Convert a 3D PyTorch tensor (C,H,W) to a PIL image.
    If your dataset is normalized, pass the same mean & std used at transform time.
    """
    # Unnormalize
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    # clamp to [0,1] range
    tensor.clamp_(0, 1)
    # convert CHW -> HWC
    np_img = tensor.permute(1,2,0).cpu().numpy()
    # to PIL
    return transforms.functional.to_pil_image(np_img)

def load_qcw_model(
    checkpoint_path: str, 
    depth: int = 18,
    whitened_layers: list[int] = [],
    act_mode: str = "pool_max",
    subspaces=None,
    num_classes: int = 200,
):
    model = build_resnet_qcw(num_classes=num_classes, depth=depth, whitened_layers=whitened_layers, act_mode=act_mode, subspaces=subspaces)

    model = resume_checkpoint(model, checkpoint_path)

    model = nn.DataParallel(model).cuda()
    model.eval()
    return model

def compute_concept_purity_auc(
    model: nn.Module,
    concept_dataset: ConceptDataset,
    batch_size: int = 64,
    device: str = "cuda",
    activation_mode: str = "pool_max"
):
    """
    Compute concept purity AUC for each sub-concept axis, 
    ignoring any axes beyond [0..num_subconcepts-1].

    Implementation:
      1) We do a single pass over concept_dataset, capturing final QCW layer's features
         (mean over spatial dims).
      2) For sub-concept c, we treat images of c as positives, all others as negatives,
         then compute ROC-AUC for the c-th axis (since c's axis = c).
    """
    model.eval()
    loader = DataLoader(concept_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    sc_names = concept_dataset.get_subconcept_names()  # sub-concepts sorted by sc2idx
    num_concepts = len(sc_names)

    # We'll store [N, num_concepts] for the final-later features
    all_activations = []
    all_labels = []

    last_cw = get_last_qcw_layer(model.module)
    hook_storage = SimpleNamespace(output=None)

    def hook_fn(module, inp, out):
        # out shape: [B, C, H, W]
        with torch.no_grad():
            # We'll do a mean over spatial dims => [B, C]
            # Then only keep columns up to num_concepts
            feat_avg = out.mean(dim=(2,3))[:, :num_concepts]  # skip channels >= num_concepts
            hook_storage.output = feat_avg.cpu()

    handle = last_cw.register_forward_hook(hook_fn)

    # Gather data
    for imgs, sc_label, _ in tqdm(loader, desc="Computing concept AUC"):
        imgs = imgs.to(device, non_blocking=True)
        hook_storage.output = None
        _ = model(imgs)
        if hook_storage.output is None:
            raise RuntimeError("QCW Hook did not capture output. Check hooking logic.")
        batch_activations = hook_storage.output.numpy()  # shape [B, num_concepts]
        all_activations.append(batch_activations)
        all_labels.append(sc_label.numpy())

    handle.remove()
    all_activations = np.concatenate(all_activations, axis=0)  # shape [N, num_concepts]
    all_labels = np.concatenate(all_labels, axis=0)            # shape [N]

    # For each sub-concept c in [0..num_concepts-1]:
    #   binary_label = (all_labels == c)
    #   predicted score = all_activations[:,c]
    auc_values = {}
    for c_idx, sc_name in enumerate(sc_names):
        binary_label = (all_labels == c_idx).astype(int)
        score = all_activations[:, c_idx]
        if len(np.unique(binary_label)) < 2:
            auc_val = float("nan")
        else:
            auc_val = roc_auc_score(binary_label, score)
        auc_values[sc_name] = auc_val
    return auc_values

def plot_concept_purity_histogram(
    auc_values: dict, 
    out_path: str = "concept_purity_hist.png"
):
    """
    Plots a bar chart of the concept purity AUC for sub-concepts, 
    sorted descending by AUC. 
    Creates the out_path folder if needed.
    """
    import matplotlib.pyplot as plt
    import os

    sc_names = list(auc_values.keys())
    sc_aucs = [auc_values[n] for n in sc_names]
    sorted_indices = np.argsort(sc_aucs)[::-1]  # descending
    sorted_names = [sc_names[i] for i in sorted_indices]
    sorted_aucs = [sc_aucs[i] for i in sorted_indices]

    plt.figure(figsize=(10,5))
    plt.bar(range(len(sorted_aucs)), sorted_aucs, color="blue")
    plt.xticks(range(len(sorted_aucs)), sorted_names, rotation=90)
    plt.ylim([0, 1.0])
    plt.ylabel("AUC (Concept Purity)")
    plt.title("Concept Purity across sub-concepts")
    plt.tight_layout()

    # Ensure directory
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    plt.savefig(out_path)
    plt.close()
    print(f"[Info] Saved concept purity histogram: {out_path}")


def plot_topk_images_for_concept_axes(model: nn.Module, concept_dataset: ConceptDataset, k: int = 10,
    output_dir: str = "./topk_concept_images", batch_size: int = 64, device: str = "cuda", save_transformed: bool = False,
    unnorm_mean: list[float] = [0.485, 0.456, 0.406], unnorm_std: list[float]  = [0.229, 0.224, 0.225]):

    model.eval()
    loader = DataLoader(concept_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    sc_names = concept_dataset.get_subconcept_names()
    sc_count = len(sc_names)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Hook the final QCW layer
    last_cw = get_last_qcw_layer(model.module)
    hook_storage = SimpleNamespace(output=None)

    def hook_fn(module, inp, out):
        # out: [B, C, H, W]
        with torch.no_grad():
            feat_avg = out.mean(dim=(2,3))[:, :sc_count]  # keep only sub-concept axes
            hook_storage.output = feat_avg.cpu()

    h = last_cw.register_forward_hook(hook_fn)

    all_activations = []
    if save_transformed:
        # We'll store the entire transformed images in memory
        # plus we still store something for indexing them properly
        all_images = []
    else:
        # We'll store file paths
        all_paths = []

    # Inference over concept_dataset
    idx_offset = 0
    for imgs, sc_label, paths in loader:
        imgs = imgs.to(device, non_blocking=True)
        hook_storage.output = None
        _ = model(imgs)
        feats = hook_storage.output.numpy()  # shape [B, sc_count]

        all_activations.append(feats)

        if save_transformed:
            # keep the CPU version of these images
            # (what the model saw, i.e. cropped, redacted, etc.)
            all_images.append(imgs.cpu())  # shape [B,C,H,W], in normalization space
        else:
            # store the original disk file paths
            all_paths.extend(paths)

        idx_offset += feats.shape[0]

    h.remove()

    all_activations = np.concatenate(all_activations, axis=0)  # shape [N, sc_count]
    # unify images or paths
    if save_transformed:
        all_images = torch.cat(all_images, dim=0)  # shape [N, C,H,W]

        # We'll define a small function to unnormalize + clamp => return a CPU image
        import torchvision.transforms.functional as F_vision

        def unnormalize_img(t):
            # t shape [C,H,W]
            # unnormalize -> clamp -> return
            mean_t = torch.tensor(unnorm_mean, dtype=t.dtype, device=t.device).view(-1,1,1)
            std_t  = torch.tensor(unnorm_std,  dtype=t.dtype, device=t.device).view(-1,1,1)
            out_t = t.clone()
            out_t.mul_(std_t).add_(mean_t).clamp_(0,1)
            return out_t

    # For each sub-concept axis c
    for c_idx, sc_name in enumerate(sc_names):
        axis_scores = all_activations[:, c_idx]
        sorted_idx = np.argsort(axis_scores)[::-1]  # descending
        topk_idx = sorted_idx[:k]

        sc_dir = os.path.join(output_dir, sc_name.replace("/", "_"))
        os.makedirs(sc_dir, exist_ok=True)

        for rank_i, arr_i in enumerate(topk_idx):
            # Save either from file or from memory
            if not save_transformed:
                # 1) copy from disk
                src_path = all_paths[arr_i]
                dest_path = os.path.join(sc_dir, f"rank_{rank_i+1}.jpg")
                try:
                    shutil.copy2(src_path, dest_path)
                except:
                    pass

            else:
                # 2) save from the dataset-transformed tensor
                import torchvision.transforms.functional as TF
                # all_images[arr_i] is shape [C,H,W], normalized
                img_tensor = unnormalize_img(all_images[arr_i])
                dest_path = os.path.join(sc_dir, f"rank_{rank_i+1}.jpg")
                # Now we can use torchvision.utils.save_image
                from torchvision.utils import save_image
                save_image(img_tensor, dest_path)

    print(f"[Done] Copied top-{k} images (save_transformed={save_transformed}) "
          f"for each sub-concept axis into: {output_dir}")

#####################################
# Extra: main_data_subconcepts logic
#####################################

def evaluate_main_dataset_subconcepts(
    model: nn.Module,
    main_dataset_folder: str,
    main_data_subconcepts_json: str,
    concept_dataset: ConceptDataset,
    batch_size: int = 64,
    device: str = "cuda",
    out_plot_file: str = "main_data_subconcept_auc.png"
):
    """
    For multi-labeled main dataset images. 
    We'll only test sub-concept axes in [0..concept_dataset.get_num_subconcepts()-1].
    Then do a multi-label AUC approach.

    main_data_subconcepts_json => "image_relative_path" : ["subconceptA", "subconceptB", ...]

    We assume the same sub-concept names are used as in concept_dataset 
    so we can reuse concept_dataset.sc2idx or sc_names.
    """
    import os
    from torchvision.datasets import ImageFolder
    import matplotlib.pyplot as plt

    sc_names = concept_dataset.get_subconcept_names()
    sc2idx = {n:i for i,n in enumerate(sc_names)}
    num_concepts = len(sc_names)

    # Load the JSON
    with open(main_data_subconcepts_json, "r") as f:
        sc_mapping = json.load(f)  # e.g. { "some_class/0001.jpg": ["eye_color::brown","nape_shape::short"], ...}

    # We'll define a wrapper dataset
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # Same normalization as your training
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    imgfolder = ImageFolder(main_dataset_folder, transform=transform)

    class MainDataMultiLabel(data.Dataset):
        def __init__(self, folder_dataset, sc_map, sc2idx):
            self.ds = folder_dataset
            self.samples = folder_dataset.samples
            self.root = folder_dataset.root
            self.sc_map = sc_map
            self.sc2idx = sc2idx
            self.sc_count = len(sc2idx)

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, _ = self.samples[idx]
            rel_path = os.path.relpath(path, self.root)
            # Load
            img = self.ds.loader(path)
            if self.ds.transform is not None:
                img = self.ds.transform(img)
            # Build multi-hot label
            label_vec = np.zeros(self.sc_count, dtype=np.float32)
            if rel_path in self.sc_map:
                subcs = self.sc_map[rel_path]
                for sname in subcs:
                    if sname in self.sc2idx:
                        label_vec[self.sc2idx[sname]] = 1.0
            return img, label_vec, path

    main_ds = MainDataMultiLabel(imgfolder, sc_mapping, sc2idx)
    main_loader = DataLoader(main_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Hook final QCW
    model.eval()
    last_cw = get_last_qcw_layer(model.module)
    hook_storage = SimpleNamespace(output=None)
    def hook_fn(module, inp, out):
        with torch.no_grad():
            # We only keep up to num_concepts
            feat_avg = out.mean(dim=(2,3))[:, :num_concepts]
            hook_storage.output = feat_avg.cpu()

    h = last_cw.register_forward_hook(hook_fn)

    all_acts = []
    all_labels = []
    all_paths = []
    for imgs, label_vecs, paths in tqdm(main_loader, desc="Main data multi-label analysis"):
        imgs = imgs.to(device)
        hook_storage.output = None
        _ = model(imgs)
        feats = hook_storage.output.numpy()  # shape [B, num_concepts]
        all_acts.append(feats)
        all_labels.append(label_vecs.numpy())
        all_paths.extend(paths)

    h.remove()
    all_acts = np.concatenate(all_acts, axis=0)    # [N, num_concepts]
    all_labels = np.concatenate(all_labels, axis=0) # [N, num_concepts]

    # compute AUC for each sub-concept axis
    auc_list = []
    for c_idx, sc_name in enumerate(sc_names):
        y_true = all_labels[:, c_idx]
        y_score = all_acts[:, c_idx]
        if (y_true.sum() == 0) or (y_true.sum() == len(y_true)):
            auc_val = float("nan")
        else:
            auc_val = roc_auc_score(y_true, y_score)
        auc_list.append(auc_val)

    # Plot
    plt.figure(figsize=(10,5))
    sorted_idx = np.argsort(auc_list)[::-1]
    sorted_names = [sc_names[i] for i in sorted_idx]
    sorted_aucs  = [auc_list[i]  for i in sorted_idx]
    plt.bar(range(len(sorted_idx)), sorted_aucs, color='b')
    plt.xticks(range(len(sorted_idx)), sorted_names, rotation=90)
    plt.ylim([0,1])
    plt.title("Main Data Multi-Label Subconcept AUC")
    plt.tight_layout()

    out_dir = os.path.dirname(out_plot_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    plt.savefig(out_plot_file)
    plt.close()
    print(f"[Info] Saved main data multi-subconcept AUC to {out_plot_file}")

    # Return dictionary
    return dict(zip(sc_names, auc_list))

###############################################
# CLI for standalone usage
###############################################
def main():
    parser = argparse.ArgumentParser(description="QCW Plot Functions - Updated")

    parser.add_argument("--model_checkpoint", required=True, type=str, help="Path to model checkpoint.")
    parser.add_argument("--data_dir", default="", help="If you want to test main dataset.")
    parser.add_argument("--concept_dir", default="", help="If you want concept-based analysis.")
    parser.add_argument("--hl_concepts", default="", help="Comma separated high-level concepts for ConceptDataset.")
    parser.add_argument("--bboxes_file", default="", help="Optional bboxes.json for ConceptDataset.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--depth", type=int, default=18)
    parser.add_argument("--whitened_layers", type=str, default="5")
    parser.add_argument("--act_mode", type=str, default="pool_max")
    parser.add_argument("--num_classes", type=int, default=200)
    parser.add_argument("--vanilla_pretrain", action="store_true")
    parser.add_argument("--use_bn_qcw", action="store_true", help="Replace BN with QCW inside ResNet blocks")
    parser.add_argument("--image_state", type=str, default="crop", help="Image state (crop, redact, none)")

    # which analysis
    parser.add_argument("--run_purity", action="store_true", help="Compute concept purity AUC on concept dataset.")
    parser.add_argument("--topk_images", action="store_true", help="Copy top-k images for each sub-concept axis.")
    parser.add_argument("--k", type=int, default=10, help="k for top-k.")
    parser.add_argument("--subspaces_json", default="", help="If you have a JSON describing subspaces.")
    parser.add_argument("--save_transformed", action="store_true", help="Save transformed images.")
    
    # main data multi-subconcept
    parser.add_argument("--main_data_subconcepts", default="", help="JSON with multi-subconcept labels for main data.")
    parser.add_argument("--evaluate_main_subconcepts", action="store_true", help="Run multi-subconcept AUC on main data.")
    parser.add_argument("--output_dir", default="qcw_plots", help="Where to save plots, top-k images, etc.")

    args = parser.parse_args()
    
    if args.use_bn_qcw:
        global build_resnet_qcw, get_last_qcw_layer
        from MODELS.model_resnet_qcw_bn import build_resnet_qcw as bn_build, get_last_qcw_layer as bn_get_last
        build_resnet_qcw = bn_build
        get_last_qcw_layer = bn_get_last

    # parse whitened_layers
    try:
        wl = [int(x) for x in args.whitened_layers.split(",") if x.strip() != ""]
    except:
        wl = []

    # Possibly load subspaces from JSON
    subspaces = None
    if args.subspaces_json and os.path.isfile(args.subspaces_json):
        with open(args.subspaces_json,"r") as f:
            subspaces = json.load(f)

    print("[Info] Building/loading model from checkpoint...")
    model = load_qcw_model(
        checkpoint_path=args.model_checkpoint,
        depth=args.depth,
        whitened_layers=wl,
        act_mode=args.act_mode,
        subspaces=subspaces,
        num_classes=args.num_classes
    )

    # If we have concept_dir for concept dataset
    if args.concept_dir and os.path.isdir(args.concept_dir):
        # Build concept dataset
        hl_list = [h.strip() for h in args.hl_concepts.split(",")] if args.hl_concepts else []
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        bboxes = args.bboxes_file or os.path.join(args.concept_dir, "bboxes.json")
        concept_ds = ConceptDataset(
            root_dir=os.path.join(args.concept_dir, "concept_val"), 
            bboxes_file=bboxes,
            high_level_filter=hl_list,
            transform=transform,
            crop_mode=args.image_state,
        )

        if args.run_purity:
            print("[Info] Computing concept purity AUC (sub-concept axes only).")
            auc_vals = compute_concept_purity_auc(
                model=model,
                concept_dataset=concept_ds,
                batch_size=args.batch_size
            )
            out_hist = os.path.join(args.output_dir, "concept_purity_hist.png")
            plot_concept_purity_histogram(auc_vals, out_hist)

        if args.topk_images:
            print("[Info] Gathering top-k images for each sub-concept axis.")
            out_dir = os.path.join(args.output_dir, "topk_concept_images")
            plot_topk_images_for_concept_axes(
                model, concept_ds,
                k=args.k,
                output_dir=out_dir,
                batch_size=args.batch_size,
                save_transformed=args.save_transformed
            )

    # If we want to evaluate main data multi-labeled with sub-concepts
    if args.evaluate_main_subconcepts and args.data_dir and os.path.isdir(args.data_dir):
        if not args.main_data_subconcepts or not os.path.isfile(args.main_data_subconcepts):
            print("[Error] You must supply a valid --main_data_subconcepts JSON for multi-label.")
        else:
            # We need a concept dataset instance to get sub-concept names & indices
            if not (args.concept_dir and os.path.isdir(args.concept_dir)):
                print("[Error] We need an existing concept dataset for sub-concept definitions to map sc2idx.")
                return
            # build a concept dataset to parse sub-concepts
            hl_list = [h.strip() for h in args.hl_concepts.split(",")] if args.hl_concepts else []
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])
            bboxes = args.bboxes_file or os.path.join(args.concept_dir, "bboxes.json")
            concept_ds = ConceptDataset(
                root_dir=os.path.join(args.concept_dir, "concept_train"),
                bboxes_file=bboxes,
                high_level_filter=hl_list,
                transform=transform,
                crop_mode="crop"
            )
            out_plot = os.path.join(args.output_dir, "main_data_subconcept_auc.png")
            evaluate_main_dataset_subconcepts(
                model=model,
                main_dataset_folder=os.path.join(args.data_dir,"val"), # or "test"
                main_data_subconcepts_json=args.main_data_subconcepts,
                concept_dataset=concept_ds,
                batch_size=args.batch_size,
                out_plot_file=out_plot
            )

    print("[Done] All requested plot functions are complete.")

if __name__ == "__main__":
    main()
