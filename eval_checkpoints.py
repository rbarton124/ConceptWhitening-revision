#!/usr/bin/env python3
"""
Evaluate a directory of QCW / vanilla checkpoints.

For each *.pth file we will
    1.  Parse the filename to infer
            • backbone           (resnet / densenet / vgg16)
            • depth              (18 / 50 …)
            • whitened_layers    (list[int] or [])
            • concept group      (LGR, SML, FREE, NA)
    2.  Instantiate the model via MODELS.factory_qcw.build_qcw
    3.  Load weights  (robust remapping à-la train_qcw)
    4.  Measure *train*, *val* and *test* top-1 accuracy
    5.  Append one row to results.csv

Verbose prints show:
    • parsed parameters
    • #matched / skipped / missing keys on load
    • per-split accuracy
    • running timer per checkpoint

Typical call
------------
python eval_checkpoints.py \
        --ckpt_dir  model_checkpoints \
        --data_dir  data/CUB/main_data \
        --workers   8 \
        --batch_size 128
"""
import argparse, os, re, time, csv, json, sys, math
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---- QCW imports -----------------------------------------------------------
from MODELS.factory_qcw import build_qcw           # <-- new factory
# ----------------------------------------------------------------------------

# ───────────────────────────────── Argument parsing ──────────────────────────────────
def get_args():
    p = argparse.ArgumentParser(description="QCW checkpoint evaluator")
    p.add_argument("--ckpt_dir",  required=True,
                   help="Directory containing *.pth checkpoints.")
    p.add_argument("--data_dir",  required=True,
                   help="Main CUB directory with train/val/test sub-folders.")
    p.add_argument("--workers",   type=int, default=4)
    p.add_argument("--batch_size",type=int, default=64)
    p.add_argument("--csv_out",   default="eval_results.csv")
    p.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

# ───────────────────────────────── Dataset helpers ───────────────────────────────────
def build_loaders(data_root, batch_size, workers):
    tr = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    splits = {}
    for split in ["train","val","test"]:
        path = os.path.join(data_root, split)
        splits[split] = DataLoader(
            dset.ImageFolder(path, tr),
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True
        )
    return splits

@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        logits = model(x)
        preds  = logits.argmax(1)
        correct += (preds==y).sum().item()
        total   += y.size(0)
    return 100.0 * correct / total

# ───────────────────────────────── Checkpoint name parser ────────────────────────────
def parse_filename(fname:str):
    """
    Deduces model-type, depth, WL index list, concept group.
    Returns dict with safe defaults; unknown patterns → type='skip'.
    """
    base = Path(fname).stem
    out  = dict(model_type=None, depth=None, whitened_layers=[],
                concept_group="NA", act_mode="pool_max")

    # QCW pattern ─────────────────────────────
    m = re.match(r"QCW(\d+)_([A-Z]+)(?:_.+)?_WL_(\d+)", base)
    if m:
        depth   = int(m.group(1))          # 18 / 50
        group   = m.group(2)               # LGR / SML / FREE / ...
        wl      = int(m.group(3))
        out.update(model_type="resnet",
                   depth=depth,
                   concept_group=group,
                   whitened_layers=[wl])
        return out

    # Vanilla ResNet e.g. res50_CUB.pth, res18_CUB.pth
    m = re.match(r"res(\d+).*", base)
    if m:
        depth = int(m.group(1))
        out.update(model_type="resnet", depth=depth)
        return out

    # DenseNet e.g. densenet161_CUB_best.pth
    m = re.match(r"densenet(\d+).*", base)
    if m:
        depth = int(m.group(1))
        out.update(model_type="densenet", depth=depth)
        return out

    # VGG
    if base.startswith("vgg16"):
        out.update(model_type="vgg16", depth=None)
        return out

    # Unknown
    out["model_type"] = "skip"
    return out

# ───────────────────────────────── Weight loader ─────────────────────────────────────
def smart_load(model, ckpt_path, verbose=True):
    sd_target = model.state_dict()
    ckpt      = torch.load(ckpt_path, map_location="cpu")
    raw_sd    = ckpt.get("state_dict", ckpt)

    def rename_key(k):
        if k.startswith("module."): k = k[7:]
        # DenseNet conv/norm patch
        k = re.sub(r"\.norm\.(\d+)", lambda m: f".norm{m.group(1)}", k)
        k = re.sub(r"\.conv\.(\d+)", lambda m: f".conv{m.group(1)}", k)
        if not k.startswith("backbone.") and ("backbone."+k in sd_target):
            return "backbone."+k
        if k.startswith("fc.") and ("backbone.fc"+k[2:] in sd_target):
            return "backbone."+k
        return k

    matched, skipped = 0,0
    load_dict = {}
    for k,v in raw_sd.items():
        nk = rename_key(k)
        if nk in sd_target and v.shape == sd_target[nk].shape:
            load_dict[nk] = v ; matched+=1
        else:
            skipped += 1
    miss = len([k for k in sd_target if k not in load_dict])

    model.load_state_dict(load_dict, strict=False)
    if verbose:
        print(f"      ↳ matched={matched}  skipped={skipped}  missing={miss}")
    return matched, skipped, miss

# ───────────────────────────────────────── Main ──────────────────────────────────────
def main():
    args   = get_args()
    device = torch.device(args.device)
    loaders= build_loaders(args.data_dir, args.batch_size, args.workers)

    results = []
    t_total = time.time()

    ckpts = sorted([f for f in Path(args.ckpt_dir).glob("*.pth")])
    if not ckpts:
        print("No .pth files found in", args.ckpt_dir)
        sys.exit(1)

    print(f"Found {len(ckpts)} checkpoints.")
    for ck in ckpts:
        print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"Processing  {ck.name}")
        meta = parse_filename(ck.name)
        if meta["model_type"] == "skip":
            print("   » pattern not recognised → skipped")
            continue
        print("   » parsed:", json.dumps(meta, indent=4))

        # build model ---------------------------------------------------------
        model = build_qcw(
            model_type   = meta["model_type"],
            num_classes  = 200,
            depth        = meta["depth"],
            whitened_layers = meta["whitened_layers"],
            act_mode     = meta["act_mode"],
            subspaces    = {},          # eval only
            use_subspace = True,
            use_free     = False,
            cw_lambda    = 0.05,
            pretrained_model=None,
            vanilla_pretrain = (len(meta["whitened_layers"]) == 0)
        )
        model.to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        # load weights --------------------------------------------------------
        smart_load(model, ck, verbose=True)

        # evaluate ------------------------------------------------------------
        accs = {}
        for split, loader in loaders.items():
            st = time.time()
            accs[split] = accuracy(model, loader, device)
            print(f"   [{split:5}]  {accs[split]:6.2f}%  ({time.time()-st:5.1f}s)")

        # collect row ---------------------------------------------------------
        results.append({
            "checkpoint"     : ck.name,
            "model_type"     : meta["model_type"],
            "depth"          : meta["depth"],
            "whitened_layers": ",".join(map(str, meta["whitened_layers"])) or "None",
            "concept_group"  : meta["concept_group"],
            "data_dir"       : args.data_dir,
            "train_top1"     : f"{accs['train']:.2f}",
            "val_top1"       : f"{accs['val']:.2f}",
            "test_top1"      : f"{accs['test']:.2f}",
        })

    # write CSV ---------------------------------------------------------------
    keys = ["checkpoint","model_type","depth","whitened_layers",
            "concept_group","data_dir","train_top1","val_top1","test_top1"]
    with open(args.csv_out, "w", newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)

    print("\n════════════════════════════════════════════════════")
    print(f"Wrote {len(results)} rows to  {args.csv_out}")
    print(f"Total runtime: {time.time()-t_total:.1f} s")

if __name__ == "__main__":
    main()
