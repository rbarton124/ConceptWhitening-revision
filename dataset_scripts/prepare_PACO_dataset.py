#!/usr/bin/env python3
import argparse, json, os, re, unicodedata
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageOps
import shutil

def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii","ignore").decode("ascii")
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+","_",s).strip("_")
    return s or "unknown"

def load_json(p):
    with open(p,"r") as f:
        print(f"loading json...") 
        return json.load(f)

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def clamp_box(x,y,w,h,W,H,margin=0.12):
    x2,y2=x+w,y+h
    dx,dy=w*margin,h*margin
    x=max(0,x-dx); y=max(0,y-dy)
    x2=min(W,x2+dx); y2=min(H,y2+dy)
    x=int(round(x)); y=int(round(y)); w=int(round(x2-x)); h=int(round(y2-y))
    if w<=0 or h<=0: return None
    return [x,y,w,h]

def crop_and_save(img_path, box, out_path):
    with Image.open(img_path).convert("RGB") as im:
        W,H=im.size
        b=clamp_box(*box,W,H,margin=0.12)
        if b is None: return False
        x,y,w,h=b
        crop=im.crop((x,y,x+w,y+h))
        s=256
        r=s/min(crop.size)
        crop=crop.resize((int(crop.size[0]*r),int(crop.size[1]*r)), Image.BICUBIC)
        cw,ch=crop.size
        lx=max(0,(cw-224)//2); ty=max(0,(ch-224)//2)
        rx=min(cw,lx+224); by=min(ch,ty+224)
        crop=crop.crop((lx,ty,rx,by))
        ensure_dir(Path(out_path).parent)
        crop.save(out_path, quality=95)
        return True

def add_padding(img, pad=16, fill=(0,0,0)):
    if isinstance(pad, int):
        pad = (pad, pad, pad, pad)
    return ImageOps.expand(img, border=pad, fill=fill)

def letterbox_to(img, target_size=(224,224), fill=(0,0,0), scale_up=True, resample=Image.BICUBIC):
    tw, th = target_size
    w, h = img.size
    r = min(tw / w, th / h)
    if not scale_up:
        r = min(1.0, r)
    nw, nh = max(1, int(round(w * r))), max(1, int(round(h * r)))
    img = img.resize((nw, nh), resample)
    out = Image.new("RGB", (tw, th), fill)
    out.paste(img, ((tw - nw) // 2, (th - nh) // 2))
    return out

def crop_and_save_with_padding(img_path, box, out_path, pad_px=0, pad_color=(0,0,0), letterbox=False, target=224):
    with Image.open(img_path).convert("RGB") as im:
        W,H = im.size
        x,y,w,h = [int(round(v)) for v in box]
        x = max(0, x); y = max(0, y)
        w = max(1, min(w, W - x)); h = max(1, min(h, H - y))
        crop = im.crop((x, y, x + w, y + h))
        if letterbox:
            out = letterbox_to(crop, (target, target), fill=pad_color)
        else:
            s = 256
            r = s / min(crop.size)
            crop = crop.resize((int(crop.size[0]*r), int(crop.size[1]*r)), Image.BICUBIC)
            cw, ch = crop.size
            lx = max(0, (cw - target) // 2); ty = max(0, (ch - target) // 2)
            out = crop.crop((lx, ty, lx + target, ty + target))
        if pad_px > 0:
            out = add_padding(out, pad=pad_px, fill=pad_color)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        out.save(out_path, quality=95)
        return True

def build_idx_map(categories):
    ids=sorted([c["id"] for c in categories])
    return {cid:f"{i:03d}" for i,cid in enumerate(ids,1)}

def part_table_key(d):
    for k in ["obj_part_categories","object_part_categories","part_categories"]:
        if k in d and isinstance(d[k],list): return k
    return None

def part_id_key(anns):
    keys=set()
    for a in anns[:2000]:
        for k,v in a.items():
            if isinstance(v,int) and "part" in k and "id" in k: keys.add(k)
    for k in ["obj_part_id","object_part_id","part_category_id"]: 
        if k in keys: return k
    return next(iter(keys), None)

def split_to_folder(split):
    return "train2017" if split=="train" else "val2017"

def resolve_image_path(image_dir, file_name):
    path = os.path.join(image_dir, file_name)
    if os.path.exists(path): return path
    return None

def link_file(src, dst, mode="symlink"):
    ensure_dir(Path(dst).parent)
    if os.path.exists(dst): return True
    try:
        if mode == "copy":
            shutil.copy2(src, dst)
        elif mode == "hardlink":
            os.link(src, dst)
        else:
            os.symlink(src, dst)
        return True
    except Exception as e:
        print(f"link failed: {e}")
        return False

def build_coco_main_no_crop(split, data, image_dir, out_root, link_mode):
    print("building main (no crop)...")
    id2img={im["id"]:im["file_name"] for im in data["images"]}
    id2cat={c["id"]:c.get("name",str(c["id"])) for c in data["categories"]}
    seen = set()
    counts = defaultdict(int)
    for ann in data["annotations"]:
        img_id=ann.get("image_id"); cat_id=ann.get("category_id")
        if img_id not in id2img or cat_id not in id2cat: continue
        key=(cat_id,img_id)
        if key in seen: continue
        seen.add(key)
        img_path = resolve_image_path(image_dir, id2img[img_id])
        if not img_path:
            print(f"missing image: {id2img[img_id]}")
            continue
        obj_name = norm(id2cat[cat_id])
        counts[obj_name]+=1
        dst = os.path.join(out_root, split, obj_name, f"{counts[obj_name]}.jpg")
        if link_file(img_path, dst, mode=link_mode):
            print(f"linked {img_path} -> {dst}")

def build_part_name_map(categories, normalizer=lambda s: s):
    objs = {c["name"]: c["id"] for c in categories if c.get("supercategory") == "OBJECT"}
    objs_norm = {normalizer(k): k for k in objs.keys()}
    part_map = {}
    for c in categories:
        if c.get("supercategory") != "PART": 
            continue
        nm = c.get("name", "")
        if ":" not in nm:
            continue
        obj_raw, part_raw = nm.split(":", 1)
        obj_name = obj_raw if obj_raw in objs else objs_norm.get(normalizer(obj_raw), obj_raw)
        part_map[c["id"]] = (obj_name, part_raw)
    return part_map

def build_coco_concept(split, data, args):

    id2img = {im["id"]: im["file_name"] for im in data["images"]}
    part_name_map = build_part_name_map(data["categories"], normalizer=norm)
    print(f"part name map: {part_name_map}")
    pann = data.get("annotations") or []
    counts = defaultdict(int)

    def pick_pid(p):
        return p.get("category_id")

    for p in pann:
        img_id = p.get("image_id")
        pid = pick_pid(p)
        box = p.get("bbox")
        area = int(p.get("area"))
        if img_id not in id2img or pid not in part_name_map or box is None:
            print(f"missing image id or category id or bbox")
            continue
        if area < int(args.threshold):
            continue
        img_path = resolve_image_path(args.image_dir, id2img[img_id])
        print(f"image path: {img_path}")
        if not img_path:
            print(f"skipped because bbox size is below threshold")
            continue
        obj_name, part_name = part_name_map[pid]
        on = norm(obj_name); pn = norm(part_name)
        counts[(on,pn)] += 1
        out_path = os.path.join(args.out_concept, split, on, pn, f"{counts[(on,pn)]}.jpg")
        crop_and_save_with_padding(img_path, box, out_path, letterbox=args.use_letterbox)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--annotations_dir", required=True)
    # ap.add_argument("--out_main", default="COCO_main")
    ap.add_argument("--out_concept", default="COCO_concept")
    ap.add_argument("--threshold", default=100, help="area of bounding box below which the image is removed")
    ap.add_argument("--use_letterbox", default=False, help="whether we use letterbox in padding")
    args=ap.parse_args()

    train_json=os.path.join(args.annotations_dir,"paco_lvis_v1_train.json")
    val_json=os.path.join(args.annotations_dir,"paco_lvis_v1_val.json")
    dt=load_json(train_json); dv=load_json(val_json)
    # idx_map=build_idx_map(dt["categories"])
    # build_coco_main_no_crop("train", dt, args.image_dir, args.out_main, idx_map)
    # build_coco_main_no_crop("val", dv, args.image_dir, args.out_main, idx_map)
    build_coco_concept("train", dt, args)
    build_coco_concept("val", dv, args)

if __name__=="__main__":
    main()
