#!/usr/bin/env python3
import argparse, json, os, re, unicodedata
from pathlib import Path
from collections import defaultdict
from PIL import Image
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

def build_coco_main(split, data, image_dir, out_root, idx_map):
    id2img={im["id"]:im["file_name"] for im in data["images"]}
    counts=defaultdict(int)
    img_split=split_to_folder(split)
    for ann in data["annotations"]:
        img_id=ann.get("image_id"); cat_id=ann.get("category_id"); box=ann.get("bbox")
        if img_id not in id2img or box is None or cat_id not in idx_map: continue
        img_path=os.path.join(image_dir,img_split,id2img[img_id])
        if not os.path.exists(img_path): continue
        obj_idx=idx_map[cat_id]
        counts[obj_idx]+=1
        out_path=os.path.join(out_root,split,obj_idx,f"{counts[obj_idx]}.jpg")
        crop_and_save(img_path, box, out_path)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--annotations_dir", required=True)
    ap.add_argument("--out_main", default="COCO_main")
    ap.add_argument("--out_concept", default="COCO_concept")
    args=ap.parse_args()

    train_json=os.path.join(args.annotations_dir,"paco_lvis_v1_train.json")
    val_json=os.path.join(args.annotations_dir,"paco_lvis_v1_val.json")
    dt=load_json(train_json); dv=load_json(val_json)
    idx_map=build_idx_map(dt["categories"])
    build_coco_main("train", dt, args.image_dir, args.out_main, idx_map)
    build_coco_main("val", dv, args.image_dir, args.out_main, idx_map)

if __name__=="__main__":
    main()
