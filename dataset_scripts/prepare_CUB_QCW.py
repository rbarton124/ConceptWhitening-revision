import os
import argparse
import pandas as pd
import json
import shutil

from collections import defaultdict
from PIL import Image, ImageDraw
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="CUB-200-2011 -> QCW format with parts-based bounding boxes (Verbose)")
    parser.add_argument("--cub_dir", required=True,
                        help="Path to unzipped CUB_200_2011 dataset")
    parser.add_argument("--output_main", default=None,
                        help="If provided, create main dataset here (train/val/test). Otherwise skip.")
    parser.add_argument("--output_concept", required=True,
                        help="Output directory for concept dataset")
    parser.add_argument("--mode", choices=["concepts","attributes"], default="concepts",
                        help="Mode for high-level concept assignment.")
    parser.add_argument("--mappings", required=True,
                        help="Path to mappings.json. In 'concepts' mode, it is subConcept->HL. In 'attributes' mode, it maps 'has_wing_color'->'wing'.")
    parser.add_argument("--concepts", required=True,
                        help="Comma-separated list of high-level concepts we include (e.g. 'wing,beak,throat,general')")
    parser.add_argument("--draw_or_copy", choices=["draw","crop","copy"], default="copy",
                        help="How to process images for concept dataset: 'draw'=draw bbox, 'crop'=crop around it, 'copy'=no modification.")
    parser.add_argument("--crop_size", type=int, default=224,
                        help="If draw_or_copy='crop', the region is resized to this size after bounding box crop.")
    parser.add_argument("--val_fraction", type=float, default=0.2,
                        help="Fraction of test images used for concept_val or main dataset val.")
    parser.add_argument("--part_margin", type=int, default=40,
                        help="Margin around single part-loc if only one visible part is found.")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="Increase verbosity level. Use -v for basic progress, -vv for per-image messages, -vvv for full debug output.")
    return parser.parse_args()

def vprint(args, level, msg):
    if args.verbose >= level:
        print(msg)

##############################################
# Basic Annotation Loaders
##############################################

def load_images(cub_dir):
    path = os.path.join(cub_dir, "images.txt")
    return pd.read_csv(path, sep=" ", header=None, names=["image_id","path"])

def load_split(cub_dir):
    path = os.path.join(cub_dir, "train_test_split.txt")
    return pd.read_csv(path, sep=" ", header=None, names=["image_id","is_train"])

def load_classes(cub_dir):
    path = os.path.join(cub_dir, "image_class_labels.txt")
    return pd.read_csv(path, sep=" ", header=None, names=["image_id","class"])

def load_bboxes(cub_dir):
    path = os.path.join(cub_dir, "bounding_boxes.txt")
    return pd.read_csv(path, sep=" ", header=None, names=["image_id","x","y","width","height"])

def load_parts_txt(cub_dir):
    path = os.path.join(cub_dir,"parts","parts.txt")
    rows=[]
    with open(path,"r") as f:
        for line in f:
            tokens=line.strip().split()
            if len(tokens)<2:
                continue
            p_id=int(tokens[0])
            p_name=" ".join(tokens[1:]).lower()
            rows.append((p_id,p_name))
    return pd.DataFrame(rows, columns=["part_id","part_name"])

def load_part_locs(cub_dir):
    path = os.path.join(cub_dir,"parts","part_locs.txt")
    df = pd.read_csv(path,sep=" ",header=None,
                     names=["image_id","part_id","x","y","visible"])
    df=df[df["visible"]==1]
    return df

def load_attributes(cub_dir):
    attr_file = os.path.join(cub_dir,"attributes","attributes.txt")
    attr_map={}
    with open(attr_file,"r") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            sid=line.split(" ",1)
            if len(sid)==2:
                attr_id=int(sid[0])
                attr_name=sid[1].strip().lower()
                attr_map[attr_id]=attr_name
    lbl_file = os.path.join(cub_dir,"attributes","image_attribute_labels.txt")
    df = pd.read_csv(lbl_file,sep=" ",header=None,
                     names=["image_id","attribute_id","is_present","certainty_id","time"],
                     on_bad_lines="skip")
    df=df[(df["is_present"]==1) & (df["certainty_id"]>=3)]
    df["attribute"]=df["attribute_id"].map(attr_map)
    return df

##############################################
# PARTS HELPER
##############################################

def build_partname_to_ids(parts_df):
    """
    e.g. 'wing' -> [9,13], 'eye'-> [7,11], ...
    """
    name_to_idlist = defaultdict(list)
    for _,row in parts_df.iterrows():
        pid = row["part_id"]
        pname = row["part_name"]  # e.g. "left wing"
        if pname.startswith("left "):
            pname=pname.replace("left ","")
        elif pname.startswith("right "):
            pname=pname.replace("right ","")
        pname=pname.strip()
        name_to_idlist[pname].append(pid)
    return dict(name_to_idlist)

def gather_part_bounding_box(image_id, part_ids, partlocs, single_margin=0):
    subset=partlocs[(partlocs["image_id"]==image_id) & (partlocs["part_id"].isin(part_ids))]
    if len(subset)==0:
        return None
    xvals = subset["x"].values
    yvals = subset["y"].values
    if len(xvals)==1:
        # single location => create a box with margin
        x=float(xvals[0]); y=float(yvals[0])
        x1=x-single_margin; y1=y-single_margin
        x2=x+single_margin; y2=y+single_margin
        return [x1,y1,x2,y2]
    else:
        # multiple => unify
        x_min = float(xvals.min())
        x_max = float(xvals.max())
        y_min = float(yvals.min())
        y_max = float(yvals.max())
        return [x_min,y_min,x_max,y_max]

##############################################
# MAIN DATASET
##############################################

def build_main_dataset(args):
    cub_dir = args.cub_dir
    out_dir = args.output_main
    val_fraction=args.val_fraction

    vprint(args,1,f"[Info] Building main dataset in {out_dir}, val_fraction={val_fraction}")
    imagesdf = load_images(cub_dir)
    splitdf = load_split(cub_dir)
    classdf = load_classes(cub_dir)
    df = imagesdf.merge(splitdf,on="image_id").merge(classdf,on="image_id")

    disable_tqdm = (args.verbose<1)

    for sub in ["train","val","test"]:
        os.makedirs(os.path.join(out_dir,sub),exist_ok=True)
    for _,row in tqdm(df.iterrows(),total=len(df),desc="[Main data]", disable=disable_tqdm):
        is_train=row["is_train"]
        if is_train==1:
            split="train"
        else:
            split="test"
        c_id=row["class"]
        classdir=os.path.join(out_dir,split,str(c_id).zfill(3))
        os.makedirs(classdir,exist_ok=True)
        src=os.path.join(cub_dir,"images",row["path"])
        dst=os.path.join(classdir,os.path.basename(row["path"]))
        if not os.path.exists(dst):
            shutil.copy(src,dst)
    # create val out of test
    testdf=df[df["is_train"]==0]
    grouped=testdf.groupby("class")
    for c,group in tqdm(grouped,desc="[Main data: Test->Val]", disable=disable_tqdm):
        n_val=int(round(len(group)*val_fraction))
        if n_val<=0: continue
        sample=group.sample(n=n_val,random_state=42)
        for _,rr in sample.iterrows():
            src=os.path.join(out_dir,"test",str(rr["class"]).zfill(3),os.path.basename(rr["path"]))
            dstdir=os.path.join(out_dir,"val",str(rr["class"]).zfill(3))
            os.makedirs(dstdir,exist_ok=True)
            dst=os.path.join(dstdir,os.path.basename(rr["path"]))
            if os.path.exists(src):
                shutil.move(src,dst)
    vprint(args,1,f"[Main Dataset] Done organizing {out_dir}")

##############################################
# CONCEPT DATASET
##############################################

def build_concept_dataset(args):
    cub_dir=args.cub_dir
    out_dir=args.output_concept
    mode=args.mode
    draw_or_copy=args.draw_or_copy
    margin=args.part_margin
    vprint(args,1,f"[Info] Building concept dataset in {out_dir}, mode={mode}, draw_or_copy={draw_or_copy}, margin={margin}")
    
    disable_tqdm = (args.verbose<1)

    imagesdf=load_images(cub_dir)
    splitdf=load_split(cub_dir)
    classdf=load_classes(cub_dir)
    base_df=imagesdf.merge(splitdf,on="image_id").merge(classdf,on="image_id")
    
    # bounding_boxes for "general"
    bboxdf=load_bboxes(cub_dir)
    
    # For part-loc bounding boxes
    parts_df=load_parts_txt(cub_dir)
    part_locs=load_part_locs(cub_dir)
    name_to_ids=build_partname_to_ids(parts_df)
    
    # Attributes
    attr_df=load_attributes(cub_dir)
    merged=attr_df.merge(base_df,on="image_id",how="left")
    
    # Mappings
    with_mappings={}
    if args.mappings and os.path.isfile(args.mappings):
        with open(args.mappings,"r") as ff:
            with_mappings=json.load(ff)
    selected_hl = set([x.strip().lower() for x in args.concepts.split(",")])
    
    def get_hl_sub(attribute_name):
        a=attribute_name.strip().lower()
        if mode=="concepts":
            if a in with_mappings:
                hl=with_mappings[a].lower()
                vprint(args,3,f"[DBG-map] 'concepts' mode: '{a}' => HL from mappings => {hl}")
            else:
                hl="general"
                vprint(args,3,f"[DBG-map] 'concepts' mode: '{a}' not in mappings => fallback general")
            sub=a
            return (hl, sub)
        else:
            # "attributes"
            parts=a.split("::",1)
            if len(parts)==2:
                prefix=parts[0].strip()
                suffix=parts[1].strip()
                if prefix in with_mappings:
                    hl=with_mappings[prefix].lower()
                    vprint(args,3,f"[DBG-map] 'attributes': prefix '{prefix}' => {hl}")
                else:
                    hl="general"
                    vprint(args,3,f"[DBG-map] 'attributes': prefix '{prefix}' => fallback general")
                sub=a
                return (hl, sub)
            else:
                vprint(args,3,f"[DBG-map] 'attributes': no '::' => fallback general => HL=general")
                return ("general", a)
    
    # concept_train or concept_val
    concept_train_dir=os.path.join(out_dir,"concept_train")
    concept_val_dir=os.path.join(out_dir,"concept_val")
    os.makedirs(concept_train_dir,exist_ok=True)
    os.makedirs(concept_val_dir,exist_ok=True)
    
    # final bounding box dict
    final_bboxes={}
    
    # Summaries for debugging part-loc usage:
    vprint(args,2,"\n[DEBUG] Checking part-loc mapping for the selected HL concepts:")
    for hlc in selected_hl:
        if hlc=="general":
            vprint(args,2,f"  HL '{hlc}' => uses official bounding_boxes.txt")
        else:
            if hlc in name_to_ids:
                vprint(args,2,f"  HL '{hlc}' => part IDs: {name_to_ids[hlc]}")
            else:
                vprint(args,2,f"  HL '{hlc}' => No part name found => fallback general???")
    vprint(args,2,"")
    
    # Iterate over each row in attributes
    for i,row in tqdm(merged.iterrows(),total=len(merged),desc="[Concept data]", disable=disable_tqdm):
        attr=row["attribute"]
        if pd.isna(attr):
            continue
        hl, sub = get_hl_sub(attr)
        if hl not in selected_hl:
            # skip if HL not in user-defined concepts
            continue
        
        image_id=row["image_id"]
        if row["is_train"] in [1,2]:
            out_subdir=os.path.join(concept_train_dir,hl,sub)
        else:
            out_subdir=os.path.join(concept_val_dir,hl,sub)
        os.makedirs(out_subdir,exist_ok=True)
        
        src_img=os.path.join(cub_dir,"images",row["path"])
        final_name=f"{str(image_id).zfill(4)}_{os.path.basename(row['path'])}"
        final_path=os.path.join(out_subdir,final_name)
        
        # Decide bounding box
        if hl=="general":
            rec=bboxdf[bboxdf["image_id"]==image_id]
            if len(rec)>0:
                r=rec.iloc[0]
                x1=r["x"]; y1=r["y"]
                x2=r["x"]+r["width"]; y2=r["y"]+r["height"]
                box=[x1,y1,x2,y2]
                vprint(args,3,f"[DEBUG] => general bounding box from bounding_boxes: {box}")
            else:
                box=None
                vprint(args,3,f"[DEBUG] => no bounding_box row => box=None")
        else:
            if hl in name_to_ids:
                pids=name_to_ids[hl]
                box=gather_part_bounding_box(image_id, pids, part_locs, single_margin=margin)
                vprint(args,3,f"[DEBUG] => HL '{hl}', pids={pids} => part-loc box {box}")
            else:
                box=None
                vprint(args,3,f"[DEBUG] => HL '{hl}' not found in name_to_ids => box=None")
        
        try:
            im=Image.open(src_img).convert("RGB")
        except Exception as e:
            vprint(args,2,f"[WARNING] Could not open {src_img}: {e}")
            continue
        
        if box is None:
            vprint(args,2,f"[Info] image_id {image_id} attr '{attr}' => no part-loc => copying as is.")
            shutil.copy(src_img, final_path)
        else:
            left,top,right,bottom = _clip_box_to_image(box, im.width, im.height)
            if right<=left or bottom<=top:
                vprint(args,2,f"[Info] image_id {image_id} => clipped box invalid => copying.")
                shutil.copy(src_img, final_path)
            else:
                if draw_or_copy=="draw":
                    draw=ImageDraw.Draw(im)
                    draw.rectangle([left,top,right,bottom], outline="red", width=3)
                    im.save(final_path)
                    vprint(args,2,f"[Info] image_id {image_id} => drew rectangle => {left,top,right,bottom}")
                elif draw_or_copy=="crop":
                    cropped=im.crop((left,top,right,bottom))
                    cropped=cropped.resize((args.crop_size,args.crop_size))
                    cropped.save(final_path)
                    vprint(args,2,f"[Info] image_id {image_id} => cropped => {left,top,right,bottom}")
                else:
                    shutil.copy(src_img, final_path)
                    vprint(args,2,f"[Info] image_id {image_id} => copy => bounding box {left,top,right,bottom}")
                
                rel_path=os.path.relpath(final_path, out_dir)
                final_bboxes[rel_path]=[float(left),float(top),float(right),float(bottom)]
    
    # Save final_bboxes
    bbjson=os.path.join(out_dir,"bboxes.json")
    with open(bbjson,"w") as f:
        json.dump(final_bboxes,f,indent=4)
    vprint(args,1,f"[Concept Dataset] Wrote bounding boxes => {bbjson}")

def _clip_box_to_image(box, img_w, img_h):
    left, top, right, bottom = box
    left = max(left, 0)
    top = max(top, 0)
    right = min(right, img_w)
    bottom = min(bottom, img_h)
    return (left, top, right, bottom)

##############################################
# MAIN
##############################################

def main():
    args = parse_args()
    
    # If verbosity=0, we disable tqdm entirely
    if args.verbose==0:
        # minimal prints
        pass

    if args.output_main:
        build_main_dataset(args)
    build_concept_dataset(args)

    vprint(args,1,"[Done] Complete dataset creation for QCW.")

if __name__=="__main__":
    main()