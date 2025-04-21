from __future__ import annotations
import argparse, os, sys, shutil, json, datetime
from pathlib import Path
from typing import List
from rich import print as rprint
from rich.table import Table
from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, HSplit
from prompt_toolkit.widgets import CheckboxList, Dialog, Button, Label
from prompt_toolkit.shortcuts import radiolist_dialog, prompt
from prompt_toolkit.styles import Style
from ruamel.yaml import YAML

# ─────────────────────────────── CLI ──────────────────────────────── #

def get_args():
    p = argparse.ArgumentParser(description="Interactive QCW concept‑dataset maintenance tool")
    p.add_argument("concept_dir", help="root concept folder with bboxes.json inside")
    p.add_argument("--output-dir", help="Write changes to a fresh copy of dataset here")
    p.add_argument("--flatten", action="store_true", help="Remove HL level (put SC at root)")
    p.add_argument("--yes", action="store_true", help="No confirmation prompts")
    p.add_argument("--dry-run", action="store_true", help="Preview only – no FS writes")
    p.add_argument("--plan", help="YAML/JSON batch‑plan to apply instead of TUI")
    p.add_argument("--save-plan", metavar="PATH", help="Write a reusable YAML plan")
    return p.parse_args()

# ─────────────────────────── Plan object ──────────────────────────── #

class EditPlan:
    def __init__(self):
        self.moves  :dict[str,str] = {}
        self.reject :set[str]      = set()
        self.free   :set[str]      = set()
        self.flatten_moves:dict[str,str] = {}
    # ---------- mutate ----------
    def set_move(self, src, dst):
        self.moves[src] = dst
        self.reject.discard(src)
    def toggle_reject(self, x):
        self.reject.remove(x) if x in self.reject else self.reject.add(x)
    def toggle_free(self, x):
        self.free.remove(x) if x in self.free else self.free.add(x)
    # ---------- helpers ----------
    def status(self, x)->str:
        if   x in self.reject: return "[R]"
        elif x in self.free:   return "[F]"
        elif x in self.moves:  return f"[→{Path(self.moves[x]).name}]"
        else:                  return "   "
    def is_noop(self):
        return not (self.moves or self.reject or self.free or self.flatten_moves)
    # ---------- serialise --------
    def to_dict(self)->dict:
        return {
            "reject"      : sorted(self.reject),
            "mark_free"   : sorted(self.free),
            "rename"      : [{"from":k,"to":v} for k,v in self.moves.items()],
            "flatten_map" : self.flatten_moves
        }
    # ---------- pretty table -----
    def table(self)->Table:
        t=Table(title="Planned changes"); t.add_column("type"); t.add_column("source"); t.add_column("target")
        for s,d in self.moves.items():   t.add_row("[cyan]move[/]",s,d)
        for r in self.reject:            t.add_row("[red]reject[/]",r,"rejects/"+r)
        for f in self.free:              t.add_row("[green]mark‑free[/]",f,f+"_free")
        for s,d in self.flatten_moves.items(): t.add_row("[magenta]flatten[/]",s,d)
        if not t.rows: t.add_row("—","no changes","—")
        return t

# ─────────────────────── dataset helpers ──────────────────────────── #

def hi_lo_pairs(root:Path)->dict[str, list[str]]:
    d={}
    for hl in (root/'concept_train').iterdir():
        if hl.is_dir():
            d[hl.name]=sorted(p.name for p in hl.iterdir() if p.is_dir())
    return d

def load_yaml_json(p:Path):
    if p.suffix.lower()==".json": return json.loads(p.read_text())
    y=YAML(typ="safe"); return y.load(p)

def backup(p:Path):
    ts=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dst=p.with_suffix(p.suffix+f".bak_{ts}")
    shutil.copy2(p,dst); return dst

# ───────────────────────────── UI bits ────────────────────────────── #

PT_STYLE = Style.from_dict({
    "dialog": "bg:#202020",
    "dialog.body": "bg:#1b1b1b",
    "dialog shadow": "bg:#101010",
})

LEGEND_TEXT = (
    "Space/Enter/CLICK = toggle    ↑↓/Wheel or W/S = move    "
    "Ctrl‑A = Select‑all    P = save‑plan    "
    "Legend:  [R] rejected   [F] mark‑free   [→x] rename/merge"
)

def choose_subconcepts(hl:str, sc_list:List[str], plan:EditPlan)->List[str] | None:
    items=[(sc, f"{plan.status(f'{hl}/{sc}')}  {sc}") for sc in sc_list]
    cb=CheckboxList(values=items)

    def move(delta:int): cb._selected_index = max(0, min(len(items)-1, cb._selected_index + delta))
    def toggle_all():     cb.current_values = [v for v,_ in items] if len(cb.current_values)<len(items) else []
    def _ok():            app.exit(result=cb.current_values)
    def _cancel():        app.exit(result=None)

    kb = KeyBindings()
    kb.add("w")(lambda e: move(-1))
    kb.add("s")(lambda e: move(+1))
    kb.add("c-a")(lambda e: toggle_all())
    kb.add("escape")(lambda e: _cancel())

    body = HSplit([Label(text=LEGEND_TEXT), cb])
    dlg  = Dialog(
        title=hl,
        body=body,
        buttons=[Button(text="Select‑all",handler=toggle_all),
                 Button(text="OK",          handler=_ok),
                 Button(text="Cancel",      handler=_cancel)],
        with_background=True
    )
    app = Application(layout=Layout(dlg), key_bindings=kb,
                      style=PT_STYLE, mouse_support=True, full_screen=True)
    return app.run()

# ─────────────────── interactive plan builder ─────────────────────── #

def _hl_summary(hl:str, sc_list:List[str], plan:EditPlan)->str:
    total=len(sc_list)
    rej=sum(1 for sc in sc_list if f"{hl}/{sc}" in plan.reject)
    fre=sum(1 for sc in sc_list if f"{hl}/{sc}" in plan.free)
    mov=sum(1 for sc in sc_list if f"{hl}/{sc}" in plan.moves)

    if total and rej==total:
        return "(ALL‑R)"                          # <- new flag
    bits=[str(total)]
    if rej: bits.append(f"R:{rej}")
    if fre: bits.append(f"F:{fre}")
    if mov: bits.append(f"→:{mov}")
    return f"({' '.join(bits)})"

def interactive_plan(root:Path)->EditPlan:
    struct=hi_lo_pairs(root); plan=EditPlan()
    while True:
        hl_vals=[(h,f"{h:<12} {_hl_summary(h,struct[h],plan)}") for h in struct]
        hl_vals.append(("__done","< Done >"))
        hl=radiolist_dialog(title="High‑level concepts",
                            text="Select with arrows / W‑S.  Enter to open.\n"
                                 "P = save‑plan   Ctrl‑C = quit\n",
                            values=hl_vals, style=PT_STYLE).run()
        if hl in (None,"__done"): break
        picked = choose_subconcepts(hl, struct[hl], plan)
        if not picked: continue

        # bulk‑action choices
        any_rej     = any(plan.status(f"{hl}/{sc}")!="[R]" for sc in picked)
        any_unrej   = any(plan.status(f"{hl}/{sc}")=="[R]"  for sc in picked)
        any_free    = any(plan.status(f"{hl}/{sc}")!="[F]" for sc in picked)
        any_unfree  = any(plan.status(f"{hl}/{sc}")=="[F]"  for sc in picked)

        acts=[("merge","Merge (into one)"),("rename","Rename / move")]
        if any_rej:    acts.insert(0,("reject","Reject"))
        if any_unrej:  acts.insert(0,("unreject","Un‑reject"))
        if any_free:   acts.append(("free","Mark‑free"))
        if any_unfree: acts.append(("unfree","Un‑free"))
        acts.append(("skip","Skip"))
        act=radiolist_dialog(title="Bulk action", values=acts, style=PT_STYLE).run()
        if act in (None,"skip"): continue

        if act in ("reject","unreject"):
            for sc in picked: plan.toggle_reject(f"{hl}/{sc}")
        elif act in ("free","unfree"):
            for sc in picked: plan.toggle_free(f"{hl}/{sc}")
        elif act=="rename":
            for sc in picked:
                new=prompt(f"{hl}/{sc} → ").strip().strip("/")
                if new: plan.set_move(f"{hl}/{sc}", new)
        elif act=="merge":
            tgt=prompt("Target (HL/newSC) : ").strip().strip("/")
            if not tgt: tgt=f"{hl}/{picked[0]}"
            for sc in picked:
                src=f"{hl}/{sc}"
                if src!=tgt: plan.set_move(src, tgt)
    return plan

# ───────────────────── plan from / to file ────────────────────────── #

def plan_from_file(p:Path)->EditPlan:
    data=load_yaml_json(p)
    pl=EditPlan()
    for r in data.get("reject",[]):              pl.toggle_reject(r)
    for f in data.get("mark_free",[]):           pl.toggle_free(f)
    for mv in data.get("rename",[]):             pl.set_move(mv["from"], mv["to"])
    for o,n in data.get("flatten_map",{}).items(): pl.flatten_moves[o]=n
    return pl

def save_plan_yaml(plan:EditPlan, path:Path):
    y=YAML(); y.default_flow_style=False
    with path.open("w") as f: y.dump(plan.to_dict(), f)
    rprint(f"[green]✓ plan saved → {path}[/]")

# ──────────────────── filesystem operations ───────────────────────── #

def apply_fs(plan:EditPlan, root:Path):
    for split in ("concept_train","concept_val"):
        base=root/split; (base/'rejects').mkdir(exist_ok=True)
        # moves
        for src,dst in plan.moves.items():
            s,d=base/src, base/dst; d.parent.mkdir(parents=True,exist_ok=True)
            if d.exists():
                for f in s.iterdir(): shutil.move(str(f), d/f.name)
                shutil.rmtree(s)
            elif s.exists(): shutil.move(s,d)
        # rejects
        for r in plan.reject:
            s=base/r; dst=base/'rejects'/r; dst.parent.mkdir(parents=True,exist_ok=True)
            if s.exists(): shutil.move(s,dst)
        # mark‑free
        for f in plan.free:
            s=base/f
            if s.exists() and not s.name.endswith("_free"):
                shutil.move(s, s.with_name(s.name+"_free"))
        # ---- clean empty HL dirs ----
        for hl_dir in (p for p in base.iterdir() if p.is_dir()):
            if hl_dir.name!="rejects" and not any(hl_dir.iterdir()):
                hl_dir.rmdir()

# ---------- flatten & helpers unchanged -------- #
def compute_flatten(root:Path)->dict[str,str]:
    moves,used={},set()
    for hl in (root/'concept_train').iterdir():
        if hl.is_dir() and hl.name!="rejects":
            for sc in hl.iterdir():
                tgt=sc.name if sc.name not in used else f"{hl.name}__{sc.name}"
                used.add(tgt); moves[f"{hl.name}/{sc.name}"]=tgt
    return moves

def exec_flatten(root:Path,mapping:dict):
    for split in ("concept_train","concept_val"):
        base=root/split
        for old,new in mapping.items():
            s,d=base/old, base/new; d.parent.mkdir(parents=True,exist_ok=True)
            if d.exists():
                for f in s.iterdir(): shutil.move(str(f), d/f.name)
                shutil.rmtree(s)
            elif s.exists(): shutil.move(s,d)
        for hl in (p for p in base.iterdir() if p.is_dir()):
            if hl.name!="rejects" and not any(hl.iterdir()): hl.rmdir()

def rewrite_bboxes(bb:dict,plan:EditPlan)->dict:
    def trans(k):
        for o,n in plan.moves.items():
            for s in ("concept_train","concept_val"):
                k=k.replace(f"{s}/{o}/", f"{s}/{n}/")
        for r in plan.reject:
            if any(k.startswith(f"{s}/{r}/") for s in ("concept_train","concept_val")):
                return None
        for f in plan.free:
            for s in ("concept_train","concept_val"):
                k=k.replace(f"{s}/{f}/", f"{s}/{f}_free/")
        for o,n in plan.flatten_moves.items():
            for s in ("concept_train","concept_val"):
                k=k.replace(f"{s}/{o}/", f"{s}/{n}/")
        return k
    return {nk:v for k,v in bb.items() if (nk:=trans(k))}

# ─────────────────── dataset copy (optional) ──────────────────────── #

def copy_dataset(src:Path,dst:Path):
    if dst.exists():
        rprint("[red]✗ output dir exists, abort.[/]"); sys.exit(1)
    rprint(f"[cyan]Copying dataset → {dst}[/]"); shutil.copytree(src,dst)

# ─────────────────────────────  MAIN  ─────────────────────────────── #

def main():
    a=get_args()
    src=Path(a.concept_dir).resolve()
    root=Path(a.output_dir).resolve() if a.output_dir else src
    if a.output_dir and not a.dry_run:
        copy_dataset(src, root)

    plan = plan_from_file(Path(a.plan)) if a.plan else interactive_plan(root)
    if a.flatten:
        plan.flatten_moves = compute_flatten(root)

    rprint(); rprint(plan.table()); rprint()
    if plan.is_noop():
        rprint("[yellow]Nothing to do.[/]"); return

    save_path = Path(a.save_plan) if a.save_plan else None
    if not save_path and (not a.yes):
        p=input("Save plan to YAML? (blank = skip) : ").strip()
        save_path = Path(p) if p else None
    if save_path: save_plan_yaml(plan, save_path)

    if (a.dry_run or not a.yes) and not a.dry_run:
        if input("Apply changes now? [y/N] ").lower()!="y": return
    elif not a.yes and not a.dry_run:
        if input("Proceed? [y/N] ").lower()!="y": return

    if not a.dry_run:
        apply_fs(plan, root)
        if a.flatten: exec_flatten(root, plan.flatten_moves)

        bb=root/"bboxes.json"
        if bb.exists():
            backup(bb)
            new_bb=rewrite_bboxes(json.loads(bb.read_text()), plan)
            bb.write_text(json.dumps(new_bb, indent=4))

        rprint("[bold green]✓ All done.[/]")
        if a.output_dir: rprint(f"Edited copy in [cyan]{root}[/]")

if __name__=="__main__":
    main()
