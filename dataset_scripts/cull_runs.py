import os
import sys
import glob
import shutil
import argparse
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def count_val_loss_steps(event_file: str, tag: str = "Val/Loss") -> int:
    """Count how many times the tag appears in the event file. Returns 0 if missing."""
    ea = EventAccumulator(event_file)
    ea.Reload()  # parse the file

    if tag in ea.Tags().get("scalars", []):
        return len(ea.Scalars(tag))
    else:
        return 0

def find_event_files(run_dir: str):
    """Find all tensorboard event files in the directory."""
    pattern = os.path.join(run_dir, "events.out.tfevents.*")
    return glob.glob(pattern)

def parse_run_name(directory_name: str):
    """Split 'myrun_1234567890' into ('myrun', '1234567890')"""
    if "_" not in directory_name:
        return directory_name, ""

    # check for timestamp suffix
    base, _, maybe_digits = directory_name.rpartition("_")
    if maybe_digits.isdigit():
        return base, maybe_digits
    else:
        # not a timestamp format
        return directory_name, ""

def main():
    parser = argparse.ArgumentParser(
        description="Delete TB runs with too few validation steps or duplicates"
    )
    parser.add_argument("runs_dir", help="Directory containing TB runs")
    parser.add_argument("--tag", default="Val/Loss", 
                        help="Scalar tag to check")
    parser.add_argument("--min_steps", type=int, default=45,
                        help="Min steps to keep a run")
    parser.add_argument("--cull_repeats", action="store_true",
                        help="Remove older runs with same name")
    args = parser.parse_args()

    runs_dir = args.runs_dir
    scalar_tag = args.tag
    min_steps = args.min_steps

    if not os.path.isdir(runs_dir):
        print(f"[ERROR] '{runs_dir}' is not a valid directory.")
        sys.exit(1)

    # find all run dirs
    potential_runs = []
    for entry in os.scandir(runs_dir):
        if entry.is_dir():
            potential_runs.append(entry.path)

    # handle duplicates first
    duplicates_cull = []
    if args.cull_repeats:
        # group by base name
        grouping = {}  # base_name -> list of (full_path, numeric_str)
        for run_path in potential_runs:
            dir_name = os.path.basename(run_path)
            base, suffix = parse_run_name(dir_name)
            grouping.setdefault(base, []).append((run_path, suffix))

        # keep newest for each base
        keep_set = set()
        for base_name, run_infos in grouping.items():
            if len(run_infos) == 1:
                # Only one run, nothing to cull
                keep_set.add(run_infos[0][0])
                continue

            # sort by timestamp
            def suffix_to_int(s):
                return int(s) if s.isdigit() else 0
            run_infos_sorted = sorted(run_infos, key=lambda x: suffix_to_int(x[1]))

            # newest = largest timestamp
            newest = run_infos_sorted[-1][0]
            keep_set.add(newest)

            # mark older ones for deletion
            for (rpath, sfx) in run_infos_sorted[:-1]:
                duplicates_cull.append(rpath)

        # remove duplicates from candidates
        potential_runs = [rp for rp in potential_runs if rp not in duplicates_cull]

    # now check step counts
    cull_list = []
    for run_path in potential_runs:
        event_files = find_event_files(run_path)
        if not event_files:
            # no events = cull
            cull_list.append((run_path, 0))
            continue

        # use biggest event file
        event_file = max(event_files, key=os.path.getsize)
        steps_count = count_val_loss_steps(event_file, scalar_tag)

        if steps_count < min_steps:
            cull_list.append((run_path, steps_count))

    # combine both deletion lists
    final_cull_set = set(duplicates_cull)
    for run_path, _ in cull_list:
        final_cull_set.add(run_path)

    # show summary
    if not final_cull_set:
        print("No runs to remove. Exiting.")
        return

    print("\nThe following runs will be removed:")
    # duplicates first
    if args.cull_repeats and duplicates_cull:
        print("\n[Duplicate Runs (older than newest)]:")
        for d in duplicates_cull:
            print(f"   {d}")
    # then step-based culls
    if cull_list:
        print(f"\n[Runs with fewer than {min_steps} steps for '{scalar_tag}']:")
        for rp, st in cull_list:
            print(f"   {rp} => {st} steps")

    confirm = input("\nAre you sure you want to DELETE these directories? (y/n): ").strip().lower()
    if confirm == "y":
        for run_path in final_cull_set:
            print(f"Deleting: {run_path}")
            shutil.rmtree(run_path, ignore_errors=True)
        print("Done.")
    else:
        print("Aborted. No directories were removed.")

if __name__ == "__main__":
    main()
