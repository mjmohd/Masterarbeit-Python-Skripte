"""
Match-and-inject script for cooling_hole annotations.

You manually annotate 10 REFERENCE images (10 scales x 2 views), each getting
cooling_hole box(es) in YOLO format. This script then takes those reference
boxes and appends them to every OTHER image's label file that shares the
same scale + view, regardless of defect class, ID, or lighting preset.

Filename pattern expected (from your naming convention):
    <DefectClass>_ID_<id>_<view>_<scale>mm_<lightingPreset>_<lightingDesc>
    e.g. Crack_ID_0_Yneg_1169p9mm_A_baseline_single_light

Matching key = (view, scale) -- lighting and defect class/ID are ignored.

Usage:
python inject_cooling_holes.py \
    --labels-dir "path/to/all/labels" \
    --reference-dir "path/to/reference/labels" \
    --dry-run

Remove --dry-run once you've confirmed the match report looks correct;
without it, the script actually writes/appends to the label files.
"""

import argparse
import re
from pathlib import Path
from collections import defaultdict

# Extract view (Yneg/Ypos) and scale (e.g. 1169p9mm, 2500mm) tokens
# regardless of surrounding structure -- handles both:
#   Crack_ID_56__Ypos__256p2mm__I_four_point_overcast.txt   (defect images)
#   undefective__Yneg__1169p9mm__I_four_point_overcast.txt  (reference/negative images)
VIEW_PATTERN = re.compile(r"(Yneg|Ypos)")
SCALE_PATTERN = re.compile(r"(\d+p\d+mm|\d+mm)")


def parse_key(stem: str):
    """Extract (view, scale) matching key from a filename stem, regardless
    of whether it follows the defect-image or reference-image naming
    pattern."""
    view_match = VIEW_PATTERN.search(stem)
    scale_match = SCALE_PATTERN.search(stem)
    if not view_match or not scale_match:
        return None
    return (view_match.group(1), scale_match.group(1))


def load_cooling_hole_lines(label_path: Path):
    """Return the cooling_hole lines from a reference label file, with the
    class index rewritten from the reference export's index to the real
    dataset's index."""
    # Index cooling_hole has in the CVAT reference export (obj.names)
    REFERENCE_COOLING_HOLE_ID = "0"
    # Index cooling_hole has in the REAL dataset's class_map.json
    TARGET_COOLING_HOLE_ID = "4"

    lines = label_path.read_text().splitlines()
    result = []
    for ln in lines:
        parts = ln.strip().split()
        if not parts:
            continue
        if parts[0] == REFERENCE_COOLING_HOLE_ID:
            parts[0] = TARGET_COOLING_HOLE_ID  # remap class index
            result.append(" ".join(parts))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-dir", required=True,
                         help="Folder containing ALL label .txt files (the full 20k+80 set)")
    parser.add_argument("--reference-dir", required=True,
                         help="Folder containing the 10 manually annotated reference .txt files")
    parser.add_argument("--dry-run", action="store_true",
                         help="Report matches without writing anything")
    args = parser.parse_args()

    labels_dir = Path(args.labels_dir)
    reference_dir = Path(args.reference_dir)

    # Build lookup: (view, scale) -> cooling_hole lines, from the 10 references
    reference_map = {}
    for ref_file in reference_dir.glob("*.txt"):
        key = parse_key(ref_file.stem)
        if key is None:
            print(f"WARNING: could not parse reference filename {ref_file.name}, skipping")
            continue
        ch_lines = load_cooling_hole_lines(ref_file)
        if not ch_lines:
            print(f"WARNING: no cooling_hole lines found in reference {ref_file.name}")
            continue
        if key in reference_map:
            print(f"WARNING: duplicate reference for key {key} "
                  f"({ref_file.name}) -- overwriting previous entry")
        reference_map[key] = ch_lines

    print(f"Loaded {len(reference_map)} reference (view, scale) combinations:")
    for k in reference_map:
        print(f"  {k}")

    # Walk all label files, match by key, append cooling_hole lines
    match_counts = defaultdict(int)
    unmatched = []
    already_had_cooling_hole = 0
    updated = 0

    all_label_files = list(labels_dir.glob("*.txt"))
    print(f"\nScanning {len(all_label_files)} label files in {labels_dir}...")

    for label_file in all_label_files:
        key = parse_key(label_file.stem)
        if key is None:
            unmatched.append(label_file.name)
            continue

        if key not in reference_map:
            unmatched.append(label_file.name)
            continue

        match_counts[key] += 1
        existing_lines = label_file.read_text().splitlines() if label_file.stat().st_size > 0 else []

        # Skip if this file already has cooling_hole lines (avoid duplicate injection)
        COOLING_HOLE_CLASS_ID = "4"
        if any(ln.strip().startswith(COOLING_HOLE_CLASS_ID + " ") for ln in existing_lines):
            already_had_cooling_hole += 1
            continue

        new_lines = existing_lines + reference_map[key]

        if args.dry_run:
            print(f"[DRY RUN] Would update {label_file.name} "
                  f"(+{len(reference_map[key])} cooling_hole lines)")
        else:
            label_file.write_text("\n".join(new_lines) + "\n")
            updated += 1

    print("\n--- Summary ---")
    for k, count in match_counts.items():
        print(f"  {k}: {count} files matched")
    print(f"Files already containing cooling_hole (skipped): {already_had_cooling_hole}")
    print(f"Files unmatched (no key parsed or no reference for that key): {len(unmatched)}")
    if unmatched:
        print("First 10 unmatched filenames:")
        for name in unmatched[:10]:
            print(f"  {name}")

    if args.dry_run:
        print("\nDRY RUN -- no files were modified. Re-run without --dry-run to apply.")
    else:
        print(f"\n{updated} label files updated with cooling_hole annotations.")


if __name__ == "__main__":
    main()
