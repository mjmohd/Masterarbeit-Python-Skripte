"""
Removes cooling_hole boxes that fall under an "occluding" defect box
(by default: Erosion), since a cooling hole covered by erosion texture
isn't actually visible in the render and shouldn't be labeled.

A cooling_hole box is removed if a sufficient fraction of its own area
overlaps with an occluding defect's box (default threshold: 30%), OR
if its center point falls inside the defect box -- whichever is more
appropriate for your data; both checks are applied (removed if either
is true) to be safe against partial/edge cases.

Usage:
    python remove_occluded_cooling_holes.py \
        --labels-dir "path/to/all/labels" \
        --occluding-classes 2 \
        --overlap-threshold 0.3 \
        --dry-run

    --occluding-classes accepts multiple, e.g. --occluding-classes 2 3
    to also treat Nick as occluding, if that turns out to apply.

Remove --dry-run once the report looks correct to actually rewrite files.
"""

import argparse
from pathlib import Path

COOLING_HOLE_CLASS_ID = "4"


def yolo_to_xyxy(cx, cy, w, h):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def box_area(x1, y1, x2, y2):
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return box_area(ix1, iy1, ix2, iy2)


def center_inside(box_small, box_big):
    cx = (box_small[0] + box_small[2]) / 2
    cy = (box_small[1] + box_small[3]) / 2
    bx1, by1, bx2, by2 = box_big
    return bx1 <= cx <= bx2 and by1 <= cy <= by2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--occluding-classes", nargs="+", default=["2"],
                         help="Class indices treated as occluding (default: 2 = Erosion)")
    parser.add_argument("--overlap-threshold", type=float, default=0.3,
                         help="Fraction of cooling_hole box area that must overlap "
                              "an occluding box to trigger removal (default 0.3)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    labels_dir = Path(args.labels_dir)
    occluding_classes = set(args.occluding_classes)

    all_label_files = list(labels_dir.glob("*.txt"))
    print(f"Scanning {len(all_label_files)} label files in {labels_dir}...")

    total_removed = 0
    files_changed = 0

    for label_file in all_label_files:
        text = label_file.read_text()
        if not text.strip():
            continue

        lines = text.splitlines()
        parsed = []
        malformed = False
        for ln in lines:
            parts = ln.strip().split()
            if not parts:
                continue
            cls_id = parts[0]
            try:
                cx, cy, w, h = map(float, parts[1:5])
            except (ValueError, IndexError):
                malformed = True
                break
            parsed.append((cls_id, cx, cy, w, h, ln))

        if malformed:
            print(f"WARNING: skipping non-YOLO-format file {label_file.name}")
            continue

        occluding_boxes = [
            yolo_to_xyxy(cx, cy, w, h)
            for cls_id, cx, cy, w, h, ln in parsed
            if cls_id in occluding_classes
        ]

        if not occluding_boxes:
            continue  # nothing to occlude with in this file

        kept_lines = []
        removed_count = 0

        for cls_id, cx, cy, w, h, ln in parsed:
            if cls_id != COOLING_HOLE_CLASS_ID:
                kept_lines.append(ln)
                continue

            ch_box = yolo_to_xyxy(cx, cy, w, h)
            ch_area = box_area(*ch_box)

            should_remove = False
            for occ_box in occluding_boxes:
                if center_inside(ch_box, occ_box):
                    should_remove = True
                    break
                if ch_area > 0:
                    overlap_frac = intersection_area(ch_box, occ_box) / ch_area
                    if overlap_frac >= args.overlap_threshold:
                        should_remove = True
                        break

            if should_remove:
                removed_count += 1
            else:
                kept_lines.append(ln)

        if removed_count > 0:
            files_changed += 1
            total_removed += removed_count
            if args.dry_run:
                print(f"[DRY RUN] {label_file.name}: would remove {removed_count} cooling_hole box(es)")
            else:
                label_file.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""))

    print("\n--- Summary ---")
    print(f"Files with occluded cooling_hole boxes: {files_changed}")
    print(f"Total cooling_hole boxes removed: {total_removed}")
    if args.dry_run:
        print("\nDRY RUN -- no files were modified. Re-run without --dry-run to apply.")
    else:
        print(f"\n{files_changed} files updated.")


if __name__ == "__main__":
    main()
