"""
Blade Inspection Decision Support Tool -- single file (German UI).

Three parts, in order:
  1. DECISION ENGINE  -- the repair/scrap rules (location, area, zone, width)
  2. DETECTION PIPELINE -- loads a trained YOLO model, runs it on an image,
     hands each detected box to the decision engine
  3. UI -- desktop window: load an image, load weights, run, see the result

Behaviour
  - First gate: "Defect detected?". No detection -> blade is HEALTHY.
    (handle_no_defect() is the hook for a future manual-annotation branch.)
  - Overall blade verdict (healthy / repair / scrap) shown in the status line.
  - "Save annotated image": exports the annotated blade on a white sheet
    with padding, a result box and a legend.
  - Dimension values (X / Y on the image, dx / dy in the cards) are shown
    as distances, i.e. always positive.
  - YOLO model is loaded once per weights file (cached) and warmed up.
  - Blade mask + per-row geometry computed once per image (BladeGeometry).
  - Merge slider is debounced; detection runs in a background thread.

Run locally (needs ultralytics + your trained weights + rules_config.json
in the same folder as this script):
    python blade_inspection_app.py
"""

import os
import re
import json
import queue
import threading
from pathlib import Path

import numpy as np
from scipy.ndimage import label, binary_closing
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2  # optional: much faster morphological closing
except ImportError:
    cv2 = None

try:
    from PIL import ImageTk
except ImportError:  # keeps the engine importable without tkinter
    ImageTk = None

_Resampling = getattr(Image, "Resampling", Image)
RESAMPLE_FAST = _Resampling.NEAREST
RESAMPLE_GOOD = _Resampling.BILINEAR

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_CANDIDATES = [SCRIPT_DIR / "rules_config.json"]
DEFAULT_CONFIG_PATH = None  # None -> search CONFIG_CANDIDATES


# ═════════════════════════════════════════════════════════════════
# 1. DECISION ENGINE
# ═════════════════════════════════════════════════════════════════
# Zone naming (matches literature convention):
#   Zone B = interior of the 10% tip band -- limit 1/32", stage-independent
#   Zone C = border along the tip edge + both side edges -- limit 5/32"
#            (stage 1-4) or 1/8" (stage 5-9/10)
#
# Flow: [model: defect detected?] -> location gate -> area gate ->
#       zone classification -> width check
# No detection -> blade is healthy (see handle_no_defect).

def find_config_path(path=None):
    if path:
        return Path(path)
    for candidate in CONFIG_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Regelkonfiguration nicht gefunden. Gesucht in: "
        + ", ".join(str(c) for c in CONFIG_CANDIDATES))


def load_config(path=None):
    with open(find_config_path(path), "r") as f:
        return json.load(f)


def inch_to_mm(inches):
    return inches * 25.4


class BladeGeometry:
    """Everything the engine needs to know about the blade silhouette,
    computed once per image: mask, per-row left/right edges, vertical
    extent, tip point and area. Replaces repeated full-mask scans."""

    def __init__(self, mask):
        self.mask = mask
        self.row_has = mask.any(axis=1)
        if not self.row_has.any():
            raise ValueError(
                "Keine Schaufel im Bild gefunden (das ganze Bild wirkt wie "
                "Hintergrund). Bild und Weißschwelle prüfen.")
        rows = np.where(self.row_has)[0]
        self.y_min, self.y_max = int(rows[0]), int(rows[-1])
        self.height = self.y_max - self.y_min
        self.shape = mask.shape
        width = mask.shape[1]
        self.left = mask.argmax(axis=1)
        self.right = width - 1 - mask[:, ::-1].argmax(axis=1)
        self.area = int(mask.sum())
        x_tip = int(round((self.left[self.y_min] + self.right[self.y_min]) / 2))
        self.tip = (x_tip, self.y_min)

    def band(self, tip_zone_fraction):
        """(band_top, band_bottom, blade_height) of the tip band."""
        return (self.y_min,
                self.y_min + tip_zone_fraction * self.height,
                self.height)

    def edges(self, y):
        """Leftmost / rightmost blade pixel in row y (None, None if empty)."""
        y = max(0, min(int(y), self.shape[0] - 1))
        if not self.row_has[y]:
            return None, None
        return int(self.left[y]), int(self.right[y])


def band_bounds(geom, tip_zone_fraction):
    return geom.band(tip_zone_fraction)


def get_tip_point(geom):
    """Highest point of the blade silhouette: topmost blade row, x =
    midpoint of that row's blade extent. Reference origin for
    defect-location-relative-to-tip reporting."""
    return geom.tip


def local_edges(geom, y):
    return geom.edges(y)


def compute_scale_factor(image_height_px, blade_length_mm):
    """px per mm. image_height_px should be the blade's own measured
    pixel height, not the full image height -- correct regardless of
    camera framing/fill fraction."""
    return image_height_px / blade_length_mm


def compute_tip_offset(bbox_px, tip_x, tip_y, scale_factor):
    """Defect bbox centroid relative to the blade's highest tip point.
    Signed internally (dy > 0 = toward the root; dx > 0 = right of the
    tip in image coordinates); the UI displays them as distances."""
    x_min, y_min, x_max, y_max = bbox_px
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    dx_px = cx - tip_x
    dy_px = cy - tip_y
    return (dx_px, dy_px), (dx_px / scale_factor, dy_px / scale_factor)


def log_spaced_lengths(L_min=82.0, L_max=2500.0, N=10):
    """Identical formula to the render script -- single source of truth
    for what the 10 synthetic sizes are."""
    return [round(L_min * (L_max / L_min) ** (i / (N - 1)), 1) for i in range(N)]


DEFAULT_LENGTHS_MM = log_spaced_lengths()


def stage_from_length(blade_length_mm, lengths=DEFAULT_LENGTHS_MM):
    """Nearest-match lookup: 1-based index of the closest known size.

    SWAP POINT for real-world use: replace with a real lookup driven by
    an actual depth-camera measurement + a real stage/size table."""
    closest_idx = min(range(len(lengths)), key=lambda i: abs(lengths[i] - blade_length_mm))
    return closest_idx + 1


def parse_length_tag(ltag):
    """Reverses fmt_len() from the render script: '137p2mm' -> 137.2"""
    return float(ltag.replace("mm", "").replace("p", "."))


def parse_render_filename(filename):
    """Splits '<stem>__Yneg__137p2mm__B_three_point_dome.png' into parts.
    blade_length_mm is None if no tag matches."""
    stem = filename.rsplit(".", 1)[0]
    parts = stem.split("__")
    length_mm = None
    for p in parts:
        if re.fullmatch(r"\d+p?\d*mm", p):
            length_mm = parse_length_tag(p)
            break
    return {"parts": parts, "blade_length_mm": length_mm}


def classify_location(bbox_px, geom, tip_zone_fraction):
    """'within_tip_zone' or 'outside_tip_zone'.

    ASSUMPTION: any overlap into the 10% tip band counts as
    'within_tip_zone' (conservative)."""
    _, y_min, _, _ = bbox_px
    _, band_bottom, _ = geom.band(tip_zone_fraction)
    if y_min >= band_bottom:
        return "outside_tip_zone"
    return "within_tip_zone"


def classify_zone(bbox_px, geom, tip_zone_fraction, top_margin_fraction, side_margin_fraction):
    """'zone_b' (interior), 'zone_c' (border), or 'spans_b_c'."""
    x_min, y_min, x_max, y_max = bbox_px
    band_top, band_bottom, _ = geom.band(tip_zone_fraction)
    top_border_y = band_top + top_margin_fraction * (band_bottom - band_top)

    touches_c = False  # border
    touches_b = False  # interior

    if y_min < top_border_y:
        touches_c = True

    if y_max >= top_border_y:
        y_start = max(y_min, top_border_y)
        ys = np.arange(int(y_start), int(y_max) + 1)
        ys = np.clip(ys, 0, geom.shape[0] - 1)
        ys = ys[geom.row_has[ys]]
        if len(ys):
            x_left = geom.left[ys].astype(float)
            x_right = geom.right[ys].astype(float)
            side_margin_px = side_margin_fraction * (x_right - x_left)
            b_left_end = x_left + side_margin_px
            b_right_start = x_right - side_margin_px

            row_x_min = np.maximum(x_min, x_left)
            row_x_max = np.minimum(x_max, x_right)

            if np.any((row_x_min < b_left_end) | (row_x_max > b_right_start)):
                touches_c = True
            if np.any((row_x_max > b_left_end) & (row_x_min < b_right_start)):
                touches_b = True

    if touches_b and touches_c:
        return "spans_b_c"
    if touches_c:
        return "zone_c"
    return "zone_b"


def compute_area_fraction(defect_area_px, blade_area_px):
    return defect_area_px / blade_area_px


def compute_width_mm(bbox_px, scale_factor):
    """Horizontal bounding-box extent (tip up / root down orientation)."""
    x_min, _, x_max, _ = bbox_px
    return (x_max - x_min) / scale_factor


def get_width_limit_mm(zone, stage, config):
    limits = config["width_limits_inch"]
    if zone == "spans_b_c":
        zone = "zone_b"  # stricter of the two governs

    if zone == "zone_b":
        return inch_to_mm(limits["zone_b"])

    if zone == "zone_c":
        breakpoint_stage = config["stage_breakpoint"]
        if stage <= breakpoint_stage:
            return inch_to_mm(limits["zone_c"]["stage_1_4"])
        return inch_to_mm(limits["zone_c"]["stage_5_9"])

    raise ValueError(f"Unknown zone: {zone}")


def decide(detection, config, geom):
    """
    detection expects:
      bbox_px          : (x_min, y_min, x_max, y_max) in pixels
      defect_area_px    : detected defect area in pixels
      blade_area_px     : total blade projected area in pixels (same image)
      blade_length_mm   : real-world blade length this render represents
      stage             : optional -- if omitted, derived from blade_length_mm
    geom: BladeGeometry of the same image.

    Every returned dict carries tip_point_px, bbox_offset_from_tip_px and
    bbox_offset_from_tip_mm regardless of decision outcome.
    """
    bbox_px = detection["bbox_px"]
    blade_length_mm = detection["blade_length_mm"]
    stage = detection.get("stage") or stage_from_length(blade_length_mm)

    tip_zone_fraction = config["tip_zone_fraction"]
    top_margin_fraction = config["top_margin_fraction"]
    side_margin_fraction = config["side_margin_fraction"]
    area_limit = config["global_area_max_fraction"]

    tip_x, tip_y = geom.tip
    scale_factor = compute_scale_factor(max(geom.height, 1), blade_length_mm)
    offset_px, offset_mm = compute_tip_offset(bbox_px, tip_x, tip_y, scale_factor)

    base = {
        "stage": stage,
        "tip_point_px": (tip_x, tip_y),
        "bbox_offset_from_tip_px": offset_px,
        "bbox_offset_from_tip_mm": offset_mm,
    }

    location = classify_location(bbox_px, geom, tip_zone_fraction)
    if location == "outside_tip_zone":
        return {
            **base, "decision": "scrap", "reason": "outside_10pct_tip_zone",
            "zone": None, "area_fraction": None, "width_mm": None, "width_limit_mm": None,
        }

    area_fraction = compute_area_fraction(detection["defect_area_px"], detection["blade_area_px"])
    if area_fraction >= area_limit:
        return {
            **base, "decision": "scrap", "reason": "area_over_30pct",
            "zone": None, "area_fraction": area_fraction, "width_mm": None, "width_limit_mm": None,
        }

    zone = classify_zone(bbox_px, geom, tip_zone_fraction, top_margin_fraction, side_margin_fraction)

    width_mm = compute_width_mm(bbox_px, scale_factor)
    width_limit_mm = get_width_limit_mm(zone, stage, config)

    decision = "scrap" if width_mm > width_limit_mm else "repair"
    reason = "width_over_limit" if decision == "scrap" else "within_limit"

    return {
        **base, "decision": decision, "reason": reason, "zone": zone,
        "area_fraction": area_fraction, "width_mm": width_mm, "width_limit_mm": width_limit_mm,
    }


def handle_no_defect(image_path=None):
    """Model found no defect -> blade is healthy.

    HOOK for the future manual-annotation branch: this is where the
    viewer-labelling step (healthy / repair / scrap, digitizing real data)
    will plug in. For now it simply returns healthy."""
    return {"decision": "healthy", "reason": "no_defect_detected"}


def overall_verdict(results):
    """Blade-level verdict: scrap if any defect is scrap, else repair if
    any defect is repair, else healthy."""
    decisions = {r["decision"] for r in results}
    if "scrap" in decisions:
        return "scrap"
    if "repair" in decisions:
        return "repair"
    return "healthy"


# ═════════════════════════════════════════════════════════════════
# 2. DETECTION PIPELINE (YOLO -> decision engine)
# ═════════════════════════════════════════════════════════════════
def _close_mask(raw_mask):
    if cv2 is not None:
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(raw_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        return closed.astype(bool)
    return binary_closing(raw_mask, structure=np.ones((5, 5)))


def blade_mask_from_rgb(arr, white_threshold=250):
    """Blade silhouette from a white-background render, closed to bridge
    small gaps and reduced to the single largest connected component."""
    raw_mask = arr.min(axis=2) < white_threshold
    closed_mask = _close_mask(raw_mask)

    labeled, num_components = label(closed_mask)
    if num_components == 0:
        return raw_mask  # nothing found -- BladeGeometry fails loudly

    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0  # background label is never the blade
    return labeled == component_sizes.argmax()


def get_blade_mask(image_path, white_threshold=250):
    img = Image.open(image_path).convert("RGB")
    return blade_mask_from_rgb(np.array(img), white_threshold)


def resolve_blade_length_mm(image_path, blade_length_mm_override):
    """Priority: explicit override > parsed from filename > error.

    SWAP POINT for real-world use: replace with a real depth-camera /
    physical measurement lookup."""
    if blade_length_mm_override is not None:
        return blade_length_mm_override
    parsed = parse_render_filename(os.path.basename(image_path))
    if parsed["blade_length_mm"] is not None:
        return parsed["blade_length_mm"]
    raise ValueError(
        "Schaufellänge (mm) nicht angegeben und nicht aus dem "
        "Dateinamen lesbar (erwartet wird ein '...mm...'-Teil wie im Render-Skript). "
        "Bitte manuell angeben."
    )


def merge_nearby_boxes(detections, gap_threshold_px):
    """Merges same-class boxes whose rectangle-to-rectangle gap (not IoU)
    is <= gap_threshold_px into one covering box. NMS only suppresses
    overlapping boxes; this handles separate-but-close fragments.

    At gap_threshold_px = 0 only overlapping/touching boxes merge, so 0
    is a safe "off" position for the slider.

    Returns [{'bbox_px', 'defect_type', 'confidence', 'merged_count'}]."""
    def rect_gap(a, b):
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        dx = max(bx0 - ax1, ax0 - bx1, 0)
        dy = max(by0 - ay1, ay0 - by1, 0)
        return (dx ** 2 + dy ** 2) ** 0.5

    n = len(detections)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    by_class = {}
    for i, d in enumerate(detections):
        by_class.setdefault(d["defect_type"], []).append(i)

    for idxs in by_class.values():
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                a = detections[idxs[i]]["bbox_px"]
                b = detections[idxs[j]]["bbox_px"]
                if rect_gap(a, b) <= gap_threshold_px:
                    union(idxs[i], idxs[j])

    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    merged = []
    for idxs in clusters.values():
        boxes = [detections[i]["bbox_px"] for i in idxs]
        merged.append({
            "bbox_px": (min(b[0] for b in boxes), min(b[1] for b in boxes),
                        max(b[2] for b in boxes), max(b[3] for b in boxes)),
            "defect_type": detections[idxs[0]]["defect_type"],
            "confidence": max(detections[i]["confidence"] for i in idxs),
            "merged_count": len(idxs),
        })
    return merged


_model_cache = {}


def get_model(weights_path):
    """Loads the YOLO model once per weights file (keyed by path + mtime)
    and runs one dummy prediction so the first real run isn't slow."""
    key = (os.path.abspath(weights_path), os.path.getmtime(weights_path))
    model = _model_cache.get(key)
    if model is None:
        from ultralytics import YOLO  # imported here so this file stays
                                      # importable without ultralytics
        model = YOLO(weights_path)
        try:
            model.predict(np.zeros((64, 64, 3), np.uint8), verbose=False)
        except Exception:
            pass  # warm-up is best effort
        _model_cache[key] = model
    return model


def get_raw_detections_yolo(image_path, weights_path, confidence=0.25):
    """Runs the YOLO model only -- no merging, no decision engine.
    Returns [{'bbox_px', 'defect_type', 'confidence'}, ...]."""
    model = get_model(weights_path)
    prediction = model.predict(image_path, conf=confidence, verbose=False)[0]

    boxes = prediction.boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    names = prediction.names

    return [{"bbox_px": tuple(float(v) for v in xyxy[i]),
             "defect_type": names[int(cls_ids[i])],
             "confidence": float(confs[i])}
            for i in range(len(xyxy))]


def evaluate_detections(raw_detections, image_path, config_path=DEFAULT_CONFIG_PATH,
                        blade_length_mm=None, gap_threshold_px=0, geom=None, config=None):
    """Merge step + decision engine, kept separate from model inference
    so a gap-threshold change never needs to re-run the model.

    Pass a precomputed `geom` (BladeGeometry) and `config` to skip
    recomputing them -- the UI does this on every slider move.
    No raw detections -> ([], geom): the blade is healthy."""
    if geom is None:
        geom = BladeGeometry(get_blade_mask(image_path))
    if not raw_detections:
        return [], geom

    if config is None:
        config = load_config(config_path)
    resolved_blade_length_mm = resolve_blade_length_mm(image_path, blade_length_mm)

    merged = merge_nearby_boxes(raw_detections, gap_threshold_px)

    results = []
    for det in merged:
        x_min, y_min, x_max, y_max = det["bbox_px"]
        detection = {
            "bbox_px": det["bbox_px"],
            "defect_area_px": (x_max - x_min) * (y_max - y_min),
            "blade_area_px": geom.area,
            "blade_length_mm": resolved_blade_length_mm,
        }
        decision = decide(detection, config, geom)
        decision.update({
            "defect_type": det["defect_type"],
            "confidence": det["confidence"],
            "bbox_px": det["bbox_px"],
            "merged_count": det["merged_count"],
        })
        results.append(decision)

    return results, geom


def run_pipeline(image_path, weights_path, config_path=DEFAULT_CONFIG_PATH,
                 blade_length_mm=None, confidence=0.25, gap_threshold_px=0):
    """Thin wrapper for CLI use: YOLO -> merge -> decide in one call."""
    raw = get_raw_detections_yolo(image_path, weights_path, confidence=confidence)
    results, _geom = evaluate_detections(
        raw, image_path, config_path=config_path,
        blade_length_mm=blade_length_mm, gap_threshold_px=gap_threshold_px)
    return results


def run_pipeline_with_stub_detections(image_path, stub_detections,
                                      config_path=DEFAULT_CONFIG_PATH,
                                      blade_length_mm=None, gap_threshold_px=0):
    """Same as run_pipeline() but skips YOLO -- takes pre-made detections.

    stub_detections: list of {'bbox_px': (...), 'defect_type': str, 'confidence': float}"""
    results, _geom = evaluate_detections(
        stub_detections, image_path, config_path=config_path,
        blade_length_mm=blade_length_mm, gap_threshold_px=gap_threshold_px)
    return results


def bbox_size_px(bbox_px):
    """(width_px, height_px) of a bbox -- x-extent, y-extent."""
    x0, y0, x1, y1 = bbox_px
    return x1 - x0, y1 - y0


def is_erosion_type(defect_type):
    return (defect_type or "").strip().lower() == "erosion"


# ═════════════════════════════════════════════════════════════════
# 3. UI
# ═════════════════════════════════════════════════════════════════
# Light theme: off greyish white.
BG = "#f2f1ee"
CANVAS_RGB = (232, 231, 227)    # area around the image inside the view
CANVAS_BG = "#e8e7e3"
PANEL_BG = "#e4e3df"
FG = "#4a4d52"
TEXT_STRONG = "#1f2328"
ACCENT = "#2f5d8a"              # UI chrome accent (buttons, headers)
BORDER = "#c9c8c3"
BUTTON_BG = "#dad9d4"
BUTTON_ACTIVE = "#c9c8c3"
ENTRY_BG = "#fbfbf9"
TIP_COLOR = "#0090c0"           # tip-point crosshair
GREEN = "#5fbf6a"               # verdict: repair / healthy
RED = "#e05a5a"                 # verdict: scrap
FONT = ("Courier New", 10)
FONT_BOLD = ("Courier New", 10, "bold")

# Zone tint colors (see build_zone_overlay) and the names used in the legend.
ZONE_C_RGB = (65, 152, 100)     # green -> Zone C (border)
ZONE_B_RGB = (81, 113, 137)     # blue  -> Zone B (interior)
ZONE_ALPHA = 0.55
ZONE_LEGEND_NAMES = {"C": "Zone C (Rand)", "B": "Zone B (Innenbereich)"}

# Colors assigned per defect *class* (stable across re-runs and class
# filtering). Lines/box stay per-class; on-image TEXT is black.
# Dark shades so they stay visible on the white background.
ANNOTATION_PALETTE = ["#e69500", "#d6336c", "#6f5de7", "#1f9d6b", "#e8590c", "#8a8a00"]

VERDICT_BOX_WIDTH = 1
ANNOTATION_LINE_WIDTH = 1

# How far the X (above) and Y (left) dimension lines sit from the blade,
# in px, and how far a size label is pushed past the blade edge.
DIM_LINE_MARGIN = 65
SIZE_LABEL_MARGIN = 10

# On-image text: bold black with a thin white stroke for legibility.
LABEL_TEXT_COLOR = "black"
LABEL_STROKE_COLOR = "white"
LABEL_STROKE_WIDTH = 2
LABEL_FONT_SIZE = 16

SLIDER_DEBOUNCE_MS = 80

STUB_DETECTIONS = [
    {"bbox_px": (152, 85, 170, 101), "defect_type": "Crack", "confidence": 0.91},
    {"bbox_px": (410, 130, 424, 142), "defect_type": "Cooling hole", "confidence": 0.87},
    {"bbox_px": (300, 210, 318, 226), "defect_type": "Cooling hole", "confidence": 0.83},
    {"bbox_px": (220, 260, 268, 278), "defect_type": "Erosion", "confidence": 0.79},
]

_label_font_cache = {}


# German display texts for engine codes and class names. Class names come
# from the trained model (English); extend CLASS_DISPLAY_NAMES to match
# your classes. The logic always uses the original class name.
ZONE_DE = {"zone_b": "Zone B", "zone_c": "Zone C", "spans_b_c": "Zone B+C"}
REASON_DE = {
    "outside_10pct_tip_zone": "außerhalb der 10 %-Spitzenzone",
    "area_over_30pct": "Fläche über 30 %",
    "width_over_limit": "Breite über Grenzwert",
    "within_limit": "innerhalb des Grenzwerts",
}
VERDICT_DE = {"healthy": "INTAKT", "repair": "REPARATUR", "scrap": "AUSSCHUSS"}
CLASS_DISPLAY_NAMES = {
    "Crack": "Riss",
    "Erosion": "Erosion",
    "Dent": "Delle",
    "Cooling hole": "Kühlbohrung",
}


def display_class(name):
    return CLASS_DISPLAY_NAMES.get(name, name)


def _rgb_to_hex(rgb):
    return "#%02x%02x%02x" % tuple(int(v) for v in rgb)


def _load_label_font(size=LABEL_FONT_SIZE):
    """Bold TrueType font for on-image labels, with fallbacks. Cached."""
    if size in _label_font_cache:
        return _label_font_cache[size]
    candidates = [
        "DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "Arial Bold.ttf",
        "arialbd.ttf",
    ]
    font = None
    for name in candidates:
        try:
            font = ImageFont.truetype(name, size)
            break
        except Exception:
            continue
    if font is None:
        try:
            font = ImageFont.load_default(size=size)  # Pillow >= 10.1
        except TypeError:
            font = ImageFont.load_default()
    _label_font_cache[size] = font
    return font


def _draw_label(draw, xy, text, fill):
    """One on-image text label, bold black with a white stroke halo."""
    draw.multiline_text(
        xy, text, fill=LABEL_TEXT_COLOR, font=_load_label_font(),
        stroke_width=LABEL_STROKE_WIDTH, stroke_fill=LABEL_STROKE_COLOR)


def _draw_dashed_line(draw, xy, fill, width=ANNOTATION_LINE_WIDTH, dash=8, gap=5):
    x0, y0, x1, y1 = xy
    length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    if length == 0:
        return
    dx, dy = (x1 - x0) / length, (y1 - y0) / length
    pos, draw_on = 0.0, True
    while pos < length:
        seg = min(dash if draw_on else gap, length - pos)
        if draw_on:
            sx, sy = x0 + dx * pos, y0 + dy * pos
            ex, ey = x0 + dx * (pos + seg), y0 + dy * (pos + seg)
            draw.line([(sx, sy), (ex, ey)], fill=fill, width=width)
        pos += seg
        draw_on = not draw_on


def _draw_dashed_rect(draw, bbox, fill, width=ANNOTATION_LINE_WIDTH, dash=8, gap=5):
    x0, y0, x1, y1 = bbox
    _draw_dashed_line(draw, (x0, y0, x1, y0), fill, width, dash, gap)
    _draw_dashed_line(draw, (x1, y0, x1, y1), fill, width, dash, gap)
    _draw_dashed_line(draw, (x1, y1, x0, y1), fill, width, dash, gap)
    _draw_dashed_line(draw, (x0, y1, x0, y0), fill, width, dash, gap)


def _draw_arrowhead(draw, tip, direction, fill, size=6):
    """direction: 'up' / 'down' / 'left' / 'right' -- filled triangle,
    apex at tip, pointing that way."""
    tx, ty = tip
    if direction == "up":
        pts = [(tx, ty), (tx - size / 2, ty + size), (tx + size / 2, ty + size)]
    elif direction == "down":
        pts = [(tx, ty), (tx - size / 2, ty - size), (tx + size / 2, ty - size)]
    elif direction == "left":
        pts = [(tx, ty), (tx + size, ty - size / 2), (tx + size, ty + size / 2)]
    else:  # right
        pts = [(tx, ty), (tx - size, ty - size / 2), (tx - size, ty + size / 2)]
    draw.polygon(pts, fill=fill)


def draw_defect_markers(draw, tip_point, bbox_px, color, is_erosion, geom, show_position):
    """Always-on dashed box around the defect. For non-Erosion classes
    with a known tip point, also draws two dimension lines:
      X -- horizontal, above the blade, tip_x vs. the defect's x-center.
      Y -- vertical, left of the blade, tip_y vs. the defect's lower edge.
    Values are labeled on the lines only when show_position is True,
    and are distances (always positive)."""
    x0, y0, x1, y1 = bbox_px
    pad = 4
    _draw_dashed_rect(draw, (x0 - pad, y0 - pad, x1 + pad, y1 + pad), color)

    if is_erosion or tip_point is None:
        return

    tip_x, tip_y = tip_point
    defect_y = y1
    defect_x_ref = (x0 + x1) / 2.0

    # ---- Y: vertical dimension, off to the left of the blade ----
    left_at_tip = geom.edges(tip_y)[0]
    left_at_defect = geom.edges(defect_y)[0]
    blade_left = min(v for v in (left_at_tip, left_at_defect, x0) if v is not None)
    y_dim_x = max(5, blade_left - DIM_LINE_MARGIN)

    _draw_dashed_line(draw, (y_dim_x, tip_y, tip_x, tip_y), color)
    _draw_dashed_line(draw, (y_dim_x, defect_y, x0, defect_y), color)
    _draw_dashed_line(draw, (y_dim_x, tip_y, y_dim_x, defect_y), color)
    _draw_arrowhead(draw, (y_dim_x, tip_y), "up" if tip_y <= defect_y else "down", color)
    _draw_arrowhead(draw, (y_dim_x, defect_y), "down" if tip_y <= defect_y else "up", color)

    # ---- X: horizontal dimension, off above the blade ----
    x_dim_y = max(5, tip_y - DIM_LINE_MARGIN)

    _draw_dashed_line(draw, (tip_x, x_dim_y, tip_x, tip_y), color)
    _draw_dashed_line(draw, (defect_x_ref, x_dim_y, defect_x_ref, defect_y), color)
    _draw_dashed_line(draw, (tip_x, x_dim_y, defect_x_ref, x_dim_y), color)
    _draw_arrowhead(draw, (tip_x, x_dim_y), "left" if tip_x <= defect_x_ref else "right", color)
    _draw_arrowhead(draw, (defect_x_ref, x_dim_y), "right" if tip_x <= defect_x_ref else "left", color)

    if show_position:
        dy_val = abs(defect_y - tip_y)
        dx_val = abs(defect_x_ref - tip_x)
        y_mid = (tip_y + defect_y) / 2
        _draw_label(draw, (max(2, y_dim_x - 80), y_mid - 10), f"Y={dy_val:.0f}px", color)
        x_mid = (tip_x + defect_x_ref) / 2
        _draw_label(draw, (max(2, x_mid - 34), max(2, x_dim_y - 26)), f"X={dx_val:.0f}px", color)


def build_size_label(det, is_erosion):
    """Text for the defect's own size -- Width/Length for Erosion,
    generic WxH for everything else."""
    w_px, h_px = bbox_size_px(det["bbox_px"])
    dtype = display_class(det.get("defect_type", "Defekt"))
    if is_erosion:
        return f"{dtype}\nB={w_px:.0f}px  L={h_px:.0f}px"
    return f"{dtype}\nGröße {w_px:.0f}x{h_px:.0f}px"


def _tint_span(arr, y, x0, x1, color, alpha):
    """Alpha-blends a horizontal span of one row in place (uint8)."""
    if x1 <= x0:
        return
    seg = arr[y, x0:x1].astype(np.float32)
    arr[y, x0:x1] = (seg * (1 - alpha) + color * alpha).astype(np.uint8)


def build_zone_overlay(img, geom, config):
    """Tints Zone C (border, green) and Zone B (interior, blue) onto the
    blade, using the same geometry the decision engine checks defects
    against. Only the tip-band rows are touched. Built once per image."""
    if config is None:
        return img.copy()

    arr = np.array(img)  # uint8 copy

    tip_zone_fraction = config["tip_zone_fraction"]
    top_margin_fraction = config["top_margin_fraction"]
    side_margin_fraction = config["side_margin_fraction"]

    band_top, band_bottom, _ = geom.band(tip_zone_fraction)
    top_border_y = band_top + top_margin_fraction * (band_bottom - band_top)

    color_c = np.array(ZONE_C_RGB, dtype=np.float32)   # green -> Zone C
    color_b = np.array(ZONE_B_RGB, dtype=np.float32)   # blue  -> Zone B
    alpha = ZONE_ALPHA

    for y in range(int(band_top), min(int(band_bottom), geom.shape[0])):
        if not geom.row_has[y]:
            continue
        x_left, x_right = int(geom.left[y]), int(geom.right[y])

        if y < top_border_y:
            _tint_span(arr, y, x_left, x_right + 1, color_c, alpha)
        else:
            side_margin_px = int(side_margin_fraction * (x_right - x_left))
            b_left_end = x_left + side_margin_px
            b_right_start = x_right - side_margin_px
            _tint_span(arr, y, x_left, b_left_end, color_c, alpha)
            _tint_span(arr, y, b_right_start, x_right + 1, color_c, alpha)
            _tint_span(arr, y, b_left_end, b_right_start, color_b, alpha)

    return Image.fromarray(arr)


def legend_tint(zone_rgb, blade_grey=(160, 164, 170)):
    """Approximate on-screen color of a zone tint over the blade body,
    so the legend swatch matches what the viewer actually sees."""
    return tuple(int(round((1 - ZONE_ALPHA) * g + ZONE_ALPHA * z))
                 for g, z in zip(blade_grey, zone_rgb))


# Colors of the verdict badges inside the exported result box.
EXPORT_BADGE = {
    "healthy": ("#dcefe1", "#1d6b34"),
    "repair": ("#dcefe1", "#1d6b34"),
    "scrap": ("#f6dcdc", "#9b2a2a"),
}
_regular_font_cache = {}


def _load_regular_font(size):
    """Regular (non-bold) TrueType font for the result box, with fallbacks."""
    if size in _regular_font_cache:
        return _regular_font_cache[size]
    font = None
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "segoeui.ttf", "arial.ttf"):
        try:
            font = ImageFont.truetype(name, size)
            break
        except Exception:
            continue
    if font is None:
        try:
            font = ImageFont.load_default(size=size)
        except TypeError:
            font = ImageFont.load_default()
    _regular_font_cache[size] = font
    return font


def _wrap_text(draw, text, font, max_w):
    """Greedy word wrap on normal spaces (non-breaking spaces stay together)."""
    lines, cur = [], ""
    for word in text.split(" "):
        trial = (cur + " " + word).strip()
        if cur and draw.textlength(trial, font=font) > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def _rounded_box(draw, xy, radius, fill, outline=None, width=1):
    try:
        draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)
    except AttributeError:  # old Pillow
        draw.rectangle(xy, fill=fill, outline=outline, width=width)


def _content_bbox(img, thresh=250):
    """Bounding box (l, t, r, b) of everything that is not white background."""
    arr = np.asarray(img.convert("RGB"))
    mask = arr.min(axis=2) < thresh
    ys = np.where(mask.any(axis=1))[0]
    xs = np.where(mask.any(axis=0))[0]
    if len(ys) == 0:
        return 0, 0, img.width, img.height
    return int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1


def _trim(img, margin):
    """Crops the image to its content and adds exactly `margin` px of white on
    every side. The margin is painted even where the content touched the
    original edge, so text is never cut by the border."""
    l, t, r, b = _content_bbox(img)
    out = Image.new("RGB", (r - l + 2 * margin, b - t + 2 * margin), (255, 255, 255))
    out.paste(img.crop((l, t, r, b)), (margin, margin))
    return out


def draw_result_box(img, results, verdict, class_colors, geom=None, hidden=0):
    """Adds the decision as a box to the exported image: overall verdict plus,
    per defect, class, zone/stage, width against limit and the reason.

    The box goes to the bottom right if that area is free (checked on the
    pixels, so it never covers the blade or a label); otherwise the sheet is
    widened to the right. Text size matches the on-image labels.
    Returns a new image (the input is not changed). `geom` is unused and kept
    only so older calls still work."""
    W, H = img.size
    fs = max(12, int(LABEL_FONT_SIZE * 0.95))
    f_body = _load_regular_font(fs)
    f_small = _load_regular_font(max(10, int(fs * 0.85)))
    f_bold = _load_label_font(fs)
    f_head = _load_label_font(int(fs * 1.2))
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    pad, margin = int(fs * 0.9), max(10, int(fs * 0.6))
    inner_w = int(fs * 17)
    box_w = inner_w + 2 * pad
    line_h = int(fs * 1.35)
    sep = int(fs * 0.6)

    def build(n_show):
        """Rows of the box for the first n_show defects: (kind, data, height)."""
        rows = [("small", "Bewertung der Schaufel", int(line_h * 0.9)),
                ("verdict", verdict, int(line_h * 1.5))]
        if not results:
            rows.append(("line", ["Kein Defekt erkannt"], line_h))
        for r in results[:n_show]:
            rows.append(("gap", None, sep))
            title = "%s  (Konf. %.2f)" % (display_class(r["defect_type"]), r.get("confidence", 0))
            rows.append(("title", (title, r), line_h))
            zone = ZONE_DE.get(r.get("zone"))
            stage = r.get("stage", "-")
            texts = ["%s · Stufe %s" % (zone, stage) if zone else "Stufe %s" % stage]
            if r.get("width_mm") is not None:
                texts.append("Breite %.2f mm (Grenzwert %.2f mm)" % (r["width_mm"], r["width_limit_mm"]))
            if r.get("area_fraction") is not None and r.get("width_mm") is None:
                texts.append("Fläche %.2f %% der Schaufel" % (r["area_fraction"] * 100))
            texts.append("Grund: " + REASON_DE.get(r.get("reason"), str(r.get("reason", "-"))))
            for t in texts:
                for ln in _wrap_text(probe, t, f_body, inner_w):
                    rows.append(("line", [ln], line_h))
        if len(results) > n_show:
            rows.append(("gap", None, sep))
            rows.append(("small", "+ %d weitere Defekte" % (len(results) - n_show), line_h))
        if hidden:
            rows.append(("gap", None, sep // 2))
            rows.append(("small", "(%d Defekt(e) ausgeblendet)" % hidden, line_h))
        return rows

    # as many defects as fit into a box no taller than ~1.5x the image
    n_show = len(results)
    rows = build(n_show)
    while n_show > 1 and 2 * pad + sum(r[2] for r in rows) > int(H * 1.5):
        n_show -= 1
        rows = build(n_show)
    box_h = 2 * pad + sum(r[2] for r in rows)

    # does the bottom-right corner hold the box without touching any content?
    gap = max(6, fs // 2)
    mask = np.asarray(img.convert("RGB")).min(axis=2) < 250
    x0 = W - margin - box_w
    y0 = H - margin - box_h
    fits = x0 >= 0 and y0 >= 0 and not mask[max(0, y0 - gap):min(H, y0 + box_h + gap),
                                            max(0, x0 - gap):min(W, x0 + box_w + gap)].any()
    if fits:
        out = img.copy()
    else:                                   # widen the sheet to the right
        out_h = max(H, box_h + 2 * margin)
        out = Image.new("RGB", (W + margin + box_w + margin, out_h), (255, 255, 255))
        out.paste(img, (0, 0))
        x0 = W + margin
        y0 = out_h - margin - box_h
    y1 = y0 + box_h

    d = ImageDraw.Draw(out)
    _rounded_box(d, (x0, y0, x0 + box_w, y1), int(fs * 0.6), (252, 252, 250),
                 (150, 150, 150), max(1, fs // 9))
    y = y0 + pad
    xt = x0 + pad
    for kind, data, h in rows:
        if kind == "small":
            d.text((xt, y), data, fill=(95, 95, 100), font=f_small)
        elif kind == "verdict":
            bg, fg = EXPORT_BADGE[data]
            label = VERDICT_DE[data]
            tw = int(d.textlength(label, font=f_head))
            _rounded_box(d, (xt, y, xt + tw + 2 * pad, y + h - 4), int(fs * 0.5), bg)
            d.text((xt + pad, y + (h - 4) // 2), label, fill=fg, font=f_head, anchor="lm")
        elif kind == "title":
            title, r = data
            dot = int(fs * 0.7)
            d.rectangle([xt, y + (h - dot) // 2, xt + dot, y + (h + dot) // 2],
                        fill=class_colors.get(r["defect_type"], "#888888"))
            d.text((xt + dot + 8, y + h // 2), title, fill=(30, 30, 30), font=f_bold, anchor="lm")
            is_repair = r["decision"] == "repair"
            bg, fg = EXPORT_BADGE["repair" if is_repair else "scrap"]
            label = "Reparatur" if is_repair else "Ausschuss"
            tw = int(d.textlength(label, font=f_small))
            bx1 = x0 + box_w - pad
            _rounded_box(d, (bx1 - tw - 14, y + 2, bx1, y + h - 2), int(fs * 0.4), bg)
            d.text((bx1 - 7 - tw // 2, y + h // 2), label, fill=fg, font=f_small, anchor="mm")
        elif kind == "line":
            d.text((xt, y), data[0], fill=(40, 40, 45), font=f_body)
        y += h
    return out


def compose_export_image(img, with_verdicts, result_box=None):
    """Export image: the annotated blade cropped to its content (no empty
    white around it), the result box, and a legend strip below. Every side
    keeps a small white margin so nothing touches the border. Used by
    'Annotiertes Bild speichern'."""
    margin = 14
    row_h, s = 24, 14
    font = _load_label_font(14)

    img = _trim(img, margin)
    if result_box:
        img = draw_result_box(img, **result_box)
        img = _trim(img, margin)
    w, h = img.size

    items = [
        ("swatch", legend_tint(ZONE_C_RGB), ZONE_LEGEND_NAMES["C"]),
        ("swatch", legend_tint(ZONE_B_RGB), ZONE_LEGEND_NAMES["B"]),
        ("cross", TIP_COLOR, "Spitzenpunkt"),
    ]
    if with_verdicts:
        items += [("outline", GREEN, "Reparatur"), ("outline", RED, "Ausschuss")]

    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    def item_width(text):
        try:
            text_w = probe.textlength(text, font=font)
        except AttributeError:
            text_w = 8 * len(text)
        return s + 6 + int(text_w) + 24

    # the legend wraps onto more rows when the image is narrow
    max_x = w - margin
    rows, x_probe = 1, margin
    for _kind, _color, text in items:
        if x_probe + item_width(text) - 24 > max_x and x_probe > margin:
            rows += 1
            x_probe = margin
        x_probe += item_width(text)

    legend_h = rows * row_h + margin // 2
    out = Image.new("RGB", (w, h + legend_h), (255, 255, 255))
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)

    x, y = margin, h - margin // 2
    for kind, color, text in items:
        if x + item_width(text) - 24 > max_x and x > margin:
            x = margin
            y += row_h
        if kind == "swatch":
            draw.rectangle([x, y, x + s, y + s], fill=color, outline=(120, 120, 120))
        elif kind == "outline":
            draw.rectangle([x, y, x + s, y + s], outline=color, width=2)
        else:
            draw.line([(x, y + s // 2), (x + s, y + s // 2)], fill=color, width=2)
            draw.line([(x + s // 2, y), (x + s // 2, y + s)], fill=color, width=2)
        draw.text((x + s + 6, y - 1), text, fill=(30, 30, 30), font=font)
        try:
            text_w = draw.textlength(text, font=font)
        except AttributeError:
            text_w = 8 * len(text)
        x += s + 6 + int(text_w) + 24
    return out


def short_name(path, max_len=28):
    """File name only, shortened -- keeps the group boxes a fixed width."""
    name = Path(path).name
    return name if len(name) <= max_len else name[:max_len - 3] + "..."


def truncate_path(path, max_len=68):
    """Keeps the tail of a path and collapses the front with '...'."""
    if len(path) <= max_len:
        return path
    return "..." + path[-(max_len - 3):]


class InspectionApp:
    MIN_ZOOM = 1.0
    MAX_ZOOM = 20.0

    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk
        self._tk = tk
        self._ttk = ttk

        self.root = root
        self.root.title("Inspektions-Entscheidungshilfe")
        self.root.configure(bg=BG)
        self.root.geometry("1100x800")
        self.root.minsize(1000, 700)
        self.root.resizable(True, True)

        self.is_fullscreen = False
        self.root.bind("<F11>", self._toggle_fullscreen)
        self.root.bind("<Escape>", self._exit_fullscreen)

        self.image_path = tk.StringVar()
        self.weights_path = tk.StringVar()
        self.image_path_display = tk.StringVar(value="(kein Bild geladen)")
        self.weights_path_display = tk.StringVar(value="(kein Modell -- Testmodus)")
        self.blade_length_mm = tk.StringVar()
        self.status = tk.StringVar(value="Zum Start ein Bild laden.")
        self.show_position_values = tk.BooleanVar(value=True)
        self.show_size_values = tk.BooleanVar(value=True)

        self.tk_image = None
        self.results = []
        self.raw_detections = None  # cached model output (None = not run yet,
                                    # [] = model found nothing -> healthy)
        self.gap_threshold_px = tk.IntVar(value=0)
        self._blade_length_mm_used = None
        self._has_verdict_boxes = False  # whether the current view shows boxes

        # per-image caches (set in _pick_image)
        self.geom = None
        self.base_overlay = None

        # background detection
        self._worker_queue = queue.Queue()
        self._running = False
        self._slider_job = None

        # per-class filter state
        self.class_vars = {}
        self.class_colors = {}

        self.config = None
        self._config_error = None
        try:
            self.config = load_config(DEFAULT_CONFIG_PATH)
        except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
            self._config_error = str(e)

        # canvas zoom/pan state -- all in full-resolution image coords
        self._current_display_img = None
        self._fit_scale = 1.0
        self._zoom = 1.0
        self._view_cx = None
        self._view_cy = None
        self._pan_start = None

        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TLabel", background=BG, foreground=FG, font=FONT)
        style.configure("TButton", background=BUTTON_BG, foreground=ACCENT,
                        font=FONT_BOLD, padding=4)
        style.configure("TEntry", fieldbackground=ENTRY_BG, foreground=TEXT_STRONG, font=FONT)
        style.configure("TCheckbutton", background=BG, foreground=FG, font=FONT)
        style.map("TButton", background=[("active", BUTTON_ACTIVE)])
        style.map("TCheckbutton", background=[("active", BG)])
        style.configure("Primary.TButton", background=ACCENT, foreground="#ffffff",
                        font=FONT_BOLD, padding=4)
        style.map("Primary.TButton",
                  background=[("active", "#274e75"), ("disabled", "#9db3c9")],
                  foreground=[("disabled", "#f2f1ee")])

        self._build_top_bar()
        self._build_content_area()

        if self._config_error:
            self.status.set("WARNUNG: Regelkonfiguration nicht gefunden -- Zonen und Entscheidungen "
                            "sind nicht verfügbar. " + truncate_path(self._config_error, 60))

    def _toggle_fullscreen(self, event=None):
        self.is_fullscreen = not self.is_fullscreen
        self.root.attributes("-fullscreen", self.is_fullscreen)

    def _exit_fullscreen(self, event=None):
        self.is_fullscreen = False
        self.root.attributes("-fullscreen", False)

    def _build_top_bar(self):
        tk, ttk = self._tk, self._ttk
        bar = tk.Frame(self.root, bg=PANEL_BG)
        bar.pack(fill="x")
        tk.Label(bar, text="  Inspektions-Entscheidungshilfe", bg=PANEL_BG,
                 fg=ACCENT, font=("Courier New", 13, "bold")).pack(side="left", padx=6, pady=8)

        # Three groups of equal width, left to right in working order:
        # 1 Eingabe -> 2 Ausführen -> 3 Anzeige. Every button fills its group
        # and sits in a fixed row, so nothing moves when a file name changes.
        controls = tk.Frame(self.root, bg=BG)
        controls.pack(fill="x", padx=12, pady=(10, 4))
        for col in range(3):
            controls.columnconfigure(col, weight=1, uniform="groups")

        def group(col, title):
            frame = tk.Frame(controls, bg=BG, highlightbackground=BORDER, highlightthickness=1)
            frame.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 8, 0))
            head = tk.Frame(frame, bg=PANEL_BG)
            head.pack(fill="x")
            tk.Label(head, text=title, bg=PANEL_BG, fg=ACCENT,
                     font=FONT_BOLD).pack(side="left", padx=8, pady=3)
            body = tk.Frame(frame, bg=BG)
            body.pack(fill="both", expand=True, padx=8, pady=8)
            body.columnconfigure(0, weight=1)
            return body

        def button(body, row, text, command, style="TButton", pady=(0, 4)):
            btn = ttk.Button(body, text=text, command=command, style=style)
            btn.grid(row=row, column=0, sticky="ew", pady=pady)
            return btn

        def label(body, row, text=None, textvariable=None, pady=(0, 8)):
            lbl = ttk.Label(body, anchor="w")
            if textvariable is not None:
                lbl.configure(textvariable=textvariable)
            else:
                lbl.configure(text=text)
            lbl.grid(row=row, column=0, sticky="ew", pady=pady)
            return lbl

        # ---- 1  Eingabe ----
        body = group(0, "1  Eingabe")
        button(body, 0, "Bild laden", self._pick_image)
        label(body, 1, textvariable=self.image_path_display)
        button(body, 2, "Detektionsmodell laden (.pt)", self._pick_weights)
        label(body, 3, textvariable=self.weights_path_display, pady=(0, 0))

        # ---- 2  Ausführen ----
        body = group(1, "2  Ausführen")
        label(body, 0, text="Schaufellänge manuell (mm):", pady=(0, 2))
        ttk.Entry(body, textvariable=self.blade_length_mm).grid(
            row=1, column=0, sticky="ew", pady=(0, 8))
        self.run_button = button(body, 2, "Detektion starten", self._run,
                                 style="Primary.TButton")
        button(body, 3, "Annotiertes Bild speichern", self._save_annotated_image,
               pady=(0, 0))

        # ---- 3  Anzeige ----
        body = group(2, "3  Anzeige")
        ttk.Checkbutton(body, text="Positionswerte anzeigen", variable=self.show_position_values,
                        command=self._on_toggle_display_values).grid(
                            row=0, column=0, sticky="w", pady=(0, 2))
        ttk.Checkbutton(body, text="Größenwerte anzeigen", variable=self.show_size_values,
                        command=self._on_toggle_display_values).grid(
                            row=1, column=0, sticky="w", pady=(0, 8))
        button(body, 2, "Ansicht zurücksetzen", self._reset_view)
        label(body, 3, text="Scrollen = Zoom, Ziehen = Verschieben", pady=(0, 0))

        # ---- class filter + status, aligned with the groups above ----
        info = tk.Frame(self.root, bg=BG)
        info.pack(fill="x", padx=12, pady=(6, 0))
        ttk.Label(info, text="Erkannte Klassen:").pack(side="left", padx=(0, 8))
        self.class_filter_container = tk.Frame(info, bg=BG)
        self.class_filter_container.pack(side="left")
        ttk.Label(self.class_filter_container,
                  text="(nach der Detektion gefüllt)").pack(side="left")

        status_bar = tk.Frame(self.root, bg=BG)
        status_bar.pack(fill="x", padx=12, pady=(4, 0))
        ttk.Label(status_bar, textvariable=self.status, anchor="w", justify="left",
                  wraplength=1040).pack(side="left", fill="x")

    def _build_content_area(self):
        tk = self._tk
        ttk = self._ttk
        content = tk.Frame(self.root, bg=BG)
        content.pack(fill="both", expand=True, padx=12, pady=(8, 12))

        left = tk.Frame(content, bg=BG)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self.canvas = tk.Canvas(left, bg=CANVAS_BG, highlightthickness=1,
                                highlightbackground=BORDER)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)   # Windows / macOS
        self.canvas.bind("<Button-4>", self._on_mousewheel)     # Linux scroll up
        self.canvas.bind("<Button-5>", self._on_mousewheel)     # Linux scroll down
        self.canvas.bind("<ButtonPress-1>", self._on_pan_start)
        self.canvas.bind("<B1-Motion>", self._on_pan_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_pan_end)

        self._build_legend(left)

        right = tk.Frame(content, bg=BG, width=460)
        right.pack(side="left", fill="both", expand=True)
        right.pack_propagate(False)

        header = tk.Frame(right, bg=PANEL_BG)
        header.pack(fill="x")
        tk.Label(header, text="  Erkannte Defekte", bg=PANEL_BG, fg=ACCENT,
                 font=FONT_BOLD).pack(side="left", pady=4)

        list_area = tk.Frame(right, bg=BG)
        list_area.pack(fill="both", expand=True, pady=(6, 0))

        self.cards_canvas = tk.Canvas(list_area, bg=BG, highlightthickness=0)
        cards_scrollbar = ttk.Scrollbar(list_area, orient="vertical",
                                        command=self.cards_canvas.yview)
        self.cards_canvas.configure(yscrollcommand=cards_scrollbar.set)
        cards_scrollbar.pack(side="right", fill="y")
        self.cards_canvas.pack(side="left", fill="both", expand=True)

        self.cards_container = tk.Frame(self.cards_canvas, bg=BG)
        self._cards_window = self.cards_canvas.create_window(
            (0, 0), window=self.cards_container, anchor="nw")

        self.cards_container.bind(
            "<Configure>",
            lambda e: self.cards_canvas.configure(scrollregion=self.cards_canvas.bbox("all")))
        self.cards_canvas.bind(
            "<Configure>",
            lambda e: self.cards_canvas.itemconfig(self._cards_window, width=e.width))
        self._bind_wheel(self.cards_canvas)

    def _build_legend(self, parent):
        """Legend under the image view: zone names, tip point, box verdicts."""
        tk = self._tk
        legend = tk.Frame(parent, bg=BG)
        legend.pack(fill="x", pady=(6, 0))

        def swatch_item(color_hex, text, outline_only=False):
            item = tk.Frame(legend, bg=BG)
            item.pack(side="left", padx=(0, 14))
            if outline_only:
                box = tk.Frame(item, bg=BG, width=12, height=12,
                               highlightbackground=color_hex, highlightthickness=2)
            else:
                box = tk.Frame(item, bg=color_hex, width=12, height=12,
                               highlightbackground=BORDER, highlightthickness=1)
            box.pack(side="left", padx=(0, 5))
            tk.Label(item, text=text, bg=BG, fg=FG, font=("Courier New", 9)).pack(side="left")

        swatch_item(_rgb_to_hex(legend_tint(ZONE_C_RGB)), ZONE_LEGEND_NAMES["C"])
        swatch_item(_rgb_to_hex(legend_tint(ZONE_B_RGB)), ZONE_LEGEND_NAMES["B"])

        tip_item = tk.Frame(legend, bg=BG)
        tip_item.pack(side="left", padx=(0, 14))
        tk.Label(tip_item, text="+", bg=BG, fg=TIP_COLOR,
                 font=("Courier New", 11, "bold")).pack(side="left")
        tk.Label(tip_item, text="Spitzenpunkt", bg=BG, fg=FG,
                 font=("Courier New", 9)).pack(side="left")

        swatch_item(GREEN, "Reparatur", outline_only=True)
        swatch_item(RED, "Ausschuss", outline_only=True)

    # ---- result-list scrolling --------------------------------------

    def _bind_wheel(self, widget):
        widget.bind("<MouseWheel>", self._on_cards_mousewheel)
        widget.bind("<Button-4>", self._on_cards_mousewheel)
        widget.bind("<Button-5>", self._on_cards_mousewheel)

    def _bind_wheel_recursive(self, widget):
        """Cards are made of many child widgets; the wheel must be bound
        on all of them or scrolling dies when the cursor is over a card."""
        self._bind_wheel(widget)
        for child in widget.winfo_children():
            self._bind_wheel_recursive(child)

    def _on_cards_mousewheel(self, event):
        if getattr(event, "delta", 0):
            steps = -1 if event.delta > 0 else 1
        elif getattr(event, "num", None) == 4:
            steps = -1
        elif getattr(event, "num", None) == 5:
            steps = 1
        else:
            return
        self.cards_canvas.yview_scroll(steps, "units")

    # ---- zoom / pan -------------------------------------------------

    def _compute_fit_scale(self):
        if self._current_display_img is None:
            return 1.0
        iw, ih = self._current_display_img.size
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 10 or ch < 10 or iw == 0 or ih == 0:
            return 1.0
        return min(cw / iw, ch / ih)

    def _clamp_view(self, eff_scale):
        iw, ih = self._current_display_img.size
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        view_w = min(iw, cw / eff_scale)
        view_h = min(ih, ch / eff_scale)
        self._view_cx = iw / 2 if view_w >= iw else min(max(self._view_cx, view_w / 2), iw - view_w / 2)
        self._view_cy = ih / 2 if view_h >= ih else min(max(self._view_cy, view_h / 2), ih - view_h / 2)

    def _canvas_to_image(self, cx, cy):
        eff_scale = self._fit_scale * self._zoom
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        view_w, view_h = cw / eff_scale, ch / eff_scale
        left, top = self._view_cx - view_w / 2, self._view_cy - view_h / 2
        return left + cx / eff_scale, top + cy / eff_scale

    def _reset_view(self):
        self._zoom = 1.0
        if self._current_display_img is not None:
            iw, ih = self._current_display_img.size
            self._view_cx, self._view_cy = iw / 2, ih / 2
        self._redraw_view()

    def _on_canvas_resize(self, event=None):
        self._redraw_view()

    def _on_mousewheel(self, event):
        if self._current_display_img is None:
            return
        if getattr(event, "delta", 0):
            direction = 1 if event.delta > 0 else -1
        elif getattr(event, "num", None) == 4:
            direction = 1
        elif getattr(event, "num", None) == 5:
            direction = -1
        else:
            return

        old_zoom = self._zoom
        factor = 1.15 if direction > 0 else (1 / 1.15)
        new_zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, old_zoom * factor))
        if new_zoom == old_zoom:
            return

        img_x, img_y = self._canvas_to_image(event.x, event.y)
        self._zoom = new_zoom
        eff_scale = self._fit_scale * self._zoom
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        view_w, view_h = cw / eff_scale, ch / eff_scale
        frac_x, frac_y = event.x / cw, event.y / ch
        self._view_cx = img_x - (frac_x - 0.5) * view_w
        self._view_cy = img_y - (frac_y - 0.5) * view_h
        self._redraw_view()

    def _on_pan_start(self, event):
        self._pan_start = (event.x, event.y, self._view_cx, self._view_cy)

    def _on_pan_move(self, event):
        if self._pan_start is None or self._current_display_img is None:
            return
        sx, sy, cx0, cy0 = self._pan_start
        eff_scale = self._fit_scale * self._zoom
        self._view_cx = cx0 - (event.x - sx) / eff_scale
        self._view_cy = cy0 - (event.y - sy) / eff_scale
        self._redraw_view(fast=True)

    def _on_pan_end(self, event=None):
        was_panning = self._pan_start is not None
        self._pan_start = None
        if was_panning:
            self._redraw_view()  # one sharp redraw after the drag

    def _redraw_view(self, fast=False):
        if self._current_display_img is None or ImageTk is None:
            return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return

        self._fit_scale = self._compute_fit_scale()
        iw, ih = self._current_display_img.size
        if self._view_cx is None:
            self._view_cx, self._view_cy = iw / 2, ih / 2

        eff_scale = self._fit_scale * self._zoom
        self._clamp_view(eff_scale)

        view_w, view_h = cw / eff_scale, ch / eff_scale
        left, top = self._view_cx - view_w / 2, self._view_cy - view_h / 2
        right, bottom = left + view_w, top + view_h

        crop_left, crop_top = max(0.0, left), max(0.0, top)
        crop_right, crop_bottom = min(float(iw), right), min(float(ih), bottom)

        canvas_img = Image.new("RGB", (cw, ch), CANVAS_RGB)
        if crop_right > crop_left and crop_bottom > crop_top:
            cropped = self._current_display_img.crop(
                (crop_left, crop_top, crop_right, crop_bottom))
            disp_w = max(1, round((crop_right - crop_left) * eff_scale))
            disp_h = max(1, round((crop_bottom - crop_top) * eff_scale))
            disp = cropped.resize(
                (disp_w, disp_h), RESAMPLE_FAST if fast else RESAMPLE_GOOD)
            offset_x = round((crop_left - left) * eff_scale)
            offset_y = round((crop_top - top) * eff_scale)
            canvas_img.paste(disp, (offset_x, offset_y))

        self.tk_image = ImageTk.PhotoImage(canvas_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

    # ---- per-class filtering -----------------------------------------

    def _color_for_class(self, defect_type):
        if defect_type not in self.class_colors:
            self.class_colors[defect_type] = ANNOTATION_PALETTE[
                len(self.class_colors) % len(ANNOTATION_PALETTE)]
        return self.class_colors[defect_type]

    def _rebuild_class_filter(self):
        tk, ttk = self._tk, self._ttk
        for child in self.class_filter_container.winfo_children():
            child.destroy()

        if not self.results:
            ttk.Label(self.class_filter_container,
                      text="  (nach der Detektion gefüllt)").pack(side="left")
            return

        counts = {}
        for r in self.results:
            counts[r["defect_type"]] = counts.get(r["defect_type"], 0) + 1

        for defect_type in sorted(counts):
            if defect_type not in self.class_vars:
                self.class_vars[defect_type] = tk.BooleanVar(value=True)
            color = self._color_for_class(defect_type)

            item = tk.Frame(self.class_filter_container, bg=BG)
            item.pack(side="left", padx=(0, 14))
            tk.Frame(item, bg=color, width=10, height=10).pack(side="left", padx=(0, 5))
            ttk.Checkbutton(
                item, text=f"{display_class(defect_type)} ({counts[defect_type]})",
                variable=self.class_vars[defect_type],
                command=self._on_class_filter_change).pack(side="left")

    def _get_visible_results(self):
        visible = []
        for r in self.results:
            var = self.class_vars.get(r["defect_type"])
            if var is None or var.get():
                visible.append(r)
        return visible

    def _on_class_filter_change(self):
        visible = self._get_visible_results()
        self._show_image(detections=visible)
        self._render_result_cards(visible)

    # ---- image loading / display ------------------------------------

    def _load_image_context(self, path):
        """Everything derived from the image alone, computed once:
        blade geometry (mask, edges, tip, area) and the zone overlay."""
        img = Image.open(path).convert("RGB")
        mask = blade_mask_from_rgb(np.array(img))
        geom = BladeGeometry(mask)  # raises ValueError on a blank image
        overlay = build_zone_overlay(img, geom, self.config)
        return geom, overlay

    def _pick_image(self):
        from tkinter import filedialog, messagebox
        path = filedialog.askopenfilename(
            title="Schaufelbild auswählen",
            filetypes=[("Bilder", "*.png *.jpg *.jpeg"), ("Alle", "*.*")])
        if not path:
            return

        try:
            geom, overlay = self._load_image_context(path)
        except Exception as e:
            messagebox.showerror("Fehler", f"Bild konnte nicht geladen werden:\n{e}")
            return

        if self._slider_job is not None:
            self.root.after_cancel(self._slider_job)
            self._slider_job = None

        self.geom, self.base_overlay = geom, overlay
        self.image_path.set(path)
        self.image_path_display.set(short_name(path))
        self.results = []
        self.raw_detections = None
        self.class_vars = {}
        self.class_colors = {}
        self._rebuild_class_filter()
        self._render_result_cards([])
        self._zoom = 1.0
        self._view_cx = None
        self._view_cy = None
        self.status.set("Bild geladen. 'Detektion starten' drücken.")
        self._show_image()

    def _pick_weights(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Detektionsmodell auswählen (YOLO, .pt)",
            filetypes=[("PyTorch-Modell", "*.pt"), ("Alle", "*.*")])
        if path:
            self.weights_path.set(path)
            self.weights_path_display.set(short_name(path))

    def _show_image(self, detections=None):
        if self.base_overlay is None or self.geom is None:
            return
        img = self.base_overlay.copy()
        draw = ImageDraw.Draw(img)
        geom = self.geom

        # Tip-point crosshair (tip is part of the cached geometry).
        tip_point = geom.tip
        tx, ty = tip_point
        r = 8
        draw.line([(tx - r, ty), (tx + r, ty)], fill=TIP_COLOR, width=2)
        draw.line([(tx, ty - r), (tx, ty + r)], fill=TIP_COLOR, width=2)

        show_position = self.show_position_values.get()
        show_size = self.show_size_values.get()

        for det in detections or []:
            is_erosion = is_erosion_type(det.get("defect_type"))
            color = self._color_for_class(det["defect_type"])
            x0, y0, x1, y1 = det["bbox_px"]

            draw_defect_markers(draw, tip_point, det["bbox_px"], color, is_erosion,
                                geom, show_position)

            if show_size:
                size_text = build_size_label(det, is_erosion)
                row = (y0 + y1) / 2
                blade_right = geom.edges(row)[1]
                label_x = max(x1, blade_right if blade_right is not None else x1) \
                    + SIZE_LABEL_MARGIN
                _draw_label(draw, (label_x, y0), size_text, color)

            verdict_color = GREEN if det["decision"] == "repair" else RED
            draw.rectangle([x0, y0, x1, y1], outline=verdict_color, width=VERDICT_BOX_WIDTH)

        self._has_verdict_boxes = bool(detections)
        self._current_display_img = img
        self._redraw_view()

    def _save_annotated_image(self):
        """Exports what is currently shown (respecting the class filter and
        the position / size toggles) at full image resolution, on a white
        sheet with padding and a legend."""
        from tkinter import filedialog, messagebox
        if self._current_display_img is None:
            messagebox.showerror("Fehler", "Zuerst ein Bild laden.")
            return

        stem = Path(self.image_path.get()).stem
        path = filedialog.asksaveasfilename(
            title="Annotiertes Bild speichern", defaultextension=".png",
            initialfile=f"{stem}_annotated.png",
            filetypes=[("PNG-Bild", "*.png"), ("JPEG-Bild", "*.jpg")])
        if not path:
            return
        try:
            result_box = None
            if self.raw_detections is not None:      # detection has been run
                visible = self._get_visible_results()
                result_box = {
                    "results": visible,
                    "verdict": overall_verdict(self.results) if self.raw_detections else "healthy",
                    "class_colors": {r["defect_type"]: self._color_for_class(r["defect_type"])
                                     for r in self.results},
                    "geom": self.geom,
                    "hidden": len(self.results) - len(visible),
                }
            out = compose_export_image(self._current_display_img, self._has_verdict_boxes,
                                       result_box)
            out.save(path)
        except Exception as e:
            messagebox.showerror("Fehler", f"Bild konnte nicht gespeichert werden:\n{e}")
            return
        self.status.set("Annotiertes Bild gespeichert: " + truncate_path(path, 60))

    # ---- running detection (background thread) ----------------------

    def _run(self):
        from tkinter import messagebox
        if self._running:
            return
        if not self.image_path.get():
            messagebox.showerror("Fehler", "Zuerst ein Bild laden.")
            return
        if self.config is None:
            messagebox.showerror(
                "Fehler", "Regelkonfiguration nicht gefunden:\n" + (self._config_error or ""))
            return

        blade_length_mm = None
        if self.blade_length_mm.get().strip():
            try:
                blade_length_mm = float(self.blade_length_mm.get().strip())
            except ValueError:
                messagebox.showerror("Fehler", "Die Schaufellänge muss eine Zahl sein.")
                return
        self._blade_length_mm_used = blade_length_mm

        image_path = self.image_path.get()
        weights_path = self.weights_path.get()

        self._running = True
        self.run_button.state(["disabled"])
        if weights_path:
            self.status.set("Detektionsmodell läuft...")
        else:
            self.status.set("Kein Detektionsmodell geladen -- Testmodus mit Beispieldetektionen.")

        def worker():
            try:
                if weights_path:
                    raw = get_raw_detections_yolo(image_path, weights_path)
                else:
                    raw = [dict(d) for d in STUB_DETECTIONS]
                self._worker_queue.put((image_path, raw, None))
            except Exception as e:
                self._worker_queue.put((image_path, None, e))

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(50, self._poll_worker)

    def _poll_worker(self):
        """Tk is not thread-safe, so the worker only posts to a queue and
        this main-thread poll picks the result up."""
        try:
            image_path, raw, error = self._worker_queue.get_nowait()
        except queue.Empty:
            self.root.after(50, self._poll_worker)
            return

        self._running = False
        self.run_button.state(["!disabled"])

        if image_path != self.image_path.get():
            return  # a different image was loaded meanwhile; discard

        if error is not None:
            from tkinter import messagebox
            messagebox.showerror("Fehler", str(error))
            self.status.set("Fehlgeschlagen -- siehe Fehlermeldung.")
            return

        self.raw_detections = raw
        self._apply_merge_and_display()

    # ---- merge slider + display -------------------------------------

    def _on_toggle_display_values(self):
        visible = self._get_visible_results()
        if self.base_overlay is not None:
            self._show_image(detections=visible or None)
        self._render_result_cards(visible)

    def _apply_merge_and_display(self):
        """Re-runs merge + decision engine only (no model call, no mask
        recomputation) at the current slider value, and redraws."""
        self._slider_job = None
        if self.raw_detections is None or self.geom is None:
            return

        try:
            results, _geom = evaluate_detections(
                self.raw_detections, self.image_path.get(),
                blade_length_mm=self._blade_length_mm_used,
                gap_threshold_px=int(self.gap_threshold_px.get()),
                geom=self.geom, config=self.config)
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Fehler", str(e))
            self.status.set("Fehlgeschlagen -- siehe Fehlermeldung.")
            return

        self.results = results
        self._rebuild_class_filter()
        visible = self._get_visible_results()

        if not self.raw_detections:
            handle_no_defect(self.image_path.get())  # future annotation hook
            self.status.set("Kein Defekt erkannt -> Bewertung der Schaufel: INTAKT.")
        else:
            n_raw = len(self.raw_detections)
            self.status.set(
                f"{n_raw} Rohdetektion(en) -> {len(results)} nach Zusammenführung -> "
                f"{len(visible)} angezeigt. Bewertung der Schaufel: "
                f"{VERDICT_DE[overall_verdict(results)]}.")

        self._show_image(detections=visible)
        self._render_result_cards(visible)

    # ---- result cards -------------------------------------------------

    def _render_result_cards(self, results):
        for child in self.cards_container.winfo_children():
            child.destroy()

        if not results:
            if self.raw_detections is not None and len(self.raw_detections) == 0:
                self._make_healthy_card(self.cards_container)
            elif self.raw_detections:
                self._plain_message("Alle erkannten Klassen sind ausgeblendet.")
            else:
                self._plain_message("Detektion starten, um Ergebnisse zu sehen.")
            return

        for i, r in enumerate(results, start=1):
            self._make_card(self.cards_container, i, r)

    def _plain_message(self, text):
        tk = self._tk
        lbl = tk.Label(self.cards_container, text=text, bg=BG, fg=FG, font=FONT)
        lbl.pack(anchor="w", padx=8, pady=8)
        self._bind_wheel(lbl)

    def _make_healthy_card(self, parent):
        tk = self._tk
        card = tk.Frame(parent, bg=PANEL_BG, highlightbackground=BORDER, highlightthickness=1)
        card.pack(fill="x", padx=4, pady=4)
        tk.Label(card, text="Kein Defekt vom Modell erkannt.", bg=PANEL_BG,
                 fg=TEXT_STRONG, font=FONT).pack(anchor="w", padx=8, pady=(8, 4))
        verdict = tk.Frame(card, bg=GREEN)
        verdict.pack(fill="x", pady=(4, 0))
        tk.Label(verdict, text="INTAKT", bg=GREEN, fg="#0a0a14",
                 font=("Courier New", 11, "bold")).pack(pady=3)
        self._bind_wheel_recursive(card)

    def _make_card(self, parent, index, r):
        tk = self._tk
        is_repair = r["decision"] == "repair"
        verdict_color = GREEN if is_repair else RED
        verdict_text = "REPARATUR" if is_repair else "AUSSCHUSS"
        annotation_color = self._color_for_class(r["defect_type"])
        is_erosion = is_erosion_type(r.get("defect_type"))
        w_px, h_px = bbox_size_px(r["bbox_px"])

        card = tk.Frame(parent, bg=PANEL_BG, highlightbackground=BORDER, highlightthickness=1)
        card.pack(fill="x", padx=4, pady=4)

        top = tk.Frame(card, bg=PANEL_BG)
        top.pack(fill="x", padx=8, pady=(6, 2))
        tk.Frame(top, bg=annotation_color, width=10, height=10).pack(side="left", padx=(0, 6))
        tk.Label(top, text=f"Defekt Nr. {index}: {display_class(r.get('defect_type', '?'))}",
                 bg=PANEL_BG, fg=ACCENT, font=FONT_BOLD).pack(side="left")
        tk.Label(top, text=f"Konf. {r.get('confidence', 0):.2f}",
                 bg=PANEL_BG, fg=FG, font=FONT).pack(side="right")

        rows = tk.Frame(card, bg=PANEL_BG)
        rows.pack(fill="x", padx=8, pady=(0, 4))

        def row(label, value):
            line = tk.Frame(rows, bg=PANEL_BG)
            line.pack(fill="x")
            tk.Label(line, text=label, bg=PANEL_BG, fg=FG, font=FONT,
                     width=20, anchor="w").pack(side="left")
            tk.Label(line, text=value, bg=PANEL_BG, fg=TEXT_STRONG, font=FONT,
                     anchor="w", justify="left", wraplength=215).pack(side="left")

        row("Zone:", ZONE_DE.get(r.get("zone"), "-"))
        row("Stufe:", r.get("stage", "-"))
        if r.get("width_mm") is not None:
            row("Breite:", f"{r['width_mm']:.2f} mm  (Grenzwert {r['width_limit_mm']:.2f} mm)")
        if r.get("area_fraction") is not None:
            row("Flächenanteil:", f"{r['area_fraction']*100:.2f} % der Schaufel")

        if is_erosion:
            if self.show_size_values.get():
                row("BB-Breite (px):", f"{w_px:.1f}")
                row("BB-Länge (px):", f"{h_px:.1f}")
        else:
            if self.show_position_values.get():
                tip_px = r.get("tip_point_px")
                offset_px = r.get("bbox_offset_from_tip_px")
                if tip_px is not None:
                    row("Spitzenpunkt (px):", f"({tip_px[0]}, {tip_px[1]})")
                if offset_px is not None:
                    row("Abstand (px):",
                        f"dx={abs(offset_px[0]):.1f}, dy={abs(offset_px[1]):.1f}")
            if self.show_size_values.get():
                row("BB-Größe (px):", f"{w_px:.1f} x {h_px:.1f}")

        row("Grund:", REASON_DE.get(r.get("reason"), r.get("reason", "-")))
        if r.get("merged_count", 1) > 1:
            row("Zusammengeführt:", f"{r['merged_count']} Rohdetektionen")

        verdict = tk.Frame(card, bg=verdict_color)
        verdict.pack(fill="x", pady=(4, 0))
        tk.Label(verdict, text=verdict_text, bg=verdict_color, fg="#0a0a14",
                 font=("Courier New", 11, "bold")).pack(pady=3)

        self._bind_wheel_recursive(card)


# ═════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import tkinter as tk
    root = tk.Tk()
    app = InspectionApp(root)
    root.mainloop()
