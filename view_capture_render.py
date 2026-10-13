"""
view_capture_render.py  (render-only, multi-class labels, pause/resume + ETA)
==================================================================================
Renders each .obj from two fixed orthographic views (Y- / Y+ side, X-Z
plane) on a WHITE background, projects the defect bounding box (from an
nTop "BB_Cordination" file) into image space, writes YOLO-format labels.

RUN MANIFEST (render_job_<timestamp>.json, written to out_dir at job
start): freezes everything needed to reproduce a job exactly — the
resolved obj->BB->class_id mapping plus every render parameter
(lengths, resolution, fill band, fit mode, base_len). Status field
(running/paused/done) is updated as the job progresses.

RESUME: `--resume <manifest.json>` (CLI) or "Resume job..." (GUI)
reloads the manifest and re-renders using its frozen params. The
per-image skip check (both .png and .txt already on disk -> skip) does
the actual work of not re-rendering finished images. If the
class_map.json has changed since the job started, resume aborts loud.

PAUSE: takes effect between images, not mid-render. Implemented as a
threading.Event checked after each completed image; Cancel is a second
Event that exits the loop at the same checkpoint and marks the manifest
"paused" so `--resume` picks it up later.

TIME-LEFT METER: rolling average over the last ~20 completed images x
images remaining. GUI mode renders on a background thread (pyrender's
OffscreenRenderer is thread-bound) while the main thread polls shared
progress state to update a progress bar + time-left label.

CLASS MAP: class_map.json is the single source of truth for class ids;
this script is the only writer, split_build_gui.py only reads it.
"""

import os, sys, math, re, json, argparse, hashlib, datetime, time, threading, collections
import numpy as np
import trimesh, pyrender
from PIL import Image, ImageDraw, ImageFont

try:
    from tqdm import tqdm as _tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ───────────────────────────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────────────────────────
SS                = 2
FRAME_MARGIN      = 1.15
BOX_COLOR         = (220, 30, 30)
BOX_WIDTH         = 2
PNG_SAVE_KWARGS   = {"format": "PNG", "compress_level": 0, "optimize": False}
BASE_LENGTH_MM    = 82.0

# ── LIGHTING OPTIONS ──────────────────────────────────────────────
LIGHTING_OPTIONS = {
    "A_baseline_single_light": {
        "lights": [
            {"type": "directional", "intensity": 8.0, "az": 0, "el": 30},
        ],
        "ambient": [0.02, 0.02, 0.02],
        "roughness": None,
        "tonemap": False,
        "percentile_norm": False,
    },
    "B_three_point_dome": {
        "lights": [
            {"type": "directional", "intensity": 3.0, "az": 25,  "el": 20},
            {"type": "directional", "intensity": 1.5, "az": -80, "el": -10},
            {"type": "directional", "intensity": 1.0, "az": 170, "el": -5},
        ],
        "ambient": [0.15, 0.15, 0.15],
        "roughness": None,
        "tonemap": False,
        "percentile_norm": False,
    },
    "C_dome_plus_roughness": {
        "lights": [
            {"type": "directional", "intensity": 3.0, "az": 25,  "el": 20},
            {"type": "directional", "intensity": 1.5, "az": -80, "el": -10},
            {"type": "directional", "intensity": 1.0, "az": 170, "el": -5},
        ],
        "ambient": [0.2, 0.2, 0.2],
        "roughness": 0.8,
        "tonemap": False,
        "percentile_norm": False,
    },
    "D_dome_roughness_tonemap": {
        "lights": [
            {"type": "directional", "intensity": 3.0, "az": 25,  "el": 20},
            {"type": "directional", "intensity": 1.5, "az": -80, "el": -10},
            {"type": "directional", "intensity": 1.0, "az": 170, "el": -5},
        ],
        "ambient": [0.25, 0.25, 0.25],
        "roughness": 0.85,
        "tonemap": True,
        "percentile_norm": False,
    },
    "E_dome_roughness_tonemap_norm": {
        "lights": [
            {"type": "directional", "intensity": 3.0, "az": 25,  "el": 20},
            {"type": "directional", "intensity": 1.5, "az": -80, "el": -10},
            {"type": "directional", "intensity": 1.0, "az": 170, "el": -5},
        ],
        "ambient": [0.25, 0.25, 0.25],
        "roughness": 0.85,
        "tonemap": True,
        "percentile_norm": True,
        "norm_percentile": 95,
        "norm_target": 235,
    },
    "G_high_key_diffuse": {
        "lights": [
            {"type": "directional", "intensity": 3.0, "az":  45, "el": 45},
            {"type": "directional", "intensity": 3.0, "az": -45, "el": 45},
            {"type": "directional", "intensity": 2.0, "az":   0, "el": 60},
            {"type": "directional", "intensity": 2.0, "az": 180, "el": 20},
        ],
        "ambient": [0.15, 0.15, 0.15],
        "roughness": None,
        "tonemap": False,
        "percentile_norm": False,
    },
    "I_four_point_overcast": {
        "lights": [
            {"type": "directional", "intensity": 2.5, "az":   0, "el": 70},
            {"type": "directional", "intensity": 2.5, "az":  90, "el": 70},
            {"type": "directional", "intensity": 2.5, "az": 180, "el": 70},
            {"type": "directional", "intensity": 2.5, "az": 270, "el": 70},
        ],
        "ambient": [0.10, 0.10, 0.10],
        "roughness": None,
        "tonemap": False,
        "percentile_norm": False,
    },
}

ACTIVE_LIGHTING_OPTIONS = [
    "A_baseline_single_light",
    "B_three_point_dome",
    "G_high_key_diffuse",
    "I_four_point_overcast",
]

_LIGHT_OFFSETS    = {}
_BASE_INTENSITIES = {}
_AMBIENT          = None
_ROUGHNESS        = None
_TONEMAP          = None
_PERCENTILE_NORM  = None
_NORM_PERCENTILE  = None
_NORM_TARGET      = None


def set_active_lighting(option_name: str) -> None:
    global _LIGHT_OFFSETS, _BASE_INTENSITIES, _AMBIENT, _ROUGHNESS
    global _TONEMAP, _PERCENTILE_NORM, _NORM_PERCENTILE, _NORM_TARGET
    opt = LIGHTING_OPTIONS[option_name]
    _LIGHT_OFFSETS    = {f"light{i}": (l["az"], l["el"])
                         for i, l in enumerate(opt["lights"])}
    _BASE_INTENSITIES = {f"light{i}": l["intensity"]
                         for i, l in enumerate(opt["lights"])}
    _AMBIENT          = opt["ambient"]
    _ROUGHNESS        = opt["roughness"] if opt["roughness"] is not None else 0.45
    _TONEMAP          = opt["tonemap"]
    _PERCENTILE_NORM  = opt["percentile_norm"]
    _NORM_PERCENTILE  = opt.get("norm_percentile", 95)
    _NORM_TARGET      = opt.get("norm_target", 235)


set_active_lighting(ACTIVE_LIGHTING_OPTIONS[0])

# ── LOG SPACING ──────────────────────────────────────────────────
def log_spaced_lengths(L_min: float, L_max: float, N: int) -> list[float]:
    return [round(L_min * (L_max / L_min) ** (i / (N - 1)), 1)
            for i in range(N)]

DEFAULT_LENGTHS = log_spaced_lengths(82.0, 2500.0, 10)

FIT_MODE  = "band"
FILL_MIN  = 0.45
FILL_MAX  = 0.92
FIT_BLEND = 0.5

VIEWS = [
    ("Yneg",   0.0, 0.0),
    ("Ypos", 180.0, 0.0),
]


# ───────────────────────────────────────────────────────────────────
# ANNOTATION CONTROL: 10 % sample, deterministic
# ───────────────────────────────────────────────────────────────────
def should_annotate(base_name: str) -> bool:
    return int(hashlib.md5(base_name.encode()).hexdigest(), 16) % 10 == 0


# ───────────────────────────────────────────────────────────────────
# SMALL HELPERS
# ───────────────────────────────────────────────────────────────────
def fmt_len(L: float) -> str:
    return ("%g" % L).replace(".", "p") + "mm"


def parse_lengths(text: str) -> list[float]:
    vals = []
    for tok in re.split(r"[,\s]+", text.strip()):
        if not tok:
            continue
        try:
            v = float(tok)
            if v > 0:
                vals.append(v)
        except ValueError:
            pass
    return vals


def as_fraction(v: float) -> float:
    f = v / 100.0 if v > 1.0 else v
    return float(min(1.0, max(1e-3, f)))


def estimate_size_mb(n_models: int, lengths: list[float],
                     rw: int, rh: int) -> float:
    n_images = n_models * len(lengths) * len(VIEWS) * len(ACTIVE_LIGHTING_OPTIONS)
    bytes_each = rw * rh * 3
    return (n_images * bytes_each) / 1e6


def parse_type(stem: str) -> str:
    """Extract defect type from `<Type>_ID_<n>` stem. Falls back to
    the full stem if the pattern doesn't match, so unrecognised
    filenames still show up in reports instead of being merged away."""
    m = re.match(r"^([A-Za-z]+)_ID_\d+", stem)
    return m.group(1) if m else stem


# ───────────────────────────────────────────────────────────────────
# CLASS MAP  (single source of truth, shared with split_build_gui.py —
# this script is the ONLY writer; split_build_gui.py only reads it)
# ───────────────────────────────────────────────────────────────────
def load_class_map(path: str) -> dict:
    """{} if the file doesn't exist yet (first-ever run)."""
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {str(k): int(v) for k, v in raw.items()}


def save_class_map(path: str, class_map: dict) -> None:
    """Atomic write (tmp + rename) so a crash mid-write can't corrupt
    the map that other runs/tools depend on."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(class_map, f, indent=2, sort_keys=False)
    os.replace(tmp, path)


def register_missing_types(class_map: dict, missing_types: list[str],
                           path: str, log=print) -> dict:
    """Appends missing types with the next free integer id(s), in
    sorted order for determinism. Saves immediately. Never reassigns
    an existing type's id."""
    next_id = (max(class_map.values()) + 1) if class_map else 0
    for t in sorted(missing_types):
        class_map[t] = next_id
        log(f"[class-map] registered new type '{t}' -> class id {next_id}")
        next_id += 1
    save_class_map(path, class_map)
    return class_map


def ensure_class_map(class_map_path: str, types_needed: list[str],
                     register_new: bool, log=print) -> dict:
    """Loads the map, checks every type needed by this run is present.
    Raises RuntimeError (loud, explicit) if types are missing and
    register_new is False. Returns the (possibly updated) map."""
    class_map = load_class_map(class_map_path)
    missing = sorted(t for t in types_needed if t not in class_map)
    if not missing:
        return class_map
    if not register_new:
        raise RuntimeError(
            f"Class map is missing {len(missing)} type(s): {', '.join(missing)}\n"
            f"        Class map file: {class_map_path}\n"
            f"        Existing entries: {class_map if class_map else '(none — file does not exist yet)'}\n"
            f"        Add the missing type(s) to the JSON manually, or re-run with\n"
            f"        --register-new-classes (CLI) / \"Auto-register new types\" (GUI)\n"
            f"        to append them automatically with the next free id(s).")
    return register_missing_types(class_map, missing, class_map_path, log)


# ───────────────────────────────────────────────────────────────────
# PAUSE / CANCEL SIGNAL
# ───────────────────────────────────────────────────────────────────
class _JobStopped(Exception):
    """Raised inside the render loop when a Cancel was requested, at
    the same between-images checkpoint used for Pause. Caught one
    level up to mark the manifest 'paused' and exit cleanly."""
    pass


# ───────────────────────────────────────────────────────────────────
# TIME-LEFT METER  (tqdm-compatible .update(n) interface — can be
# passed anywhere a tqdm pbar is currently accepted, no other changes
# needed at the call sites)
# ───────────────────────────────────────────────────────────────────
class ProgressMeter:
    """Rolling average over the last `window` completed images x images
    remaining. Plain int/float attributes read by a polling GUI thread;
    under the GIL this is safe enough for a ~300ms poll (worst case one
    stale read, self-correcting next poll) without needing a lock for
    what is only ever a progress display."""

    def __init__(self, total: int = 0, window: int = 20):
        self.total = total
        self.done = 0
        self._times = collections.deque(maxlen=window)
        self._last = time.monotonic()
        self.eta_seconds = None

    def update(self, n: int = 1) -> None:
        now = time.monotonic()
        dt_per_image = (now - self._last) / max(1, n)
        self._last = now
        for _ in range(n):
            self._times.append(dt_per_image)
        self.done += n
        if self._times and self.total:
            avg = sum(self._times) / len(self._times)
            remaining = max(0, self.total - self.done)
            self.eta_seconds = avg * remaining
        else:
            self.eta_seconds = None

    def close(self) -> None:
        pass  # tqdm-interface compatibility no-op


def format_hms(seconds: float | None) -> str:
    if seconds is None:
        return "--:--:--"
    s = int(max(0, seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ───────────────────────────────────────────────────────────────────
# RUN MANIFEST  (render_job_<run_id>.json — enables --resume)
# ───────────────────────────────────────────────────────────────────
def new_run_id() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def manifest_path_for(out_dir: str, run_id: str) -> str:
    return os.path.join(out_dir, f"render_job_{run_id}.json")


def write_manifest(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def load_manifest(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def update_manifest(path: str, **fields) -> None:
    data = load_manifest(path)
    data.update(fields)
    data["updated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    write_manifest(path, data)


def list_resumable_manifests(out_dir: str) -> list[tuple[str, dict]]:
    """Manifests in out_dir whose status isn't 'done' yet."""
    out = []
    if not out_dir or not os.path.isdir(out_dir):
        return out
    for fn in sorted(os.listdir(out_dir)):
        if fn.startswith("render_job_") and fn.endswith(".json"):
            p = os.path.join(out_dir, fn)
            try:
                d = load_manifest(p)
            except Exception:
                continue
            if d.get("status") != "done":
                out.append((p, d))
    return out


# ───────────────────────────────────────────────────────────────────
# BB FILE COLLECTION  (recursive) + MATCHING (no silent fallback)
# ───────────────────────────────────────────────────────────────────
def collect_bb_files_recursive(folder: str) -> list[str]:
    """os.walk the folder for .txt BB files. Fixes the same bug already
    found in split_build_gui.py: BB files one level deeper than expected
    were silently missed by flat/top-level scans."""
    found = []
    for root, _dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".txt"):
                found.append(os.path.join(root, f))
    return found


def build_bb_stem_map(bb_paths: list[str]) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Map stem(lower) -> path. Reports duplicate stems (same stem found
    in multiple files) instead of silently picking one."""
    stem_map: dict[str, str] = {}
    duplicates: dict[str, list[str]] = {}
    for p in bb_paths:
        stem = os.path.splitext(os.path.basename(p))[0].lower()
        if stem in stem_map and stem_map[stem] != p:
            duplicates.setdefault(stem, [stem_map[stem]]).append(p)
        else:
            stem_map[stem] = p
    return stem_map, duplicates


def resolve_bb_for(obj_path: str, bb_stem_map: dict[str, str]) -> str | None:
    """Match by stem, or sibling .txt next to the .obj. NO fallback to
    an arbitrary other BB file — that was the bug. Unmatched -> None."""
    stem = os.path.splitext(os.path.basename(obj_path))[0].lower()
    if stem in bb_stem_map:
        return bb_stem_map[stem]
    sib = os.path.splitext(obj_path)[0] + ".txt"
    if os.path.isfile(sib):
        return sib
    return None


def preflight_match_report(obj_paths: list[str], bb_stem_map: dict[str, str],
                           duplicates: dict[str, list[str]],
                           out_dir: str) -> tuple[list[str], list[str]]:
    """Runs BEFORE any rendering. Returns (matched_obj_paths, unmatched_obj_paths).
    Writes a full per-type breakdown to disk and stdout so undercounts are
    caught immediately instead of discovered after the fact."""
    matched, unmatched = [], []
    by_type_total: dict[str, int] = {}
    by_type_matched: dict[str, int] = {}

    for p in obj_paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        t = parse_type(stem)
        by_type_total[t] = by_type_total.get(t, 0) + 1
        if resolve_bb_for(p, bb_stem_map):
            matched.append(p)
            by_type_matched[t] = by_type_matched.get(t, 0) + 1
        else:
            unmatched.append(p)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(out_dir, f"BB_match_report_{ts}.txt")
    lines = []
    lines.append(f"BB match report  {ts}")
    lines.append(f"Total .obj: {len(obj_paths)}   Matched: {len(matched)}   Unmatched: {len(unmatched)}")
    lines.append("")
    lines.append("Per type:")
    for t in sorted(by_type_total):
        tot = by_type_total[t]
        mat = by_type_matched.get(t, 0)
        lines.append(f"  {t:12s} matched {mat:4d} / {tot:4d}")
    if duplicates:
        lines.append("")
        lines.append(f"WARNING: {len(duplicates)} stem(s) matched by more than one BB file "
                     f"(first occurrence used, later ones ignored):")
        for stem, paths in duplicates.items():
            lines.append(f"  {stem}: " + " | ".join(paths))
    if unmatched:
        lines.append("")
        lines.append(f"Unmatched .obj stems ({len(unmatched)}) — these will NOT be rendered:")
        for p in unmatched:
            lines.append(f"  {os.path.splitext(os.path.basename(p))[0]}")
    report_text = "\n".join(lines)

    os.makedirs(out_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print("=" * 60)
    print(report_text)
    print("=" * 60)
    print(f"[report] written to {report_path}")

    return matched, unmatched


# ───────────────────────────────────────────────────────────────────
# BB FILE PARSING
# ───────────────────────────────────────────────────────────────────
def parse_bb_file(path: str) -> dict:
    text = open(path, "r", encoding="utf-8", errors="ignore").read()

    def grab(key: str) -> float:
        m = re.search(
            rf"{re.escape(key)}\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", text)
        if not m:
            raise ValueError(f"Could not find '{key}' in {path}")
        return float(m.group(1))

    bb = {
        "x_min": grab("BB_Min.X"), "x_max": grab("BB_Max.X"),
        "z_min": grab("BB_Min.Z"), "z_max": grab("BB_Max.Z"),
    }
    if bb["x_min"] > bb["x_max"]:
        bb["x_min"], bb["x_max"] = bb["x_max"], bb["x_min"]
    if bb["z_min"] > bb["z_max"]:
        bb["z_min"], bb["z_max"] = bb["z_max"], bb["z_min"]
    return bb


# ───────────────────────────────────────────────────────────────────
# CAMERA MATHS
# ───────────────────────────────────────────────────────────────────
def spherical_to_pose(az_deg: float, el_deg: float, dist: float) -> np.ndarray:
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    x  = dist * math.cos(el) * math.sin(az)
    y  = dist * math.sin(el)
    z  = dist * math.cos(el) * math.cos(az)
    eye = np.array([x, y, z])
    fwd = -eye / np.linalg.norm(eye)
    up0 = np.array([0., 1., 0.])
    if abs(np.dot(fwd, up0)) > 0.99:
        up0 = np.array([0., 0., 1.])
    right = np.cross(fwd, up0);  right /= np.linalg.norm(right)
    up    = np.cross(right, fwd); up   /= np.linalg.norm(up)
    pose = np.eye(4)
    pose[:3, 0] = right; pose[:3, 1] = up
    pose[:3, 2] = -fwd;  pose[:3, 3] = eye
    return pose


def project_to_pixels(points_world: np.ndarray, cam_pose: np.ndarray,
                       proj: np.ndarray, width: int, height: int) -> np.ndarray:
    pts_h = np.hstack([points_world, np.ones((len(points_world), 1))])
    view  = np.linalg.inv(cam_pose)
    cam   = (view @ pts_h.T).T
    clip  = (proj @ cam.T).T
    w     = clip[:, 3:4].copy()
    w[np.abs(w) < 1e-9] = 1e-9
    ndc   = clip[:, :3] / w
    px = (ndc[:, 0] * 0.5 + 0.5) * width
    py = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * height
    return np.stack([px, py], axis=1)


# ───────────────────────────────────────────────────────────────────
# MESH + DEFECT BOX
# ───────────────────────────────────────────────────────────────────
def load_normalised_mesh(obj_path: str):
    mesh = trimesh.load(obj_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    orig_bounds = mesh.bounds.copy()
    center      = mesh.bounding_box.bounds.mean(axis=0)
    T = trimesh.transformations.translation_matrix(-center)
    R = trimesh.transformations.rotation_matrix(math.pi / 2, [-1, 0, 0])
    M = R @ T
    mesh.apply_transform(M)
    return mesh, M, orig_bounds


def defect_box_corners(bb: dict, y_min: float, y_max: float,
                        M: np.ndarray) -> np.ndarray:
    xs = [bb["x_min"], bb["x_max"]]
    ys = [y_min, y_max]
    zs = [bb["z_min"], bb["z_max"]]
    corners   = np.array([[x, y, z] for x in xs for y in ys for z in zs])
    corners_h = np.hstack([corners, np.ones((8, 1))])
    return (M @ corners_h.T).T[:, :3]


# ───────────────────────────────────────────────────────────────────
# FRAMING
# ───────────────────────────────────────────────────────────────────
def view_for_fraction(ex_L: float, ey_L: float, frac: float,
                      aspect: float) -> tuple[float, float]:
    ymag = max(ey_L / (2.0 * frac), (ex_L / (2.0 * frac)) / aspect)
    return ymag, ymag * aspect


# ───────────────────────────────────────────────────────────────────
# SCENE / LIGHTING
# ───────────────────────────────────────────────────────────────────
def _light_pose(az_deg, el_deg):
    return spherical_to_pose(az_deg, el_deg, 1.0)


def build_scene(mesh: trimesh.Trimesh) -> pyrender.Scene:
    scene = pyrender.Scene(ambient_light=_AMBIENT,
                           bg_color=[1.0, 1.0, 1.0, 1.0])
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.62, 0.64, 0.68, 1.0],
        metallicFactor=0.55, roughnessFactor=_ROUGHNESS, doubleSided=True)
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True, material=material))
    for name, (az, el) in _LIGHT_OFFSETS.items():
        scene.add(pyrender.DirectionalLight(
                      color=[1.0, 1.0, 1.0],
                      intensity=_BASE_INTENSITIES[name]),
                  pose=_light_pose(az, el), name=name)
    return scene


def update_lights(scene, cam_az, cam_el):
    for node in scene.get_nodes():
        if node.name not in _LIGHT_OFFSETS:
            continue
        daz, del_ = _LIGHT_OFFSETS[node.name]
        scene.set_pose(node, _light_pose(
            cam_az + daz, float(np.clip(cam_el + del_, -89, 89))))
        node.light.intensity = _BASE_INTENSITIES[node.name]


# ───────────────────────────────────────────────────────────────────
# POST-PROCESSING
# ───────────────────────────────────────────────────────────────────
def apply_reinhard_tonemap(img_float: np.ndarray) -> np.ndarray:
    return img_float / (img_float + 1.0)


def apply_percentile_norm(img_float: np.ndarray, percentile: float = 95,
                           target_255: float = 235) -> np.ndarray:
    lum = (0.2126 * img_float[..., 0] + 0.7152 * img_float[..., 1]
           + 0.0722 * img_float[..., 2])
    p_val = np.percentile(lum, percentile)
    if p_val < 1e-6:
        return img_float
    scale = (target_255 / 255.0) / p_val
    return np.clip(img_float * scale, 0.0, 1.0)


def postprocess_render(color_arr: np.ndarray) -> np.ndarray:
    if not (_TONEMAP or _PERCENTILE_NORM):
        return color_arr
    img_float = color_arr.astype(np.float64) / 255.0
    if _TONEMAP:
        img_float = apply_reinhard_tonemap(img_float * 4.0)
    if _PERCENTILE_NORM:
        img_float = apply_percentile_norm(
            img_float, percentile=_NORM_PERCENTILE, target_255=_NORM_TARGET)
    return np.clip(img_float * 255.0, 0, 255).astype(np.uint8)


# ───────────────────────────────────────────────────────────────────
# RENDER ONE MODEL
# ───────────────────────────────────────────────────────────────────
def render_model(obj_path: str, bb: dict, out_dir: str, rw: int, rh: int,
                 lengths: list[float], base_len: float,
                 fit_mode: str, fill_min: float, fill_max: float,
                 fit_blend: float, class_id: int, type_name: str,
                 pbar=None, annot_dir: str = "",
                 resume_event: threading.Event | None = None,
                 stop_event: threading.Event | None = None):
    stem = os.path.splitext(os.path.basename(obj_path))[0]
    base_mesh, M_base, orig_bounds = load_normalised_mesh(obj_path)
    records = []  # (img_path, label_path)

    y_min, y_max = float(orig_bounds[0, 1]), float(orig_bounds[1, 1])
    bex, bey, bez = base_mesh.extents
    aspect = rw / rh

    scales = [(L, L / base_len) for L in lengths]
    L_min, L_max = min(lengths), max(lengths)
    span = (L_max - L_min) or 1.0

    if fit_mode == "shared":
        min_scale = L_min / base_len
        max_scale = L_max / base_len
        ref_scale = min_scale + fit_blend * (max_scale - min_scale)
        rex, rey, rez = bex * ref_scale, bey * ref_scale, bez * ref_scale
        shared_ymag = max(rey / 2.0, (rex / 2.0) / aspect) * FRAME_MARGIN
        shared_xmag = shared_ymag * aspect
        big = max(bex, bey, bez) * max_scale
        shared_depth = float(big) * 2.0

    rw_ss, rh_ss = rw * SS, rh * SS
    renderer = pyrender.OffscreenRenderer(rw_ss, rh_ss)

    for L, scale in scales:
        Sx   = trimesh.transformations.scale_matrix(scale)
        M    = Sx @ M_base
        mesh = base_mesh.copy()
        mesh.apply_transform(Sx)

        ex_L, ey_L, ez_L = bex * scale, bey * scale, bez * scale

        if fit_mode == "shared":
            xmag, ymag, depth = shared_xmag, shared_ymag, shared_depth
        else:
            frac = fill_max if L_max == L_min else \
                   fill_min + (fill_max - fill_min) * (L - L_min) / span
            ymag, xmag = view_for_fraction(ex_L, ey_L, frac, aspect)
            depth = float(max(ex_L, ey_L, ez_L)) * 2.0

        box_corners = defect_box_corners(bb, y_min, y_max, M)
        ltag        = fmt_len(L)

        for lp_name in ACTIVE_LIGHTING_OPTIONS:
            set_active_lighting(lp_name)
            scene = build_scene(mesh)

            for vname, az, el in VIEWS:
                base_name  = f"{stem}__{vname}__{ltag}__{lp_name}"
                clean_path = os.path.join(out_dir, f"{base_name}.png")
                label_path = os.path.join(out_dir, f"{base_name}.txt")

                if os.path.isfile(clean_path) and os.path.isfile(label_path):
                    records.append((clean_path, label_path))
                    if pbar is not None:
                        pbar.update(1)
                    if stop_event is not None and stop_event.is_set():
                        raise _JobStopped()
                    if resume_event is not None:
                        resume_event.wait()
                    continue

                cam  = pyrender.OrthographicCamera(
                    xmag=xmag, ymag=ymag,
                    znear=max(0.01, depth * 1e-4), zfar=depth * 4.0)
                pose = spherical_to_pose(az, el, depth)
                update_lights(scene, az, el)
                cam_node = scene.add(cam, pose=pose)
                color_ss, _ = renderer.render(scene)
                scene.remove_node(cam_node)

                color_ss  = postprocess_render(color_ss)
                color_img = Image.fromarray(color_ss).convert("RGB").resize(
                    (rw, rh), Image.Resampling.LANCZOS)

                proj = cam.get_projection_matrix(rw, rh)
                px   = project_to_pixels(box_corners, pose, proj, rw, rh)
                x0 = max(0.0, float(px[:, 0].min()))
                x1 = min(rw,  float(px[:, 0].max()))
                y0 = max(0.0, float(px[:, 1].min()))
                y1 = min(rh,  float(px[:, 1].max()))

                color_img.save(clean_path, **PNG_SAVE_KWARGS)
                records.append((clean_path, label_path))

                if x1 > x0 and y1 > y0:
                    cx = ((x0 + x1) / 2.0) / rw
                    cy = ((y0 + y1) / 2.0) / rh
                    bw = (x1 - x0) / rw
                    bh = (y1 - y0) / rh
                    with open(label_path, "w") as f:
                        f.write(f"{class_id} {cx:.6f} {cy:.6f}"
                                f" {bw:.6f} {bh:.6f}\n")

                    if annot_dir and should_annotate(base_name):
                        annot = color_img.copy()
                        d = ImageDraw.Draw(annot)
                        d.rectangle([x0, y0, x1, y1],
                                    outline=BOX_COLOR, width=BOX_WIDTH)
                        d.text((x0, max(0, y0 - 12)), f"{type_name} ({class_id})",
                               fill=BOX_COLOR)
                        annot.save(
                            os.path.join(annot_dir, f"{base_name}__annotated.png"),
                            **PNG_SAVE_KWARGS)
                else:
                    open(label_path, "w").close()

                if pbar is not None:
                    pbar.update(1)
                if stop_event is not None and stop_event.is_set():
                    renderer.delete()
                    raise _JobStopped()
                if resume_event is not None:
                    resume_event.wait()

    renderer.delete()
    return records


# ───────────────────────────────────────────────────────────────────
# SHARED RENDER LOOP  (used by both a fresh run_batch and resume_run,
# so resuming never re-derives anything — it replays the exact frozen
# job list from the manifest)
# ───────────────────────────────────────────────────────────────────
def _render_jobs(jobs: list[dict], out_dir: str, rw: int, rh: int,
                 lengths: list[float], base_len: float,
                 fit_mode: str, fill_min: float, fill_max: float,
                 fit_blend: float, annot_dir: str, manifest_path: str,
                 pbar=None,
                 resume_event: threading.Event | None = None,
                 stop_event: threading.Event | None = None) -> list:
    all_records = []
    stopped = False
    for j in jobs:
        obj_path, bb_path = j["obj"], j["bb"]
        class_id, type_name = j["class_id"], j["type"]
        if not os.path.isfile(obj_path):
            print(f"[skip] obj no longer on disk: {obj_path}")
            continue
        if not os.path.isfile(bb_path):
            print(f"[skip] bb no longer on disk: {bb_path}")
            continue
        try:
            bb = parse_bb_file(bb_path)
        except ValueError as e:
            print(f"[skip] {e}")
            continue

        if pbar is None:
            print(f"[model] {os.path.basename(obj_path)}"
                  f"  ←  {os.path.basename(bb_path)}"
                  f"  [{type_name}={class_id}]")

        try:
            all_records.extend(
                render_model(obj_path, bb, out_dir, rw, rh,
                            lengths, base_len,
                            fit_mode, fill_min, fill_max, fit_blend,
                            class_id, type_name,
                            pbar=pbar, annot_dir=annot_dir,
                            resume_event=resume_event, stop_event=stop_event))
        except _JobStopped:
            stopped = True
            break

    if pbar is not None:
        pbar.close()

    if stopped:
        update_manifest(manifest_path, status="paused", done_images=len(all_records))
        print(f"\n[paused] {len(all_records)} image/label pairs written so far.")
        print(f"[paused] resume with: --resume \"{manifest_path}\"")
    else:
        update_manifest(manifest_path, status="done", done_images=len(all_records))
        print(f"\n[done] {len(all_records)} image/label pairs → {out_dir}")
        print(f"[done] Annotation_control → {annot_dir}")
    return all_records


# ───────────────────────────────────────────────────────────────────
# BATCH RUN  (render only — no split, no zip)
# ───────────────────────────────────────────────────────────────────
def run_batch(obj_paths: list[str],
              bb_paths: list[str],
              out_dir: str,
              rw: int, rh: int,
              lengths: list[float],
              base_len: float,
              fit_mode: str = FIT_MODE,
              fill_min: float = FILL_MIN,
              fill_max: float = FILL_MAX,
              fit_blend: float = FIT_BLEND,
              class_map_path: str | None = None,
              register_new_classes: bool = False,
              pbar_override=None,
              resume_event: threading.Event | None = None,
              stop_event: threading.Event | None = None) -> list:

    lengths     = lengths or DEFAULT_LENGTHS
    fit_mode    = fit_mode if fit_mode in ("band", "shared") else "band"
    os.makedirs(out_dir, exist_ok=True)

    annot_dir = os.path.join(out_dir, "Annotation_control")
    os.makedirs(annot_dir, exist_ok=True)

    bb_stem_map, duplicates = build_bb_stem_map(bb_paths)

    # PRE-FLIGHT: full match report before any rendering happens
    matched_objs, unmatched_objs = preflight_match_report(
        obj_paths, bb_stem_map, duplicates, out_dir)

    if not matched_objs:
        print("[abort] zero matched obj/BB pairs — nothing to render.")
        return []

    # ── CLASS MAP PRE-FLIGHT (before any rendering) ──────────────
    class_map_path = class_map_path or os.path.join(out_dir, "class_map.json")
    types_needed = sorted({parse_type(os.path.splitext(os.path.basename(p))[0])
                           for p in matched_objs})
    try:
        class_map = ensure_class_map(class_map_path, types_needed,
                                     register_new_classes, log=print)
    except RuntimeError as e:
        print(f"[abort] {e}")
        return []
    print(f"[class-map] {class_map_path}")
    print(f"[class-map] types this run: " +
          ", ".join(f"{t}={class_map[t]}" for t in types_needed))

    scale_summary = ", ".join(
        f"{L:g}mm ({L / base_len:.2f}\u00d7)" for L in lengths)
    print(f"[lighting] presets={ACTIVE_LIGHTING_OPTIONS}")
    print(f"[scales]   base={base_len:g}mm  →  {scale_summary}")
    if fit_mode == "band":
        print(f"[framing]  band  fill {fill_min:.0%} (smallest) → "
              f"{fill_max:.0%} (largest)")
    else:
        print(f"[framing]  shared  blend={fit_blend:g}")

    total_images = (len(matched_objs)
                    * len(lengths)
                    * len(VIEWS)
                    * len(ACTIVE_LIGHTING_OPTIONS))
    print(f"[total]    {total_images} images to render "
          f"(~{estimate_size_mb(len(matched_objs), lengths, rw, rh):.0f} MB "
          f"uncompressed upper bound)")

    # ── Freeze the exact job list (obj -> bb -> class_id) once, so a
    #    later resume replays it verbatim instead of re-deriving it. ──
    jobs = []
    for obj_path in matched_objs:
        bb_path = resolve_bb_for(obj_path, bb_stem_map)
        stem = os.path.splitext(os.path.basename(obj_path))[0]
        type_name = parse_type(stem)
        jobs.append({"obj": obj_path, "bb": bb_path,
                     "type": type_name, "class_id": class_map[type_name]})

    run_id = new_run_id()
    manifest_path = manifest_path_for(out_dir, run_id)
    write_manifest(manifest_path, {
        "run_id": run_id,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "status": "running",
        "out_dir": out_dir, "rw": rw, "rh": rh, "lengths": lengths,
        "base_len": base_len, "fit_mode": fit_mode,
        "fill_min": fill_min, "fill_max": fill_max, "fit_blend": fit_blend,
        "class_map_path": class_map_path,
        "class_map_snapshot": {t: class_map[t] for t in types_needed},
        "jobs": jobs, "total_images": total_images, "done_images": 0,
    })
    print(f"[manifest] {manifest_path}")

    pbar = pbar_override if pbar_override is not None else (
        _tqdm(total=total_images, unit="img",
             bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                         "[{elapsed}<{remaining}]")
        if HAS_TQDM else None)
    if pbar is not None:
        pbar.total = total_images

    if unmatched_objs:
        print(f"[note] {len(unmatched_objs)} obj file(s) skipped — see BB_match_report_*.txt")

    return _render_jobs(jobs, out_dir, rw, rh, lengths, base_len,
                        fit_mode, fill_min, fill_max, fit_blend,
                        annot_dir, manifest_path, pbar=pbar,
                        resume_event=resume_event, stop_event=stop_event)


# ───────────────────────────────────────────────────────────────────
# RESUME  (replays a manifest's frozen job list verbatim)
# ───────────────────────────────────────────────────────────────────
def resume_run(manifest_path: str,
               pbar_override=None,
               resume_event: threading.Event | None = None,
               stop_event: threading.Event | None = None) -> list:
    manifest = load_manifest(manifest_path)
    if manifest.get("status") == "done":
        print(f"[info] job already done: {manifest_path}")
        return []

    out_dir = manifest["out_dir"]
    class_map_path = manifest.get("class_map_path")
    snapshot = manifest.get("class_map_snapshot", {})
    if class_map_path:
        current = load_class_map(class_map_path)
        mismatched = {t: (snap_id, current.get(t))
                     for t, snap_id in snapshot.items()
                     if current.get(t) != snap_id}
        if mismatched:
            detail = ", ".join(f"{t} (job had {a}, map now has {b})"
                               for t, (a, b) in mismatched.items())
            print(f"[abort] class_map.json changed since this job started — "
                 f"resuming would risk writing labels under different class\n"
                 f"        ids than the rest of this job's already-rendered images.\n"
                 f"        Mismatched: {detail}\n"
                 f"        Fix the class map back, or start a fresh job instead of resuming.")
            return []

    annot_dir = os.path.join(out_dir, "Annotation_control")
    os.makedirs(annot_dir, exist_ok=True)

    jobs = manifest["jobs"]
    total_images = manifest["total_images"]
    already_done = manifest.get("done_images", 0)
    update_manifest(manifest_path, status="running")
    print(f"[resume] {manifest_path}")
    print(f"[resume] {len(jobs)} model(s), {total_images} images total "
         f"({already_done} recorded done at last pause; the per-image "
         f"skip check will fast-forward through anything already on disk)")

    pbar = pbar_override if pbar_override is not None else (
        _tqdm(total=total_images, unit="img",
             bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                         "[{elapsed}<{remaining}]")
        if HAS_TQDM else None)
    if pbar is not None:
        pbar.total = total_images

    return _render_jobs(jobs, out_dir, manifest["rw"], manifest["rh"],
                        manifest["lengths"], manifest["base_len"],
                        manifest["fit_mode"], manifest["fill_min"],
                        manifest["fill_max"], manifest["fit_blend"],
                        annot_dir, manifest_path, pbar=pbar,
                        resume_event=resume_event, stop_event=stop_event)


# ───────────────────────────────────────────────────────────────────
# LAUNCHER GUI
# ───────────────────────────────────────────────────────────────────
def run_launcher() -> dict:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog

    root = tk.Tk()
    root.title("Two-View Multi-Scale Defect Renderer — Setup")
    root.configure(bg="#0a0a14")
    root.resizable(False, False)

    style = ttk.Style(root); style.theme_use("clam")
    style.configure("TLabel",     background="#0a0a14", foreground="#b0bcd0",
                    font=("Courier New", 10))
    style.configure("TButton",    background="#1a2a4a", foreground="#7eb8f7",
                    font=("Courier New", 10, "bold"), padding=4)
    style.configure("TEntry",     fieldbackground="#111122",
                    foreground="#e0e8f0", font=("Courier New", 10))
    style.configure("TCombobox",  fieldbackground="#111122",
                    foreground="#e0e8f0")
    style.map("TButton", background=[("active", "#2a3a6a")])

    obj_paths: list[str] = []
    bb_paths:  list[str] = []

    def make_file_list(title, paths, row, filetypes, height=4, allow_folder=False):
        hdr = tk.Frame(root, bg="#111128")
        hdr.grid(row=row, column=0, columnspan=3,
                 sticky="ew", padx=0, pady=(10, 0))
        tk.Label(hdr, text=f"  {title}", bg="#111128", fg="#7eb8f7",
                 font=("Courier New", 10, "bold")).pack(
                     side="left", padx=6, pady=4)
        count_lbl = tk.Label(hdr, text="0 files", bg="#111128",
                              fg="#556677", font=("Courier New", 8))
        count_lbl.pack(side="right", padx=8)

        frame = tk.Frame(root, bg="#0d0d1e",
                         highlightbackground="#2a3a5a", highlightthickness=1)
        frame.grid(row=row+1, column=0, columnspan=2,
                   sticky="ew", padx=12, pady=(2, 2))
        lb = tk.Listbox(frame, bg="#0d0d1e", fg="#8899bb",
                        font=("Courier New", 8), height=height,
                        selectmode="extended",
                        highlightthickness=0, borderwidth=0)
        lb.pack(side="left", fill="both", expand=True)
        sb = tk.Scrollbar(frame, orient="vertical", command=lb.yview,
                          bg="#111128", troughcolor="#0a0a14")
        sb.pack(side="right", fill="y")
        lb.config(yscrollcommand=sb.set)

        def _refresh():
            lb.delete(0, "end")
            for p in paths:
                lb.insert("end", p)
            count_lbl.config(text=f"{len(paths)} file(s)")
            _update_size()

        def _add_files():
            ps = filedialog.askopenfilenames(
                title=f"Select {title}", filetypes=filetypes)
            for p in ps:
                if p not in paths:
                    paths.append(p)
            _refresh()

        def _add_folder():
            d = filedialog.askdirectory(title=f"Select folder ({title}) — scanned recursively")
            if not d:
                return
            found = collect_bb_files_recursive(d)
            n_before = len(paths)
            for p in found:
                if p not in paths:
                    paths.append(p)
            _refresh()
            messagebox.showinfo(
                "Folder scan",
                f"Recursive scan found {len(found)} .txt file(s) under:\n{d}\n"
                f"Added {len(paths) - n_before} new file(s) "
                f"(list now has {len(paths)}).")

        def _remove():
            for i in reversed(lb.curselection()):
                paths.pop(i)
            _refresh()

        def _clear():
            paths.clear(); _refresh()

        btn_row = tk.Frame(root, bg="#0a0a14")
        btn_row.grid(row=row+2, column=0, columnspan=3,
                     sticky="w", padx=12, pady=(0, 4))
        ttk.Button(btn_row, text="Add files",
                   command=_add_files).pack(side="left", padx=(0, 4))
        if allow_folder:
            ttk.Button(btn_row, text="Add folder (recursive)",
                       command=_add_folder).pack(side="left", padx=(0, 4))
        ttk.Button(btn_row, text="Remove sel.",
                   command=_remove).pack(side="left", padx=(0, 4))
        ttk.Button(btn_row, text="Clear",
                   command=_clear).pack(side="left")

    make_file_list("▸  OBJ FILES  (target models)", obj_paths, 0,
                   [("OBJ files", "*.obj"), ("All", "*.*")])
    make_file_list("▸  BB COORDINATE FILES  (matched by stem to .obj)",
                   bb_paths, 3,
                   [("Text files", "*.txt"), ("All", "*.*")],
                   allow_folder=True)

    pair_var = tk.StringVar(value="")
    tk.Label(root, textvariable=pair_var, bg="#0a0a14",
             fg="#5a8a5a", font=("Courier New", 8),
             justify="left").grid(row=6, column=0, columnspan=3,
                                  sticky="w", padx=14, pady=(2, 0))

    def _preview_pairs():
        stem_map, dup = build_bb_stem_map(bb_paths)
        matched, unmatched = preflight_dry_run(obj_paths, stem_map)
        lines = [f"✓ {len(matched)} matched"]
        if unmatched:
            lines.append(f"⚠ {len(unmatched)} unmatched: "
                         + ", ".join(sorted(unmatched)[:4])
                         + ("…" if len(unmatched) > 4 else ""))
        if dup:
            lines.append(f"⚠ {len(dup)} duplicate stem(s) across BB files")
        pair_var.set("   ".join(lines) if obj_paths or bb_paths else
                     "Add files to preview pairs.")

    def preflight_dry_run(obj_paths_, stem_map):
        matched, unmatched = [], []
        for p in obj_paths_:
            stem = os.path.splitext(os.path.basename(p))[0]
            if resolve_bb_for(p, stem_map):
                matched.append(stem)
            else:
                unmatched.append(stem)
        return matched, unmatched

    ttk.Button(root, text="Preview pairs",
               command=_preview_pairs).grid(
                   row=7, column=0, sticky="w", padx=12, pady=(4, 0))

    # ── Output + settings ────────────────────────────────────────
    out_hdr = tk.Frame(root, bg="#111128")
    out_hdr.grid(row=8, column=0, columnspan=3,
                 sticky="ew", padx=0, pady=(10, 0))
    tk.Label(out_hdr, text="  ▸  OUTPUT + SETTINGS", bg="#111128",
             fg="#7eb8f7", font=("Courier New", 10, "bold")).pack(
                 side="left", padx=6, pady=4)

    settings = ttk.Frame(root)
    settings.grid(row=9, column=0, columnspan=3,
                  sticky="ew", padx=12, pady=(4, 4))

    ttk.Label(settings, text="Output folder:").grid(
        row=0, column=0, sticky="w")
    out_var = tk.StringVar()
    ttk.Entry(settings, textvariable=out_var, width=40).grid(
        row=0, column=1, columnspan=2, padx=(8, 8), sticky="w")
    ttk.Button(settings, text="Browse",
               command=lambda: out_var.set(
                   filedialog.askdirectory(title="Output folder")
                   or out_var.get())
               ).grid(row=0, column=3)

    ttk.Label(settings, text="Width px:").grid(
        row=1, column=0, sticky="w", pady=(6, 0))
    rw_var = tk.StringVar(value="800")
    ttk.Entry(settings, textvariable=rw_var, width=7).grid(
        row=1, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

    ttk.Label(settings, text="Height px:").grid(
        row=2, column=0, sticky="w", pady=(4, 0))
    rh_var = tk.StringVar(value="800")
    ttk.Entry(settings, textvariable=rh_var, width=7).grid(
        row=2, column=1, sticky="w", padx=(8, 0), pady=(4, 0))

    ttk.Label(settings, text="Lengths (mm):").grid(
        row=3, column=0, sticky="w", pady=(4, 0))
    len_var = tk.StringVar(
        value=", ".join(str(L) for L in DEFAULT_LENGTHS))
    ttk.Entry(settings, textvariable=len_var, width=40).grid(
        row=3, column=1, columnspan=2, sticky="w", padx=(8, 0), pady=(4, 0))

    size_var = tk.StringVar(value="")
    tk.Label(settings, textvariable=size_var, bg="#0a0a14",
             fg="#7eb8f7", font=("Courier New", 9),
             justify="left").grid(
                 row=4, column=0, columnspan=4, sticky="w", pady=(2, 0))

    def _update_size(*_):
        try:
            rw  = int(rw_var.get())
            rh  = int(rh_var.get())
            ls  = parse_lengths(len_var.get())
            nm  = len(obj_paths)
            if not ls or nm == 0 or rw <= 0 or rh <= 0:
                size_var.set("")
                return
            n_img = nm * len(ls) * len(VIEWS) * len(ACTIVE_LIGHTING_OPTIONS)
            mb    = estimate_size_mb(nm, ls, rw, rh)
            size_var.set(
                f"  est. {n_img} images (before BB-match filtering)  |  "
                f"~{mb/1024:.2f} GB uncompressed upper bound")
        except Exception:
            size_var.set("")

    rw_var.trace_add("write", _update_size)
    rh_var.trace_add("write", _update_size)
    len_var.trace_add("write", _update_size)

    ttk.Label(settings, text="Base length (mm):").grid(
        row=5, column=0, sticky="w", pady=(4, 0))
    base_var = tk.StringVar(value=f"{BASE_LENGTH_MM:g}")
    ttk.Entry(settings, textvariable=base_var, width=7).grid(
        row=5, column=1, sticky="w", padx=(8, 0), pady=(4, 0))

    ttk.Label(settings, text="Fit mode:").grid(
        row=6, column=0, sticky="w", pady=(8, 0))
    mode_var = tk.StringVar(value="band")
    mode_cb  = ttk.Combobox(settings, textvariable=mode_var, width=10,
                             state="readonly", values=["band", "shared"])
    mode_cb.grid(row=6, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

    band_lbl = ttk.Label(settings, text="Fill % small → large:")
    band_lbl.grid(row=7, column=0, sticky="w", pady=(4, 0))
    fmin_var = tk.StringVar(value="45")
    fmax_var = tk.StringVar(value="92")
    fmin_e   = ttk.Entry(settings, textvariable=fmin_var, width=6)
    fmax_e   = ttk.Entry(settings, textvariable=fmax_var, width=6)
    fmin_e.grid(row=7, column=1, sticky="w", padx=(8, 0), pady=(4, 0))
    fmax_e.grid(row=7, column=2, sticky="w", padx=(4, 0), pady=(4, 0))

    blend_lbl = ttk.Label(settings, text="Shared blend (0–1):")
    blend_var = tk.StringVar(value="0.5")
    blend_e   = ttk.Entry(settings, textvariable=blend_var, width=6)

    hint = tk.Label(settings, bg="#0a0a14", fg="#556677",
                    font=("Courier New", 8), justify="left",
                    text="band: each size framed to a fill % (all usable)  •  "
                         "shared: one frame, big sizes overflow")
    hint.grid(row=9, column=0, columnspan=4, sticky="w", pady=(6, 0))

    ttk.Label(settings, text="Class map (.json):").grid(
        row=10, column=0, sticky="w", pady=(8, 0))
    classmap_var = tk.StringVar(value="")
    ttk.Entry(settings, textvariable=classmap_var, width=32).grid(
        row=10, column=1, columnspan=2, sticky="w", padx=(8, 0), pady=(8, 0))
    ttk.Button(settings, text="Browse",
               command=lambda: classmap_var.set(
                   filedialog.askopenfilename(
                       title="Class map (.json) — leave blank to use "
                             "<out_dir>/class_map.json",
                       filetypes=[("JSON", "*.json"), ("All", "*.*")])
                   or classmap_var.get())
               ).grid(row=10, column=3, pady=(8, 0))

    register_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(settings, text="Auto-register new types found in filenames "
                                   "(off = abort loud on unknown type)",
                    variable=register_var).grid(
                        row=11, column=0, columnspan=4, sticky="w", pady=(4, 0))

    def _sync_mode(*_):
        if mode_var.get() == "band":
            band_lbl.grid(); fmin_e.grid(); fmax_e.grid()
            blend_lbl.grid_remove(); blend_e.grid_remove()
        else:
            band_lbl.grid_remove(); fmin_e.grid_remove(); fmax_e.grid_remove()
            blend_lbl.grid(row=7, column=0, sticky="w", pady=(4, 0))
            blend_e.grid(row=7, column=1, sticky="w", padx=(8, 0), pady=(4, 0))
    mode_cb.bind("<<ComboboxSelected>>", _sync_mode)
    _sync_mode()

    result = {"ok": False}

    def on_ok():
        if not obj_paths:
            messagebox.showerror("Error", "Select at least one .obj file.")
            return
        if not bb_paths:
            messagebox.showerror("Error", "Select at least one BB file (or a folder).")
            return
        if not out_var.get().strip():
            messagebox.showerror("Error", "Select an output folder.")
            return
        try:
            rw = int(rw_var.get()); rh = int(rh_var.get())
            assert rw > 0 and rh > 0
        except Exception:
            messagebox.showerror("Error",
                "Width and height must be positive integers.")
            return
        lengths = parse_lengths(len_var.get())
        if not lengths:
            messagebox.showerror("Error",
                "Enter at least one positive length.")
            return
        try:
            base_len = float(base_var.get()); assert base_len > 0
        except Exception:
            messagebox.showerror("Error",
                "Base length must be a positive number.")
            return
        fit_mode = mode_var.get()
        fill_min = fill_max = fit_blend = None
        try:
            if fit_mode == "band":
                fmn = as_fraction(float(fmin_var.get()))
                fmx = as_fraction(float(fmax_var.get()))
                fill_min, fill_max = min(fmn, fmx), max(fmn, fmx)
            else:
                fit_blend = float(min(1.0, max(0.0,
                                               float(blend_var.get()))))
        except Exception:
            messagebox.showerror("Error",
                "Framing values must be numbers.")
            return
        try:
            os.makedirs(out_var.get().strip(), exist_ok=True)
        except Exception:
            messagebox.showerror("Error", "Cannot create output folder.")
            return
        result.update({
            "ok": True, "mode": "render",
            "objs": list(obj_paths), "bbs": list(bb_paths),
            "out": out_var.get().strip(), "rw": rw, "rh": rh,
            "lengths": lengths, "base_len": base_len,
            "fit_mode": fit_mode,
            "fill_min":  fill_min  if fill_min  is not None else FILL_MIN,
            "fill_max":  fill_max  if fill_max  is not None else FILL_MAX,
            "fit_blend": fit_blend if fit_blend is not None else FIT_BLEND,
            "class_map_path": classmap_var.get().strip() or None,
            "register_new_classes": register_var.get(),
        })
        root.destroy()

    def on_cancel():
        root.destroy()

    def on_resume():
        p = filedialog.askopenfilename(
            title="Select render_job_*.json to resume",
            filetypes=[("Render job manifest", "render_job_*.json"),
                      ("JSON", "*.json"), ("All", "*.*")])
        if not p:
            return
        result.update({"ok": True, "mode": "resume", "manifest": p})
        root.destroy()

    fb = tk.Frame(root, bg="#0a0a14")
    fb.grid(row=10, column=0, columnspan=3,
            sticky="e", padx=12, pady=(8, 16))
    ttk.Button(fb, text="Resume job...",
               command=on_resume).grid(row=0, column=0, padx=(0, 8))
    ttk.Button(fb, text="Cancel",
               command=on_cancel).grid(row=0, column=1, padx=(0, 8))
    ttk.Button(fb, text="Render →",
               command=on_ok).grid(row=0, column=2)

    root.mainloop()
    if not result["ok"]:
        raise SystemExit("Canceled.")
    return result


# ───────────────────────────────────────────────────────────────────
# PROGRESS WINDOW  (background render thread + live poll — Pause/Cancel)
# ───────────────────────────────────────────────────────────────────
def show_progress_gui(worker) -> None:
    """worker(progress_meter, resume_event, stop_event) runs the actual
    render job (run_batch(...) or resume_run(...) with these passed as
    pbar_override/resume_event/stop_event) on a background thread.

    pyrender's OffscreenRenderer is tied to the thread that creates it,
    so the background thread must own the entire render lifecycle; this
    window's main-thread loop only ever reads plain counters off
    `progress_meter` and flips `resume_event`/`stop_event` — it never
    touches the renderer."""
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Rendering...")
    root.configure(bg="#0a0a14")
    root.geometry("500x200")
    root.resizable(False, False)

    resume_event = threading.Event()
    resume_event.set()          # set = running, clear = paused
    stop_event = threading.Event()
    meter = ProgressMeter(total=0)

    style = ttk.Style(root); style.theme_use("clam")
    style.configure("TLabel", background="#0a0a14", foreground="#b0bcd0",
                    font=("Courier New", 10))
    style.configure("TButton", background="#1a2a4a", foreground="#7eb8f7",
                    font=("Courier New", 10, "bold"), padding=4)
    style.map("TButton", background=[("active", "#2a3a6a")])

    status_var = tk.StringVar(value="Starting...")
    ttk.Label(root, textvariable=status_var).pack(pady=(18, 8))

    pb = ttk.Progressbar(root, mode="determinate", length=440, maximum=1)
    pb.pack(pady=4)

    progress_var = tk.StringVar(value="0/0  (0.0%)   time left: --:--:--")
    ttk.Label(root, textvariable=progress_var).pack(pady=(6, 0))

    btns = tk.Frame(root, bg="#0a0a14")
    btns.pack(pady=16)
    pause_text = tk.StringVar(value="Pause")

    def toggle_pause():
        if resume_event.is_set():
            resume_event.clear()
            pause_text.set("Resume")
            status_var.set("Pausing... (finishing current image)")
        else:
            resume_event.set()
            pause_text.set("Pause")
            status_var.set("Rendering...")

    pause_btn = ttk.Button(btns, textvariable=pause_text, command=toggle_pause)
    pause_btn.pack(side="left", padx=6)

    def cancel():
        stop_event.set()
        resume_event.set()      # unblock if currently paused so it can see the stop
        status_var.set("Cancelling... (finishing current image)")
        cancel_btn.config(state="disabled")
        pause_btn.config(state="disabled")

    cancel_btn = ttk.Button(btns, text="Cancel", command=cancel)
    cancel_btn.pack(side="left", padx=6)

    close_btn = ttk.Button(root, text="Close", command=root.destroy, state="disabled")
    close_btn.pack(pady=(4, 12))

    outcome = {}

    def run():
        try:
            worker(meter, resume_event, stop_event)
            outcome["ok"] = True
        except Exception as e:
            outcome["ok"] = False
            outcome["error"] = str(e)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    def poll():
        total, done = meter.total, meter.done
        if total:
            pb.config(maximum=total)
            pb["value"] = done
            pct = 100.0 * done / total
        else:
            pct = 0.0
        progress_var.set(f"{done}/{total}  ({pct:.1f}%)   "
                         f"time left: {format_hms(meter.eta_seconds)}")

        if not thread.is_alive():
            if outcome.get("ok"):
                status_var.set("Paused." if stop_event.is_set() else "Done.")
            else:
                status_var.set("Error — see console for details.")
                print(f"[error] {outcome.get('error', 'unknown error')}")
            pause_btn.config(state="disabled")
            cancel_btn.config(state="disabled")
            close_btn.config(state="normal")
            return

        if not resume_event.is_set():
            status_var.set("Paused.")
        elif not stop_event.is_set():
            status_var.set("Rendering...")
        root.after(300, poll)

    root.after(300, poll)
    root.mainloop()


# ───────────────────────────────────────────────────────────────────
# ENTRY POINT
# ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj",         nargs="+")
    ap.add_argument("--bb",          nargs="+",
                    help="BB .txt files (explicit list)")
    ap.add_argument("--bb-folder",   nargs="+", default=[],
                    help="Folder(s) to scan recursively for BB .txt files")
    ap.add_argument("--out",         default="./renders")
    ap.add_argument("--res",         nargs=2, type=int, default=[800, 800],
                    metavar=("W", "H"))
    ap.add_argument("--lengths",     nargs="+", type=float,
                    default=DEFAULT_LENGTHS)
    ap.add_argument("--base-len",    type=float, default=BASE_LENGTH_MM)
    ap.add_argument("--fit-mode",    choices=["band", "shared"],
                    default=FIT_MODE)
    ap.add_argument("--fill-min",    type=float, default=FILL_MIN)
    ap.add_argument("--fill-max",    type=float, default=FILL_MAX)
    ap.add_argument("--fit-blend",   type=float, default=FIT_BLEND)
    ap.add_argument("--class-map",   default=None,
                    help="Path to class_map.json (default: <out>/class_map.json)")
    ap.add_argument("--register-new-classes", action="store_true",
                    help="Auto-append unknown defect types to the class map "
                         "instead of aborting.")
    ap.add_argument("--resume", default=None,
                    help="Path to a render_job_*.json manifest to resume "
                         "instead of starting a new job. All other render "
                         "params are ignored (taken from the manifest).")
    args = ap.parse_args()

    if args.resume:
        resume_run(args.resume)
    elif args.obj:
        bb_list = list(args.bb or [])
        for folder in args.bb_folder:
            bb_list.extend(collect_bb_files_recursive(folder))
        fmn, fmx = as_fraction(args.fill_min), as_fraction(args.fill_max)
        run_batch(args.obj, bb_list,
                  args.out, args.res[0], args.res[1],
                  lengths=args.lengths, base_len=args.base_len,
                  fit_mode=args.fit_mode,
                  fill_min=min(fmn, fmx), fill_max=max(fmn, fmx),
                  fit_blend=float(min(1.0, max(0.0, args.fit_blend))),
                  class_map_path=args.class_map,
                  register_new_classes=args.register_new_classes)
    else:
        res = run_launcher()
        if res["mode"] == "resume":
            manifest_path = res["manifest"]
            show_progress_gui(
                lambda meter, rev, sev: resume_run(
                    manifest_path, pbar_override=meter,
                    resume_event=rev, stop_event=sev))
        else:
            cfg = res
            show_progress_gui(
                lambda meter, rev, sev: run_batch(
                    cfg["objs"], cfg["bbs"], cfg["out"], cfg["rw"], cfg["rh"],
                    lengths=cfg["lengths"], base_len=cfg["base_len"],
                    fit_mode=cfg["fit_mode"],
                    fill_min=cfg["fill_min"], fill_max=cfg["fill_max"],
                    fit_blend=cfg["fit_blend"],
                    class_map_path=cfg.get("class_map_path"),
                    register_new_classes=cfg.get("register_new_classes", False),
                    pbar_override=meter, resume_event=rev, stop_event=sev))
