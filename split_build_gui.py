"""
split_build_gui.py
==================================================================
EIN Werkzeug: optimierten Split ERZEUGEN und die YOLO-Zip BAUEN, in
einem Lauf.

DREI-WEGE-SPLIT (TRAIN/VAL/TEST)
--------------------------------
Der Split kann optional einen Test-Anteil enthalten (Standard 0.0 ->
reiner train/val-Split; "test:" wird dann auch NICHT in die data.yaml
geschrieben).

Strategie: zweistufig, damit die leckagefreie, stratifizierte Logik
(Zellen aus Schadensart x Groesse x rel_z x rel_x, largest-remainder
je Zelle) wiederverwendet werden kann:
  1. Erste Stufe: train vs. (val+test).
  2. Zweite Stufe: der in Stufe 1 NICHT dem Train zugeteilte Rest
     jeder Zelle wird nochmal per largest-remainder in val vs. test
     geteilt, mit dem Anteil test/(val+test) INNERHALB dieses Rests
     — dadurch treffen alle drei globalen Anteile (train/val/test)
     exakt, ohne dass Rundung sich zwischen den Stufen aufaddiert.

Negative (undefektive) Bilder werden auf Bild-Ebene je Skalen-Bucket
geteilt (gleiches Zweistufen-Prinzip), Schaden-Modelle auf Stem-Ebene.

START
    python split_build_gui.py
"""
import os, re, json, math, hashlib, threading, traceback, zipfile, collections
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext


# ======================================================================
# FESTER POSITIONSRAHMEN DER BASIS-SCHAUFEL  (aus nTop Bounding Box_1)
# Bei Schaufelwechsel hier anpassen.
# ======================================================================
BLADE_X_MIN, BLADE_X_MAX = -29.8296, 11.7774     # Sehne (chord)
BLADE_Z_MIN, BLADE_Z_MAX = -0.55,    82.3054     # Hoehe (span, Wurzel->Spitze)
BLADE_X_SPAN = BLADE_X_MAX - BLADE_X_MIN
BLADE_Z_SPAN = BLADE_Z_MAX - BLADE_Z_MIN


# ----------------------------------------------------------------------
# Einlesen
# ----------------------------------------------------------------------
def parse_stem(base: str) -> str:
    return base.split("__")[0]


def defect_type(stem: str) -> str:
    """Schadensart aus dem Stem: Text vor '_ID'. 'Crack_ID_4' -> 'Crack'."""
    m = re.match(r"(.+?)_ID", stem, re.IGNORECASE)
    return m.group(1).title() if m else "Unknown"


def parse_bb_file(path: str):
    """mm-Groesse UND BB-Mittelpunkt aus einer nTop-BB-Datei."""
    text = open(path, "r", encoding="utf-8", errors="ignore").read()

    def grab(key):
        m = re.search(
            rf"{re.escape(key)}\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", text)
        return float(m.group(1)) if m else None

    xmin, xmax = grab("BB_Min.X"), grab("BB_Max.X")
    zmin, zmax = grab("BB_Min.Z"), grab("BB_Max.Z")
    if None in (xmin, xmax, zmin, zmax):
        return None
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    if zmin > zmax:
        zmin, zmax = zmax, zmin
    w, h = xmax - xmin, zmax - zmin
    cx, cz = 0.5 * (xmin + xmax), 0.5 * (zmin + zmax)
    return {
        "w_mm": w, "h_mm": h,
        "diag_mm": float(np.hypot(w, h)), "area_mm": w * h,
        "rel_x": (cx - BLADE_X_MIN) / BLADE_X_SPAN,
        "rel_z": (cz - BLADE_Z_MIN) / BLADE_Z_SPAN,
    }


def _iter_bb_files(bb_source):
    """bb_source kann ein Ordner (rekursiv durchsucht) oder eine Datei-Liste sein."""
    if isinstance(bb_source, (list, tuple)):
        for p in bb_source:
            if p and p.lower().endswith(".txt") and os.path.isfile(p):
                yield p
    elif bb_source and os.path.isdir(bb_source):
        for root, _dirs, files in os.walk(bb_source):   # rekursiv, auch Unterordner
            for fn in files:
                if fn.lower().endswith(".txt"):
                    yield os.path.join(root, fn)


def build_bb_map(bb_source):
    m = {}
    for p in _iter_bb_files(bb_source):
        m[os.path.splitext(os.path.basename(p))[0].lower()] = p
    return m


def count_bb_files(bb_source):
    return sum(1 for _ in _iter_bb_files(bb_source))


# ----------------------------------------------------------------------
# Bilderquelle: Ordner (rekursiv) ODER explizite Datei-Auswahl
# ----------------------------------------------------------------------
def _iter_image_pairs(img_source):
    """img_source kann ein Ordner (rekursiv durchsucht) oder eine Liste von
    Dateipfaden (.png und/oder .txt gemischt) sein. Liefert
    (png_path, txt_path, base) fuer jedes vollstaendige Paar. "__annotated"
    wird uebersprungen (Annotation_control-Renders)."""
    if isinstance(img_source, (list, tuple)):
        by_key = {}
        for p in img_source:
            if not p or not os.path.isfile(p):
                continue
            d = os.path.dirname(p)
            base, ext = os.path.splitext(os.path.basename(p))
            if "__annotated" in base:
                continue
            ext = ext.lower()
            if ext not in (".png", ".txt"):
                continue
            by_key.setdefault((d, base), {})[ext] = p
        for (_d, base), files in by_key.items():
            png, txt = files.get(".png"), files.get(".txt")
            if png and txt:
                yield png, txt, base
    elif img_source and os.path.isdir(img_source):
        for root, _dirs, files in os.walk(img_source):
            fileset = set(files)
            for fn in files:
                if not fn.lower().endswith(".png") or "__annotated" in fn:
                    continue
                base = os.path.splitext(fn)[0]
                txt_name = base + ".txt"
                if txt_name in fileset:
                    yield os.path.join(root, fn), os.path.join(root, txt_name), base


def count_image_pairs(img_source):
    return sum(1 for _ in _iter_image_pairs(img_source))


def infer_default_dir(img_source):
    """Bester Rueckfallordner fuer Ausgabe/Zip, falls nicht explizit gesetzt.
    Ordner -> der Ordner selbst. Datei-Liste -> None (kein impliziter Ordner,
    da Dateien aus verschiedenen Unterordnern stammen koennen)."""
    if isinstance(img_source, str) and os.path.isdir(img_source):
        return img_source
    return None


# ----------------------------------------------------------------------
# Klassen-Zuordnung (class_map.json) — NUR LESEN.
# view_capture_render.py ist der einzige Schreiber. Zwei Schreiber
# auf dieselbe Datei waeren eine Race Condition; dieses Tool validiert
# nur, dass Labels und Map zusammenpassen, und baut daraus data.yaml.
# ----------------------------------------------------------------------
def load_class_map(path):
    if not path or not os.path.isfile(path):
        raise RuntimeError(
            f"class_map.json nicht gefunden: {path}\n"
            f"        Diese Datei schreibt der Renderer "
            f"(view_capture_render.py) beim ersten Lauf.\n"
            f"        Erst rendern, dann Split/Zip bauen.")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    class_map = {str(k): int(v) for k, v in raw.items()}
    if not class_map:
        raise RuntimeError(f"class_map.json ist leer: {path}")
    return class_map


def build_names_list(class_map):
    """names[] Index = Klassen-ID. Laenge = max_id+1 (YOLO braucht
    lueckenlose IDs 0..nc-1). Falls eine ID reserviert aber aktuell
    keinem Typ zugeordnet ist (Typ wurde umbenannt o.ae.), wird eine
    sichtbare Platzhalter-Bezeichnung eingesetzt statt still eine
    falsche Klasse zu erzeugen."""
    max_id = max(class_map.values())
    by_id = {v: k for k, v in class_map.items()}
    names, gaps = [], []
    for i in range(max_id + 1):
        if i in by_id:
            names.append(by_id[i])
        else:
            names.append(f"unused_id_{i}")
            gaps.append(i)
    return names, gaps


def scan_label_class_ids(pairs):
    """Zaehlt, welche Klassen-IDs tatsaechlich in den Label-.txt-Dateien
    vorkommen, und in wie vielen Dateien. Leere Label-Dateien (kein
    Schaden im Bild) werden uebersprungen, tragen keine ID."""
    counts = collections.Counter()
    file_counts = collections.Counter()   # id -> Anzahl Dateien, die es enthalten
    for _png, txt, _base in pairs:
        try:
            with open(txt, "r", encoding="utf-8") as f:
                lines = [ln for ln in f.read().splitlines() if ln.strip()]
        except OSError:
            continue
        ids_in_file = set()
        for ln in lines:
            tok = ln.split()
            if not tok:
                continue
            try:
                cid = int(tok[0])
            except ValueError:
                continue
            counts[cid] += 1
            ids_in_file.add(cid)
        for cid in ids_in_file:
            file_counts[cid] += 1
    return counts, file_counts


def validate_class_map_against_labels(class_map, pairs, log):
    """Abbruch, wenn Labels IDs enthalten, die nicht in der Map stehen —
    das ist ein Version-Mismatch zwischen Renderer-Lauf und geladener
    class_map.json und darf nicht still in die Zip durchrutschen."""
    box_counts, file_counts = scan_label_class_ids(pairs)
    known_ids = set(class_map.values())
    unknown = sorted(set(box_counts) - known_ids)
    if unknown:
        lines = [f"Label-Dateien enthalten Klassen-ID(s), die NICHT in der "
                f"geladenen class_map.json stehen: {unknown}"]
        for cid in unknown:
            lines.append(f"    ID {cid}: {box_counts[cid]} Box(en) in "
                        f"{file_counts[cid]} Datei(en)")
        lines.append("    Vermutlich wurde mit einer neueren/aelteren "
                    "class_map.json gerendert als der hier geladenen.")
        lines.append("    Richtige class_map.json waehlen oder neu rendern.")
        raise RuntimeError("\n".join(lines))
    log("[info] Klassen-IDs in Labels stimmen mit class_map.json ueberein: %s"
        % ", ".join(f"{k}={v}" for k, v in sorted(class_map.items(),
                                                   key=lambda kv: kv[1])))
    unused = sorted(known_ids - set(box_counts))
    if unused:
        by_id = {v: k for k, v in class_map.items()}
        log("[info] In class_map.json vorhanden, aber in keinem Label dieses "
            "Laufs verwendet: " + ", ".join(f"{by_id.get(i, i)}={i}" for i in unused))


# ----------------------------------------------------------------------
# Negative (undefektive) Modelle — Split PRO BILD, PRO SKALEN-BUCKET
# ----------------------------------------------------------------------
def stem_or_negative(base: str, negative_type: str) -> str:
    """Erkennt ein negatives Bild direkt am Dateinamen-Praefix
    `<negative_type>_`, gross/klein-unempfindlich, UNABHAENGIG von der
    Anzahl Unterstriche danach (Renderer-Ausgaben sind zuverlaessig
    doppelt-unterstrichen, aber real vorkommende Negativ-Dateien waren
    es nicht durchgehend). Alle Bilder eines Negativ-Typs werden so auf
    GENAU EINEN kanonischen Stem-Namen zusammengefuehrt — sonst wuerde
    parse_stem() View+Laenge faelschlich als Teil eines Fake-Stems
    lesen, sobald der doppelte Unterstrich fehlt.

    Faellt fuer alles andere auf das normale parse_stem() zurueck
    (Schaden-Stems, bei denen der Renderer die Benennung korrekt
    schreibt)."""
    if negative_type and base.lower().startswith(negative_type.lower() + "_"):
        return negative_type
    return parse_stem(base)


def negative_stems(df: pd.DataFrame, negative_type: str) -> set:
    """Vergleicht direkt gegen df.stem — stem_or_negative() hat die
    Kanonisierung schon erledigt, jeder negative Stem heisst jetzt
    exakt negative_type. Bewusst NICHT ueber defect_type()/"_ID", denn
    ein negativer Stem muss kein _ID-Suffix tragen."""
    if not negative_type:
        return set()
    return set(df[df.stem.str.lower() == negative_type.lower()].stem)


def parse_scale_bucket(base: str) -> str | None:
    """Laengen-Token (z.B. '256p2mm') per Regex, unabhaengig davon ob
    ein oder zwei Unterstriche drumherum stehen — funktioniert fuer
    sauber doppelt-unterstrichene Schaden-Dateien genauso wie fuer
    inkonsistent benannte Negativ-Dateien. None, falls kein Laengen-
    Token im Namen gefunden wird."""
    m = re.search(r"_(\d+(?:p\d+)?mm)_", base)
    return m.group(1) if m else None


def _length_sort_key(ltag: str) -> float:
    """Numerische Sortierung der Skalen-Buckets (fmt_len-Kodierung
    rueckgaengig gemacht: '105p3mm' -> 105.3)."""
    m = re.match(r"^(\d+)(?:p(\d+))?mm$", ltag)
    if not m:
        return float("inf")
    whole, frac = m.groups()
    return float(whole + ("." + frac if frac else ""))


def split_negatives_by_scale(img_source, neg_stems: set, negative_type: str,
                             train: float, test: float, log):
    """Bild-Ebene-Split fuer negative Stems, stratifiziert je Skalen-
    Bucket (largest-remainder, wie bei den Schaden-Zellen, nur auf
    Bild- statt Modell-Ebene). Jetzt DREI-teilig (train/val/test),
    zweistufig wie split_stratified: erst train vs. Rest, dann der
    Rest in val vs. test. test=0.0 -> nur train/val.

    Rueckgabe: (image_assign, report_lines, total_images_by_split).
    image_assign ist KEY=base-Dateiname (ohne Endung), VALUE=
    "train"|"val"|"test" — nur fuer Bilder negativer Stems.

    Erkennung ueber stem_or_negative(), nicht parse_stem() direkt —
    negative Dateien folgen nicht zwingend dem doppelt-unterstrichenen
    Schema."""
    val = 1.0 - train - test
    image_assign: dict[str, str] = {}
    if not neg_stems:
        return image_assign, [], {"train": 0, "val": 0, "test": 0}

    buckets: dict[str, list[str]] = {}
    for _png, _txt, base in _iter_image_pairs(img_source):
        if stem_or_negative(base, negative_type) not in neg_stems:
            continue
        ltag = parse_scale_bucket(base)
        if ltag is None:
            log("[warn] Negativ-Bild folgt nicht dem Schema "
                "<stem>__view__laenge__licht, uebersprungen: %s" % base)
            continue
        buckets.setdefault(ltag, []).append(base)

    lines = []
    totals = {"train": 0, "val": 0, "test": 0}
    if not buckets:
        return image_assign, lines, totals

    header = ("Negative (undefektive) Bilder — Split je Skalen-Bucket "
              "(Bild-Ebene, kein Stem-Leck-Risiko, da keine "
              "Schaden-Instanz zu schuetzen ist):")
    lines.append(header)
    if test > 0:
        lines.append("%-14s %6s %7s %7s %7s %8s %8s" %
                     ("Skala", "n", "train", "val", "test", "val%", "test%"))
    else:
        lines.append("%-14s %6s %7s %7s %8s" % ("Skala", "n", "train", "val", "val%"))
    for ltag in sorted(buckets, key=_length_sort_key):
        bases = sorted(buckets[ltag],
                       key=lambda b: hashlib.md5(b.encode()).hexdigest())
        n = len(bases)
        nt = int(round(train * n))          # Stufe 1: train vs. Rest
        remaining = n - nt
        if test > 0 and remaining > 0:
            test_share = test / (val + test)
            nte = int(round(test_share * remaining))   # Stufe 2: Rest -> val/test
        else:
            nte = 0
        nv = remaining - nte
        for i, b in enumerate(bases):
            if i < nt:
                image_assign[b] = "train"
            elif i < nt + nv:
                image_assign[b] = "val"
            else:
                image_assign[b] = "test"
        totals["train"] += nt
        totals["val"]   += nv
        totals["test"]  += nte
        if test > 0:
            lines.append("%-14s %6d %7d %7d %7d %7.1f%% %7.1f%%" %
                         (ltag, n, nt, nv, nte,
                          100.0 * nv / n if n else 0.0,
                          100.0 * nte / n if n else 0.0))
        else:
            lines.append("%-14s %6d %7d %7d %7.1f%%" %
                         (ltag, n, nt, nv, 100.0 * nv / n if n else 0.0))
    n_all = totals["train"] + totals["val"] + totals["test"]
    if test > 0:
        lines.append("%-14s %6d %7d %7d %7d %7.1f%% %7.1f%%" %
                     ("GESAMT", n_all, totals["train"], totals["val"], totals["test"],
                      100.0 * totals["val"] / n_all if n_all else 0.0,
                      100.0 * totals["test"] / n_all if n_all else 0.0))
    else:
        lines.append("%-14s %6d %7d %7d %7.1f%%" %
                     ("GESAMT", n_all, totals["train"], totals["val"],
                      100.0 * totals["val"] / n_all if n_all else 0.0))
    return image_assign, lines, totals


def crack_int(stem):
    m = re.search(r"(\d+)\s*$", stem)
    return int(m.group(1)) if m else None


def parse_doe(path):
    """Optional. {crack_id: {S, X, Z}} nur als Report-Metadaten."""
    if not path or not os.path.isfile(path):
        return {}
    import openpyxl
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    hdr = [str(h).strip().lower() if h is not None else "" for h in rows[0]]

    def col(name):
        return hdr.index(name) if name in hdr else None
    ci, cs = col("crack_id"), col("s_value")
    cx, cz = col("x_value"), col("z_value")
    out = {}
    if ci is None:
        return out
    for r in rows[1:]:
        if r[ci] is None:
            continue
        out[int(r[ci])] = {
            "S": r[cs] if cs is not None else None,
            "X": r[cx] if cx is not None else None,
            "Z": r[cz] if cz is not None else None,
        }
    return out


def collect_stems(img_source, bb_folder, doe, log, negative_type: str = "Baseline"):
    """Eine Zeile pro Modell: stem, Bildzahl, Groesse, rel_x, rel_z.
    negative_type: siehe stem_or_negative() — kanonisiert alle Bilder
    eines Negativ-Typs auf EINEN Stem-Namen, unabhaengig von der
    tatsaechlichen Unterstrich-Schreibweise im Dateinamen."""
    bb_map = build_bb_map(bb_folder)
    counts = {}
    for _png, _txt, base in _iter_image_pairs(img_source):
        st = stem_or_negative(base, negative_type)
        counts[st] = counts.get(st, 0) + 1

    rows = []
    for stem, n in counts.items():
        bb = None
        p = bb_map.get(stem.lower())
        if p:
            bb = parse_bb_file(p)
        row = {
            "stem": stem, "defect_type": defect_type(stem), "n_images": n,
            "diag_mm": bb["diag_mm"] if bb else np.nan,
            "rel_x": bb["rel_x"] if bb else np.nan,
            "rel_z": bb["rel_z"] if bb else np.nan,
        }
        if doe:
            d = doe.get(crack_int(stem))
            if d:
                row["S_value"] = d["S"]
                row["X_value"] = d["X"]
                row["Z_value"] = d["Z"]
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("stem").reset_index(drop=True)
    miss = int(df.diag_mm.isna().sum())
    log("[info] %d Modelle, %d Bilder gesamt." %
        (len(df), int(df.n_images.sum())))
    if miss:
        log("[warn] %d Modelle ohne BB-Datei -> nur per Hash geteilt, "
            "nicht stratifiziert." % miss)
    return df


# ----------------------------------------------------------------------
# Split-Strategien (stem-Ebene -> leckagefrei)
# ----------------------------------------------------------------------
def h100(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % 100


def split_hash(stems, salt="", train=0.70, test=0.0):
    """Naive Referenz-Split per MD5-Hash, jetzt optional 3-teilig.
    test=0.0 (Standard) -> nur train/val.
    Dient nur als Vergleichsbasis in score()/den Abbildungen, NICHT
    fuer den tatsaechlich gebauten Split (der ist stratifiziert)."""
    val = 1.0 - train - test
    t_train = int(round(train * 100))
    t_val = int(round((train + val) * 100))
    out = {}
    for s in stems:
        h = h100(salt + s)
        if h < t_train:
            out[s] = "train"
        elif h < t_val:
            out[s] = "val"
        else:
            out[s] = "test"
    return out


def _bincol(series, k):
    if k <= 1:
        return pd.Series(0, index=series.index, dtype=int)
    k = min(k, max(1, series.nunique()))
    return pd.qcut(series.rank(method="first"), q=k, labels=False).astype(int)


def _allocate_binary(cell_sizes, frac, T):
    """Largest-remainder: teilt cell_sizes in ('a'-Anteil laut frac,
    Rest) so, dass die Summe der 'a'-Anteile GLOBAL exakt T ergibt.
    Rueckgabe: dict cell -> Anzahl 'a'-Elemente in dieser Zelle (der
    Rest der Zelle ist 'b'). Wird zweimal genutzt:
    einmal train-vs-Rest, einmal (auf dem Rest) val-vs-test."""
    base, rem = {}, {}
    for c, n in cell_sizes.items():
        exact = frac * n
        base[c] = int(math.floor(exact))
        rem[c] = exact - base[c]
    leftover = T - sum(base.values())
    if leftover > 0:
        for c in sorted(cell_sizes, key=lambda c: rem[c], reverse=True)[:leftover]:
            base[c] += 1
    elif leftover < 0:
        for c in sorted(cell_sizes, key=lambda c: rem[c]):
            if leftover == 0:
                break
            if base[c] > 0:
                base[c] -= 1
                leftover += 1
    return base


def split_stratified(df, n_size, n_relz, n_relx, train, test,
                     exclude_stems: set = frozenset()):
    """Zellen = Schadensart x Groesse x rel_z x rel_x. In jeder Zelle
    deterministisch, global exakt per largest-remainder — jetzt
    DREI-teilig (train/val/test), zweistufig:
      Stufe 1: train vs. (val+test), wie bei einem reinen train/val-Split.
      Stufe 2: der in Stufe 1 uebrige Rest jeder Zelle wird nochmal
               per largest-remainder in val/test geteilt, mit dem
               Anteil test/(val+test) INNERHALB dieses Rests.
    test=0.0 -> Stufe 2 liefert ueberall 0 Test-Elemente.

    exclude_stems: negative Stems, die komplett von dieser Funktion
    ferngehalten werden — sie werden durch split_negatives_by_scale()
    ersetzt (Bild-Ebene statt Stem-Ebene), nicht zusaetzlich per
    Stem-Hash behandelt. Ein Stem OHNE BB, der NICHT in exclude_stems
    steht, ist ein echter Match-Fehler und faellt weiterhin in den
    Hash-Fallback unten — das bleibt der laute Warnpfad, unveraendert."""
    val = 1.0 - train - test
    d = df[~df.stem.isin(exclude_stems)].copy()
    have = d.diag_mm.notna() & d.rel_x.notna() & d.rel_z.notna()
    assign = {}
    sized = d[have].copy()
    if len(sized):
        sized["sb"] = _bincol(sized.diag_mm, n_size)
        sized["zb"] = _bincol(sized.rel_z, n_relz)
        sized["xb"] = _bincol(sized.rel_x, n_relx)
        # Schadensart ist eine BALANCE-Achse: Teil des Zellenschluessels.
        sized["cell"] = list(zip(sized.defect_type,
                                 sized.sb, sized.zb, sized.xb))
        groups = {c: sorted(g.stem,
                            key=lambda s: hashlib.md5(s.encode()).hexdigest())
                  for c, g in sized.groupby("cell")}
        cell_sizes = {c: len(v) for c, v in groups.items()}
        N = sum(cell_sizes.values())

        # Stufe 1: train vs. Rest (val+test)
        T_train = round(train * N)
        train_count = _allocate_binary(cell_sizes, train, T_train)

        # Stufe 2: Rest -> val vs. test, Anteil relativ zum Rest
        remaining_sizes = {c: cell_sizes[c] - train_count[c] for c in cell_sizes}
        R = sum(remaining_sizes.values())
        if test > 0 and R > 0:
            test_share = test / (val + test)
            T_test = round(test_share * R)
            test_count = _allocate_binary(remaining_sizes, test_share, T_test)
        else:
            test_count = {c: 0 for c in cell_sizes}

        for c, order in groups.items():
            nt = train_count[c]
            nte = test_count[c]
            nv = cell_sizes[c] - nt - nte
            for i, s in enumerate(order):
                if i < nt:
                    assign[s] = "train"
                elif i < nt + nv:
                    assign[s] = "val"
                else:
                    assign[s] = "test"
    for s in d[~have].stem:            # echter BB-Match-Fehler -> Hash
        assign.update(split_hash([s], train=train, test=test))
    return assign


def cell_labels(df, n_size, n_relz, n_relx):
    d = df.dropna(subset=["diag_mm", "rel_x", "rel_z"]).copy()
    d["sb"] = _bincol(d.diag_mm, n_size)
    d["zb"] = _bincol(d.rel_z, n_relz)
    d["xb"] = _bincol(d.rel_x, n_relx)
    return d.assign(cell=list(zip(d.defect_type, d.sb, d.zb, d.xb))) \
            .set_index("stem")["cell"]


# ----------------------------------------------------------------------
# Bewertung: KS D auf Groesse, rel_z, rel_x
# ----------------------------------------------------------------------
def _ks(tr, va):
    if len(tr) >= 2 and len(va) >= 2:
        k = stats.ks_2samp(tr, va)
        return float(k.statistic), float(k.pvalue)
    return np.nan, np.nan


def score(df, assign):
    """Jetzt mit Test-Zaehlung/-Vergleich, falls im assign vorhanden.
    KS D wird weiterhin primaer train-vs-val berichtet (die Achse, auf
    der bisher optimiert wird); train-vs-test wird zusaetzlich
    berechnet, wenn Test-Elemente existieren, rein informativ."""
    d = df.copy()
    d["split"] = d.stem.map(assign)
    s = d.dropna(subset=["diag_mm", "rel_x", "rel_z"])
    tr = s[s.split == "train"]
    va = s[s.split == "val"]
    te = s[s.split == "test"]
    res = {
        "n_train": int((d.split == "train").sum()),
        "n_val": int((d.split == "val").sum()),
        "n_test": int((d.split == "test").sum()),
        "img_train": int(d[d.split == "train"].n_images.sum()),
        "img_val": int(d[d.split == "val"].n_images.sum()),
        "img_test": int(d[d.split == "test"].n_images.sum()),
    }
    res["train_frac"] = res["n_train"] / max(1, len(d))
    res["test_frac"] = res["n_test"] / max(1, len(d))
    for key, col in [("size", "diag_mm"), ("relz", "rel_z"), ("relx", "rel_x")]:
        D, p = _ks(tr[col], va[col])
        res[key + "_D"], res[key + "_p"] = D, p
        if len(te):
            Dt, pt = _ks(tr[col], te[col])
            res[key + "_D_test"], res[key + "_p_test"] = Dt, pt
    return res


# ----------------------------------------------------------------------
# Abbildungen
# ----------------------------------------------------------------------
def per_type_lines(df):
    """Groessen- und Positions-Streuung je Schadensart."""
    d = df.dropna(subset=["diag_mm", "rel_x", "rel_z"])
    L = []
    L.append("%-10s %4s  %-27s %-14s %-14s"
             % ("Typ", "n", "diag_mm  min/med/max/std",
                "rel_z med/std", "rel_x med/std"))
    for t, g in d.groupby("defect_type"):
        L.append("%-10s %4d  %6.1f/%6.1f/%6.1f/%5.2f   %.2f/%.2f    %.2f/%.2f"
                 % (t, len(g), g.diag_mm.min(), g.diag_mm.median(),
                    g.diag_mm.max(), g.diag_mm.std(),
                    g.rel_z.median(), g.rel_z.std(),
                    g.rel_x.median(), g.rel_x.std()))
    return L


def fig_by_type(df, out_path):
    d = df.dropna(subset=["diag_mm", "rel_x", "rel_z"])
    types = sorted(d.defect_type.unique())
    cols = [("diag_mm", "BB-Diagonale [mm]"),
            ("rel_z", "rel. Hoehe (Wurzel->Spitze)"),
            ("rel_x", "rel. Sehne (Kante->Kante)")]
    fig, ax = plt.subplots(len(types), 3, figsize=(14, 3.1 * len(types)),
                           squeeze=False)
    for i, t in enumerate(types):
        g = d[d.defect_type == t]
        for j, (col, xl) in enumerate(cols):
            a = ax[i, j]
            a.hist(g[col], bins=20, color="#4C72B0", edgecolor="white")
            a.set_xlabel(xl)
            if col in ("rel_z", "rel_x"):
                a.set_xlim(0, 1)
            if j == 0:
                a.set_ylabel("%s\nAnzahl" % t)
    fig.suptitle("Verteilung je Schadensart: Groesse und Position")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _ecdf(ax, series, label):
    x = np.sort(series.values)
    if len(x):
        ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", label=label)


def fig_ecdf(df, md5, strat, out_path, has_test: bool):
    """train/val/test ECDF je Merkmal. Zeichnet die test-Kurve nur,
    wenn has_test True ist (test>0) — sonst nur train/val."""
    props = [("diag_mm", "BB-Diagonale [mm]", "Groesse"),
             ("rel_z", "rel. Hoehe (Wurzel->Spitze)", "Position Z"),
             ("rel_x", "rel. Sehne (Kante->Kante)", "Position X")]
    strategies = [("MD5 (Referenz)", md5), ("Stratifiziert", strat)]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for r, (sname, assign) in enumerate(strategies):
        d = df.dropna(subset=["diag_mm", "rel_x", "rel_z"]).copy()
        d["split"] = d.stem.map(assign)
        for c, (col, xlabel, pretty) in enumerate(props):
            a = ax[r, c]
            tr = d[d.split == "train"][col]
            va = d[d.split == "val"][col]
            D, _ = _ks(tr, va)
            _ecdf(a, tr, "train")
            _ecdf(a, va, "val")
            if has_test:
                te = d[d.split == "test"][col]
                _ecdf(a, te, "test")
            a.set_ylim(0, 1)
            a.set_xlabel(xlabel)
            if c == 0:
                a.set_ylabel(sname + "\nECDF")
            a.set_title("%s  (KS D train/val=%.3f)" % (pretty, D))
            a.legend(fontsize=8)
    fig.suptitle("Train vs. Val (vs. Test) je Merkmal: MD5-Referenz (oben) "
                 "vs. stratifiziert (unten)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fig_coverage(df, md5, strat, out_path, has_test: bool):
    d0 = df.dropna(subset=["diag_mm", "rel_x", "rel_z"]).copy()
    smax = d0.diag_mm.max() or 1.0
    types = sorted(d0.defect_type.unique())
    markers = dict(zip(types, ["o", "s", "^", "D", "v", "P"]))
    colors = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}
    fig, ax = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for i, (name, assign) in enumerate([("MD5 (Referenz)", md5),
                                        ("Stratifiziert", strat)]):
        d = d0.copy()
        d["split"] = d.stem.map(assign)
        splits = ("train", "val", "test") if has_test else ("train", "val")
        for t in types:
            for s in splits:
                sub = d[(d.split == s) & (d.defect_type == t)]
                if not len(sub):
                    continue
                ax[i].scatter(sub.rel_x, sub.rel_z,
                              s=20 + 120 * sub.diag_mm / smax,
                              alpha=0.6, label="%s/%s" % (t, s),
                              color=colors[s], marker=markers[t],
                              edgecolor="k", linewidth=0.3)
        ax[i].set_title(name)
        ax[i].set_xlabel("rel. Sehne X (Kante->Kante)")
        ax[i].set_xlim(0, 1)
        ax[i].set_ylim(0, 1)
        ax[i].legend()
    ax[0].set_ylabel("rel. Hoehe Z (Wurzel->Spitze)")
    fig.suptitle("Positionsabdeckung Train vs. Val" +
                 (" vs. Test" if has_test else "") +
                 " (Punktgroesse ~ Schadensgroesse)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------
# Zip-Build (aus fertiger Zuordnung, kein JSON-Umweg)
# ----------------------------------------------------------------------
def build_zip(img_source, stem_assign, neg_stems, image_assign, class_map,
             zip_name, log, zip_dir, train=0.70, test=0.0,
             negative_type="Baseline"):
    """stem_assign: Stem->Split fuer Schaden-Stems (aus split_stratified,
    enthaelt KEINE negativen Stems mehr). neg_stems: Menge der negativen
    Stems. image_assign: Bildname->Split fuer negative Bilder (aus
    split_negatives_by_scale, Bild-Ebene). Ein Bild eines negativen
    Stems schlaegt IMMER in image_assign nach, nie in stem_assign — das
    ist der Punkt: verschiedene Bilder DESSELBEN negativen Stems duerfen
    auf verschiedenen Seiten landen. Erkennung ueber stem_or_negative(),
    nicht parse_stem() direkt, aus demselben Grund wie in collect_stems()
    und split_negatives_by_scale().

    test>0 -> "test:" Zeile wird in die data.yaml geschrieben und
    dataset/images(labels)/test/ entsteht im Zip. test=0.0 -> data.yaml
    enthaelt kein test-Feld, kein test-Ordner."""
    pairs = list(_iter_image_pairs(img_source))
    if not pairs:
        log("[warn] keine .png+.txt Paare fuer den Build.")
        return

    validate_class_map_against_labels(class_map, pairs, log)
    names, gaps = build_names_list(class_map)
    if gaps:
        log("[warn] Reservierte ID(s) ohne aktuellen Typ (Platzhalter in "
            "data.yaml): %s" % ", ".join(map(str, gaps)))
    log("[info] data.yaml Klassen (nc=%d): %s" % (len(names), names))

    fallback_stem = set()
    fallback_img = set()
    counts = {"train": 0, "val": 0, "test": 0}
    yaml_lines = [
        "nc: %d" % len(names),
        "names: [%s]" % ", ".join("'%s'" % n for n in names),
        "train: images/train",
        "val:   images/val",
    ]
    if test > 0:
        yaml_lines.append("test:  images/test")
    yaml = "\n".join(yaml_lines) + "\n"

    zp = os.path.join(zip_dir, zip_name)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("dataset/data.yaml", yaml)
        for png, txt, base in pairs:
            stem = stem_or_negative(base, negative_type)
            if stem in neg_stems:
                sp = image_assign.get(base)
                if sp is None:
                    sp = split_hash([base], train=train, test=test)[base]
                    fallback_img.add(base)
            else:
                sp = stem_assign.get(stem)
                if sp is None:
                    sp = split_hash([stem], train=train, test=test)[stem]
                    fallback_stem.add(stem)
            counts[sp] += 1
            z.write(png, "dataset/images/%s/%s" % (sp, os.path.basename(png)))
            z.write(txt, "dataset/labels/%s/%s" % (sp, os.path.basename(txt)))
    if fallback_stem:
        log("[warn] %d Schaden-Stem(s) nicht in der Zuordnung, per Hash: %s"
            % (len(fallback_stem), ", ".join(sorted(fallback_stem)[:5])))
    if fallback_img:
        log("[warn] %d negatives Bild(er) nicht in der Bild-Zuordnung, per Hash: %s"
            % (len(fallback_img), ", ".join(sorted(fallback_img)[:5])))
    if test > 0:
        log("[done] Zip -> %s   train=%d val=%d test=%d total=%d Bilder"
            % (zp, counts["train"], counts["val"], counts["test"], sum(counts.values())))
    else:
        log("[done] Zip -> %s   train=%d val=%d total=%d Bilder"
            % (zp, counts["train"], counts["val"], sum(counts.values())))


# ----------------------------------------------------------------------
# Hauptlauf
# ----------------------------------------------------------------------
def run_all(img_source, bb_folder, doe_path, class_map_path, train, test,
            n_size, n_relz, n_relx, do_build, zip_name, log,
            out_dir=None, zip_dir=None, negative_type="Baseline"):
    # img_source: Ordner (rekursiv) ODER Liste von .png/.txt Dateipfaden.
    # out_dir: Ziel fuer Reports/Abbildungen/CSV/JSON.
    # zip_dir: Ziel fuer die Zip.
    # Beide muessen explizit gesetzt sein, wenn img_source eine Datei-Liste
    # ist (kein impliziter Ordner, auf den zurueckgefallen werden koennte).
    #
    # train, test: Anteile in [0,1). val wird als 1-train-test abgeleitet
    # und muss > 0 sein (sonst RuntimeError) — es gibt bewusst kein
    # separates val-Feld, damit train+val+test=1 nie durch Rundung
    # auseinanderlaufen kann.
    if not (0 < train < 1):
        raise RuntimeError("Train-Anteil muss zwischen 0 und 1 liegen.")
    if not (0 <= test < 1):
        raise RuntimeError("Test-Anteil muss zwischen 0 und 1 liegen (0 = kein Test-Split).")
    val = 1.0 - train - test
    if val <= 0:
        raise RuntimeError(
            "train + test muss < 1 sein, damit ein Val-Anteil > 0 bleibt "
            "(aktuell train=%.3f, test=%.3f -> val=%.3f)." % (train, test, val))

    default_dir = infer_default_dir(img_source)
    if out_dir is None:
        if default_dir is None:
            raise RuntimeError(
                "Ausgabeordner muss explizit gesetzt werden, wenn die "
                "Bilderquelle eine Datei-Auswahl (kein Ordner) ist.")
        out_dir = default_dir
    if zip_dir is None:
        if default_dir is None:
            raise RuntimeError(
                "Zip-Ordner muss explizit gesetzt werden, wenn die "
                "Bilderquelle eine Datei-Auswahl (kein Ordner) ist.")
        zip_dir = default_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(zip_dir, exist_ok=True)
    doe = parse_doe(doe_path)
    if doe:
        log("[info] DoE geladen (nur Report-Metadaten): %d Eintraege." % len(doe))
    df = collect_stems(img_source, bb_folder, doe, log, negative_type=negative_type)
    if df.empty:
        raise RuntimeError("Keine Modelle gefunden.")

    neg_stems = negative_stems(df, negative_type)
    if neg_stems:
        log("[info] %d negative(s) Modell(e) erkannt (Typ-Praefix '%s'): %s"
            % (len(neg_stems), negative_type, ", ".join(sorted(neg_stems))))

    has_test = test > 0
    defect_df = df[~df.stem.isin(neg_stems)]
    md5 = split_hash(defect_df.stem, salt="", train=train, test=test)
    strat = split_stratified(df, n_size, n_relz, n_relx, train, test,
                             exclude_stems=neg_stems)
    s_md5, s_str = score(defect_df, md5), score(defect_df, strat)

    image_assign, neg_report_lines, neg_totals = split_negatives_by_scale(
        img_source, neg_stems, negative_type, train, test, log)

    sfx = "train%02d" % round(train * 100)
    if has_test:
        sfx += "_test%02d" % round(test * 100)
    R = []
    def p(x): R.append(x)

    p("=" * 72)
    p("SPLIT ERZEUGEN + ZIP BAUEN  (leckagefrei, Groesse + Position)")
    p("=" * 72)
    if has_test:
        p("\nModelle: %d   Bilder: %d   Ziel train=%.0f%%  val=%.0f%%  test=%.0f%%"
          % (len(df), int(df.n_images.sum()), train * 100, val * 100, test * 100))
    else:
        p("\nModelle: %d   Bilder: %d   Ziel train=%.0f%%"
          % (len(df), int(df.n_images.sum()), train * 100))
    if neg_stems:
        p("  davon negativ (Typ '%s'): %d Modell(e), %d Bild(er) — "
          "Split auf Bild-Ebene, siehe unten"
          % (negative_type, len(neg_stems),
             neg_totals["train"] + neg_totals["val"] + neg_totals["test"]))
    p("Positionsrahmen Basis-Schaufel:  X[%.3f, %.3f]  Z[%.3f, %.3f]"
      % (BLADE_X_MIN, BLADE_X_MAX, BLADE_Z_MIN, BLADE_Z_MAX))

    sized = df.dropna(subset=["diag_mm", "rel_x", "rel_z"])
    if len(sized):
        p("\nStreuung der Merkmale (Standardabweichung):")
        p("  Groesse diag_mm : %.3f" % sized.diag_mm.std())
        p("  rel_z (Hoehe)   : %.3f   %s"
          % (sized.rel_z.std(),
             "-> lohnt Bins" if sized.rel_z.std() > 0.05 else "-> kaum Variation"))
        p("  rel_x (Sehne)   : %.3f   %s"
          % (sized.rel_x.std(),
             "-> lohnt Bins" if sized.rel_x.std() > 0.05 else "-> kaum Variation"))
    n_types = df.dropna(subset=["diag_mm"]).defect_type.nunique()
    p("\nBins:  Typ=%d(Balance)  Groesse=%d  rel_z=%d  rel_x=%d  -> bis %d Zellen"
      % (n_types, n_size, n_relz, n_relx,
         n_types * max(1, n_size) * max(1, n_relz) * max(1, n_relx)))

    p("\n" + "-" * 72)
    p("JE SCHADENSART  (Groessen- und Positions-Streuung pro Typ)")
    for line in per_type_lines(df):
        p(line)

    p("\n" + "-" * 72)
    p("VERGLEICH  (KS D klein = Train/Val aehnlich; MD5 nur als Referenz)")
    p("%-18s %8s %8s %8s %8s" %
      ("Strategie", "D_size", "D_relz", "D_relx", "train%"))
    for name, s in [("MD5 (Referenz)", s_md5), ("Stratifiziert", s_str)]:
        p("%-18s %8.3f %8.3f %8.3f %8.1f" %
          (name, s["size_D"], s["relz_D"], s["relx_D"], s["train_frac"] * 100))
    if has_test:
        p("\nZusaetzlich train-vs-test (rein informativ, Stratifiziert):")
        p("  D_size=%.3f  D_relz=%.3f  D_relx=%.3f  test%%=%.1f"
          % (s_str.get("size_D_test", float("nan")),
             s_str.get("relz_D_test", float("nan")),
             s_str.get("relx_D_test", float("nan")),
             s_str["test_frac"] * 100))
    p("\nGebauter Split = Stratifiziert.  Bilder train/val%s = %d/%d%s"
      % ("/test" if has_test else "", s_str["img_train"], s_str["img_val"],
         ("/%d" % s_str["img_test"]) if has_test else ""))

    # Val-Anteil je Zelle
    cells = cell_labels(df, n_size, n_relz, n_relx)
    dd = df.set_index("stem")
    dd = dd.loc[cells.index]
    dd["cell"] = cells.astype(str)
    dd["split"] = dd.index.map(strat)
    ct = pd.crosstab(dd["cell"], dd["split"])
    for col in ("train", "val", "test"):
        if col not in ct:
            ct[col] = 0
    ct["val_frac"] = (ct["val"] / ct[["train", "val", "test"]].sum(axis=1)).round(2)
    if has_test:
        ct["test_frac"] = (ct["test"] / ct[["train", "val", "test"]].sum(axis=1)).round(2)
    p("\nVal-Anteil je Zelle (Ziel ~%.2f)%s, Zelle=(Groesse,relZ,relX):"
      % (val, ", Test-Anteil (Ziel ~%.2f)" % test if has_test else ""))
    p(ct.to_string())

    # Leckage
    seen = {}
    for stem, spl in strat.items():
        seen.setdefault(stem, set()).add(spl)
    leak = sum(1 for v in seen.values() if len(v) > 1)
    # Train/Val/Test-Balance je Schadensart (prueft, dass kein Typ einseitig faellt)
    dt = df.dropna(subset=["diag_mm"]).copy()
    dt["split"] = dt.stem.map(strat)
    ctt = pd.crosstab(dt.defect_type, dt.split)
    for c in ("train", "val", "test"):
        if c not in ctt:
            ctt[c] = 0
    ctt["val_frac"] = (ctt["val"] / ctt[["train", "val", "test"]].sum(axis=1)).round(2)
    if has_test:
        ctt["test_frac"] = (ctt["test"] / ctt[["train", "val", "test"]].sum(axis=1)).round(2)
    p("\nTrain/Val%s je Schadensart (Ziel val ~%.2f%s):"
      % ("/Test" if has_test else "", val,
         ", test ~%.2f" % test if has_test else ""))
    p(ctt.to_string())

    p("\n" + "-" * 72)
    p("LECKAGE-CHECK: %d Modelle in mehr als einem Split (muss 0 sein)." % leak)

    if neg_report_lines:
        p("\n" + "-" * 72)
        for line in neg_report_lines:
            p(line)

    # Export (nach out_dir)
    df_out = df.copy()
    df_out["split"] = df_out.stem.map(strat)
    df_out.loc[df_out.stem.isin(neg_stems), "split"] = "mixed(image-level, siehe negative_split_%s)" % sfx
    df_out["cell"] = df_out.stem.map(
        lambda s: str(cells.get(s, "")) if s in cells.index else "")
    df_out.to_csv(os.path.join(out_dir, "split_assignment_%s.csv" % sfx),
                  index=False)
    with open(os.path.join(out_dir, "split_assignment_%s.json" % sfx), "w") as f:
        json.dump(strat, f, indent=0)
    if image_assign:
        with open(os.path.join(out_dir, "negative_split_%s.json" % sfx), "w") as f:
            json.dump(image_assign, f, indent=0)
        neg_csv_rows = [{"base": b, "scale": parse_scale_bucket(b), "split": s}
                        for b, s in image_assign.items()]
        pd.DataFrame(neg_csv_rows).to_csv(
            os.path.join(out_dir, "negative_split_%s.csv" % sfx), index=False)
    with open(os.path.join(out_dir, "split_precheck_report_%s.txt" % sfx),
              "w", encoding="utf-8") as f:
        f.write("\n".join(R))
    fig_ecdf(defect_df, md5, strat, os.path.join(out_dir, "split_ecdf_%s.png" % sfx), has_test)
    fig_coverage(defect_df, md5, strat,
                 os.path.join(out_dir, "split_coverage_%s.png" % sfx), has_test)
    fig_by_type(defect_df, os.path.join(out_dir, "split_by_type_%s.png" % sfx))

    if do_build:
        p("\n" + "-" * 72)
        class_map = load_class_map(class_map_path)   # raises loud if missing/empty
        build_zip(img_source, strat, neg_stems, image_assign, class_map,
                 zip_name, p, zip_dir=zip_dir, train=train, test=test,
                 negative_type=negative_type)

    return "\n".join(R), sfx, out_dir


# ======================================================================
# GUI
# ======================================================================
class App:
    def __init__(self, root):
        self.root = root
        root.title("Split erzeugen + Zip bauen  (Groesse + Position, leckagefrei)")
        root.geometry("940x820")
        pad = {"padx": 6, "pady": 3}
        frm = ttk.Frame(root, padding=10)
        frm.pack(fill="both", expand=True)

        r = 0

        # --- Bilderquelle-Zeile: Ordner ODER mehrere Dateien, mit Zaehler ---
        self._img_source = None           # str (Ordner) oder Liste (Dateien)
        ttk.Label(frm, text="Bilderquelle (.png + .txt):").grid(
            row=r, column=0, sticky="w", **pad)
        self.img = tk.StringVar()
        self.img.trace_add("write", self._on_img_typed)
        ttk.Entry(frm, textvariable=self.img, width=66).grid(
            row=r, column=1, sticky="we", **pad)
        imgbtns = ttk.Frame(frm); imgbtns.grid(row=r, column=2, **pad)
        ttk.Button(imgbtns, text="Ordner", width=8,
                   command=self._pick_img_folder).pack(side="left", padx=1)
        ttk.Button(imgbtns, text="Dateien", width=8,
                   command=self._pick_img_files).pack(side="left", padx=1)
        r += 1
        self.img_count = tk.StringVar(value="gefundene Bildpaare: 0")
        ttk.Label(frm, textvariable=self.img_count,
                  foreground="#555").grid(row=r, column=1, sticky="w",
                                          padx=6); r += 1

        # --- BB-Zeile: Ordner ODER mehrere Dateien, mit Zaehler darunter ---
        self._bb_source = None            # str (Ordner) oder Liste (Dateien)
        ttk.Label(frm, text="BB (nTop mm-Dateien):").grid(
            row=r, column=0, sticky="w", **pad)
        self.bb = tk.StringVar()
        self.bb.trace_add("write", self._on_bb_typed)
        ttk.Entry(frm, textvariable=self.bb, width=66).grid(
            row=r, column=1, sticky="we", **pad)
        bbbtns = ttk.Frame(frm); bbbtns.grid(row=r, column=2, **pad)
        ttk.Button(bbbtns, text="Ordner", width=8,
                   command=self._pick_bb_folder).pack(side="left", padx=1)
        ttk.Button(bbbtns, text="Dateien", width=8,
                   command=self._pick_bb_files).pack(side="left", padx=1)
        r += 1
        self.bb_count = tk.StringVar(value="gefundene BB-Dateien: 0")
        ttk.Label(frm, textvariable=self.bb_count,
                  foreground="#555").grid(row=r, column=1, sticky="w",
                                          padx=6); r += 1

        self.doe    = self._row(frm, r, "DoE .xlsx (optional, nur Report):",
                                dir=False); r += 1
        self.outdir = self._row(frm, r, "Ausgabeordner "
                                "(Pflicht bei Datei-Auswahl oben):",
                                dir=True); r += 1
        self.zipdir = self._row(frm, r, "Zip-Ordner "
                                "(Pflicht bei Datei-Auswahl oben):",
                                dir=True); r += 1
        ttk.Label(frm, text="Klassen-Zuordnung class_map.json "
                            "(vom Renderer geschrieben, Pflicht fuer Zip-Bau):").grid(
            row=r, column=0, sticky="w", **pad)
        self.classmap = tk.StringVar()
        ttk.Entry(frm, textvariable=self.classmap, width=66).grid(
            row=r, column=1, sticky="we", **pad)
        ttk.Button(frm, text="Durchsuchen",
                   command=lambda: self.classmap.set(
                       filedialog.askopenfilename(
                           title="class_map.json waehlen",
                           filetypes=[("JSON", "*.json"), ("Alle", "*.*")])
                       or self.classmap.get())
                   ).grid(row=r, column=2, **pad)
        r += 1

        ttk.Separator(frm, orient="horizontal").grid(
            row=r, column=0, columnspan=3, sticky="we", pady=8); r += 1

        opt = ttk.Frame(frm); opt.grid(row=r, column=0, columnspan=3,
                                       sticky="we"); r += 1
        self.train   = self._field(opt, 0, 0, "Train-Anteil (0-1):", "0.70", 8)
        self.testfrac = self._field(opt, 0, 2, "Test-Anteil (0-1, 0=kein Test):", "0.00", 8)
        self.nsize   = self._field(opt, 1, 0, "Bins Groesse:", "2", 8)
        self.nrelz   = self._field(opt, 1, 2, "Bins rel_z (Hoehe):", "2", 8)
        self.nrelx   = self._field(opt, 2, 0, "Bins rel_x (Sehne):", "1", 8)
        self.zipname = self._field(opt, 2, 2, "Zip-Name:", "yolo_dataset.zip", 22)
        self.negtype = self._field(opt, 3, 0, "Negativ-Typ (Baseline):", "Baseline", 22)
        ttk.Label(opt, text="(Val-Anteil wird als 1-Train-Test berechnet)",
                  foreground="#555").grid(row=3, column=2, columnspan=2,
                                          sticky="w", padx=6, pady=3)

        self.do_build = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text="Nach dem Split die Zip bauen",
                        variable=self.do_build).grid(
            row=r, column=0, columnspan=3, sticky="w", **pad); r += 1

        self.run_btn = ttk.Button(frm, text="Split erzeugen + bauen",
                                  command=self.start)
        self.run_btn.grid(row=r, column=0, columnspan=3, pady=8); r += 1

        self.out = scrolledtext.ScrolledText(frm, height=24, width=118,
                                             font=("Consolas", 9))
        self.out.grid(row=r, column=0, columnspan=3, sticky="nsew", **pad)
        frm.rowconfigure(r, weight=1)
        frm.columnconfigure(1, weight=1)

    def _row(self, frm, r, label, dir=True):
        pad = {"padx": 6, "pady": 3}
        ttk.Label(frm, text=label).grid(row=r, column=0, sticky="w", **pad)
        var = tk.StringVar()
        ttk.Entry(frm, textvariable=var, width=66).grid(
            row=r, column=1, sticky="we", **pad)
        cmd = ((lambda: self._pick_dir(var)) if dir
               else (lambda: self._pick_file(var)))
        ttk.Button(frm, text="Durchsuchen", command=cmd).grid(
            row=r, column=2, **pad)
        return var

    def _field(self, parent, r, c, label, default, width):
        ttk.Label(parent, text=label).grid(row=r, column=c, sticky="w",
                                           padx=6, pady=3)
        var = tk.StringVar(value=default)
        ttk.Entry(parent, textvariable=var, width=width).grid(
            row=r, column=c + 1, sticky="w", padx=6, pady=3)
        return var

    def _pick_dir(self, var):
        d = filedialog.askdirectory()
        if d:
            var.set(d)

    def _pick_file(self, var):
        f = filedialog.askopenfilename(
            filetypes=[("Excel", "*.xlsx"), ("Alle", "*.*")])
        if f:
            var.set(f)

    # -- Bilderquelle --
    def _pick_img_folder(self):
        d = filedialog.askdirectory(title="Bilderordner (wird rekursiv durchsucht)")
        if d:
            self._img_source = d
            self.img.set(d)
            self._refresh_img_count()

    def _pick_img_files(self):
        fs = filedialog.askopenfilenames(
            title="Bilddateien waehlen (.png und/oder .txt, mehrere moeglich)",
            filetypes=[("PNG + Text", "*.png *.txt"), ("Alle", "*.*")])
        if fs:
            self._img_source = list(fs)
            self.img.set("%d Dateien gewaehlt" % len(fs))
            self._refresh_img_count()

    def _on_img_typed(self, *args):
        text = self.img.get().strip()
        if text and os.path.isdir(text):
            self._img_source = text
        elif not text:
            self._img_source = None
        self._refresh_img_count()

    def _refresh_img_count(self):
        src = self._img_source
        try:
            n = count_image_pairs(src) if src else 0
        except Exception:
            n = 0
        self.img_count.set("gefundene Bildpaare: %d%s"
                           % (n, "  (rekursiv)" if isinstance(src, str) else ""))

    # -- BB-Quelle --
    def _pick_bb_folder(self):
        d = filedialog.askdirectory(title="BB-Ordner (wird rekursiv durchsucht)")
        if d:
            self._bb_source = d
            self.bb.set(d)
            self._refresh_bb_count()

    def _pick_bb_files(self):
        fs = filedialog.askopenfilenames(
            title="BB-Dateien waehlen (mehrere moeglich)",
            filetypes=[("Text", "*.txt"), ("Alle", "*.*")])
        if fs:
            self._bb_source = list(fs)
            self.bb.set("%d Dateien gewaehlt" % len(fs))
            self._refresh_bb_count()

    def _on_bb_typed(self, *args):
        text = self.bb.get().strip()
        if text and os.path.isdir(text):
            self._bb_source = text          # getippter oder gewaehlter Ordner
        elif not text:
            self._bb_source = None
        # sonst: bestehende Dateien-Liste behalten (Platzhaltertext)
        self._refresh_bb_count()

    def _refresh_bb_count(self):
        src = self._bb_source
        try:
            n = count_bb_files(src) if src else 0
        except Exception:
            n = 0
        self.bb_count.set("gefundene BB-Dateien: %d%s"
                          % (n, "  (rekursiv)" if isinstance(src, str) else ""))

    def log(self, m):
        self.out.insert("end", m + "\n")
        self.out.see("end")
        self.root.update_idletasks()

    def start(self):
        # Bilderquelle: gewaehlte Dateien-Liste, gewaehlter Ordner, oder getippter Pfad
        img_source = self._img_source
        if img_source is None:
            typed = self.img.get().strip()
            img_source = typed if typed and os.path.isdir(typed) else None
        if img_source is None:
            self.log("[error] Bilderquelle waehlen (Ordner oder Dateien)."); return
        self._refresh_img_count()
        n_img = count_image_pairs(img_source)
        if n_img == 0:
            self.log("[error] Keine .png+.txt Paare in der Auswahl gefunden.")
            return
        self.log("[info] Bildpaare in der Auswahl: %d" % n_img)

        # BB-Quelle: gewaehlte Dateien-Liste, gewaehlter Ordner, oder getippter Pfad
        bb = self._bb_source
        if bb is None:
            typed = self.bb.get().strip()
            bb = typed if typed and os.path.isdir(typed) else None
        if isinstance(bb, str) and not os.path.isdir(bb):
            self.log("[error] BB-Ordner ungueltig."); return
        if bb:
            self._refresh_bb_count()
            n_bb = count_bb_files(bb)
            if n_bb == 0:
                self.log("[error] Keine BB-.txt-Dateien in der Auswahl gefunden.")
                return
            self.log("[info] BB-Dateien in der Auswahl: %d" % n_bb)
        doe = self.doe.get().strip() or None
        try:
            train = float(self.train.get())
            test = float(self.testfrac.get())
            n_size = int(self.nsize.get())
            n_relz = int(self.nrelz.get())
            n_relx = int(self.nrelx.get())
            assert 0 < train < 1 and n_size >= 1 and n_relz >= 1 and n_relx >= 1
            assert 0 <= test < 1
            assert train + test < 1
        except AssertionError:
            self.log("[error] Train zwischen 0 und 1, Test zwischen 0 und 1 "
                     "(0 = kein Test-Split), train+test < 1 (damit Val > 0 "
                     "bleibt), Bins ganzzahlig >=1.")
            return
        except Exception:
            self.log("[error] Train/Test/Bins nicht lesbar (Zahl erwartet).")
            return
        class_map_path = self.classmap.get().strip()
        if self.do_build.get() and not class_map_path:
            self.log("[error] class_map.json Pfad ist Pflicht, wenn die Zip "
                     "gebaut wird. Vom Renderer geschrieben, siehe Feld "
                     "'Klassen-Zuordnung'.")
            return
        zip_name = self.zipname.get().strip() or "yolo_dataset.zip"
        do_build = self.do_build.get()
        out_dir = self.outdir.get().strip() or None
        zip_dir = self.zipdir.get().strip() or None

        if out_dir is None and infer_default_dir(img_source) is None:
            self.log("[error] Ausgabeordner ist Pflicht, wenn die Bilderquelle "
                     "eine Datei-Auswahl ist (kein einzelner Ordner).")
            return
        if zip_dir is None and infer_default_dir(img_source) is None:
            self.log("[error] Zip-Ordner ist Pflicht, wenn die Bilderquelle "
                     "eine Datei-Auswahl ist (kein einzelner Ordner).")
            return
        if out_dir and not os.path.isdir(out_dir):
            try:
                os.makedirs(out_dir, exist_ok=True)
            except Exception:
                self.log("[error] Ausgabeordner ungueltig."); return
        if zip_dir and not os.path.isdir(zip_dir):
            try:
                os.makedirs(zip_dir, exist_ok=True)
            except Exception:
                self.log("[error] Zip-Ordner ungueltig."); return

        self.run_btn.config(state="disabled")
        self.out.delete("1.0", "end")

        def worker():
            try:
                report, sfx, used_out = run_all(
                    img_source, bb, doe, class_map_path, train, test,
                    n_size, n_relz, n_relx, do_build,
                    zip_name, self.log, out_dir=out_dir, zip_dir=zip_dir,
                    negative_type=self.negtype.get().strip() or "Baseline")
                self.log("\n" + "=" * 72)
                self.log(report)
                self.log("\n[fertig] Export in: %s" % used_out)
                for f in ["split_assignment_%s.json", "split_assignment_%s.csv",
                          "split_precheck_report_%s.txt", "split_ecdf_%s.png",
                          "split_coverage_%s.png", "split_by_type_%s.png",
                          "negative_split_%s.json (falls Negative vorhanden)",
                          "negative_split_%s.csv (falls Negative vorhanden)"]:
                    self.log("        " + f % sfx)
            except Exception:
                self.log("[error]\n" + traceback.format_exc())
            finally:
                self.run_btn.config(state="normal")

        threading.Thread(target=worker, daemon=True).start()


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
