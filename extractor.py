"""
Quantity extraction from architectural plans.

Pipeline
  1. Load PDF (PyMuPDF) or image (OpenCV)
  2. Auto-rotate: try 0/90/180/270 and keep the orientation where OCR finds the most
     dimension strings ("8.50 X 8.92M", "75.82 SQ.M")
  3. OCR (Tesseract LSTM) -> words -> text lines
  4. DBSCAN clusters text lines into room labels (name + dimensions + stated area)
  5. OpenCV measures the enclosed pixel area around each label; the median of
     sqrt(stated_area / pixel_area) gives the drawing scale (m per pixel) and a
     cross-check of every room
  6. Area x height -> volume, plus slab concrete volume
"""
import base64
import re
import statistics

import cv2
import fitz  # PyMuPDF
import numpy as np
import pytesseract
from pytesseract import Output
from sklearn.cluster import DBSCAN

NUM = r"(\d+(?:[.,]\d+)?)"
DIM_RE = re.compile(NUM + r"\s*[xX×*]\s*" + NUM + r"\s*[mM]?\b")
AREA_RE = re.compile(NUM + r"\s*S[QO]\.?\s*M", re.I)


def _f(s):
    return float(s.replace(",", "."))


# ----------------------------------------------------------------------------- loading
def load_image(path, dpi=300):
    if path.lower().endswith(".pdf"):
        doc = fitz.open(path)
        page = doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Could not read the image file.")
    return img


def rotate(img, k):
    return {0: img,
            90: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
            180: cv2.rotate(img, cv2.ROTATE_180),
            270: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)}[k]


def auto_rotate(img):
    """Return (rotated_image, degrees) using OCR hit-count on a downscaled copy."""
    h, w = img.shape[:2]
    scale = 1800 / max(h, w)
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img
    best, best_score = 0, -1
    for deg in (0, 90, 180, 270):
        g = cv2.cvtColor(rotate(small, deg), cv2.COLOR_BGR2GRAY)
        txt = pytesseract.image_to_string(g, config="--psm 11")
        score = len(DIM_RE.findall(txt)) * 2 + len(AREA_RE.findall(txt)) * 2 + len(re.findall(r"\b[A-Z]{4,}\b", txt)) * 0.1
        if score > best_score:
            best, best_score = deg, score
    return rotate(img, best), best


# ----------------------------------------------------------------------------- OCR
def ocr_words(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    d = pytesseract.image_to_data(g, config="--psm 11 --oem 1", output_type=Output.DICT)
    words = []
    for i, t in enumerate(d["text"]):
        t = t.strip()
        if t and float(d["conf"][i]) >= 25:
            words.append(dict(text=t, x=d["left"][i], y=d["top"][i], w=d["width"][i], h=d["height"][i]))
    return words


def group_lines(words):
    """Merge words that sit on the same row into text lines."""
    lines = []
    for w in sorted(words, key=lambda k: (k["y"] + k["h"] / 2, k["x"])):
        cy = w["y"] + w["h"] / 2
        placed = False
        for ln in lines:
            if abs(ln["cy"] - cy) < 0.6 * max(ln["h"], w["h"]) and -0.3 * w["h"] <= w["x"] - ln["x2"] < 1.4 * w["h"]:
                ln["words"].append(w)
                ln["x2"] = w["x"] + w["w"]
                ln["x1"] = min(ln["x1"], w["x"])
                ln["y1"] = min(ln["y1"], w["y"])
                ln["y2"] = max(ln["y2"], w["y"] + w["h"])
                ln["h"] = max(ln["h"], w["h"])
                ln["cy"] = (ln["y1"] + ln["y2"]) / 2
                placed = True
                break
        if not placed:
            lines.append(dict(words=[w], x1=w["x"], x2=w["x"] + w["w"], y1=w["y"], y2=w["y"] + w["h"], h=w["h"], cy=cy))
    for ln in lines:
        ln["text"] = " ".join(w["text"] for w in sorted(ln["words"], key=lambda k: k["x"]))
        ln["cx"] = (ln["x1"] + ln["x2"]) / 2
    return lines


def cluster_labels(lines):
    """DBSCAN over line centres -> one cluster per drawing label."""
    if not lines:
        return []
    med_h = statistics.median(l["h"] for l in lines)
    X = np.array([[l["cx"], l["cy"]] for l in lines])
    # stretch x so only vertically-stacked, horizontally-aligned lines merge
    X[:, 0] *= 1.0
    labels = DBSCAN(eps=4.6 * med_h, min_samples=1).fit_predict(X)
    groups = {}
    for ln, lab in zip(lines, labels):
        groups.setdefault(lab, []).append(ln)
    return list(groups.values())


VOCAB = ["BOARD ROOM", "OFFICE ALL INCLUSIVE", "PRINCIPAL OFFICE", "CLASSROOM", "LAB", "LIBRARY",
         "STORE ROOM", "TOILET", "LIFT", "STAIRCASE", "KITCHEN", "RECEPTION", "CANTEEN", "SEMINAR HALL",
         "CONFERENCE ROOM", "BEDROOM", "LIVING ROOM", "DINING", "HALL", "LOBBY", "CABIN"]


def fix_name(name):
    """Snap OCR-damaged names (e.g. 'OARD ROOM') to a known architectural vocabulary."""
    import difflib
    base = re.sub(r"[-\s]*0*\d+$", "", name.upper()).strip()
    suffix = name[len(base):].strip() if name.upper().startswith(base) else ""
    hit = difflib.get_close_matches(base, VOCAB, n=1, cutoff=0.75)
    return (hit[0] + (" " + suffix if suffix else "")) if hit else name



def label_boxes(group):
    """Boxes (pixels) for each printed measurement and the room name inside one label."""
    out = []
    for ln in group:
        t = ln["text"]
        if AREA_RE.search(t):
            kind = "area"
        elif re.search(r"\d+[.,]\d+", t) or re.fullmatch(r"[\d.,xX×mM\s|]+", t):
            kind = "dim"
        else:
            kind = "name"
        out.append(dict(kind=kind, text=t, x1=ln["x1"], y1=ln["y1"], x2=ln["x2"], y2=ln["y2"]))
    return out


def parse_group(group):
    group = sorted(group, key=lambda l: (round(l["cy"] / (0.8 * l["h"])), l["cx"]))
    joined = " ".join(l["text"] for l in group)
    joined = re.sub(r"(\d)\s*[,]\s*(\d)", r"\1.\2", joined)
    area_m = AREA_RE.search(joined)
    area = _f(area_m.group(1)) if area_m else None
    text_wo_area = AREA_RE.sub(" ", joined)
    dim_m = DIM_RE.search(text_wo_area)
    dims = (_f(dim_m.group(1)), _f(dim_m.group(2))) if dim_m else None
    if dims is None and area is None:
        return None
    name_parts = []
    for ln in group:
        t = ln["text"]
        if re.search(r"\d+[.,]\d+", t) or AREA_RE.search(t) or re.fullmatch(r"[\dXx.,M ]+", t):
            continue
        name_parts.append(t)
    x1 = min(l["x1"] for l in group); x2 = max(l["x2"] for l in group)
    y1 = min(l["y1"] for l in group); y2 = max(l["y2"] for l in group)
    name = re.sub(r"[^A-Za-z0-9 \-/&.]", "", " ".join(name_parts)).strip(" -.")
    name = re.sub(r"\bLAB\s*-?\s*[O0]+1\b", "LAB-01", name)
    name = fix_name(name) or "Unnamed space"
    return dict(name=name.title() if name.isupper() else name, dims=dims, stated_area=area,
                bbox=(x1, y1, x2, y2), center=((x1 + x2) // 2, (y1 + y2) // 2))



KEYWORDS = re.compile(r"room|office|lab|class|store|toilet|hall|kitchen|lobby|cabin|library|canteen|reception", re.I)


def refine_group(img, group):
    """Re-OCR a cropped, upscaled region around a label whose numbers were not read cleanly."""
    x1 = min(l["x1"] for l in group); x2 = max(l["x2"] for l in group)
    y1 = min(l["y1"] for l in group); y2 = max(l["y2"] for l in group)
    H, W = img.shape[:2]
    crop = img[max(0, y1 - 170):min(H, y2 + 170), max(0, x1 - 140):min(W, x2 + 140)]
    crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    txt = pytesseract.image_to_string(g, config="--psm 6")
    txt = re.sub(r"\s+", " ", txt)
    am = AREA_RE.search(txt)
    dm = DIM_RE.search(AREA_RE.sub(" ", txt))
    return (_f(am.group(1)) if am else None,
            (_f(dm.group(1)), _f(dm.group(2))) if dm else None)

# ----------------------------------------------------------------------------- geometry
def measure_regions(img, words, rooms):
    """Pixel area of the enclosed free space around every label (text erased first)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ink = (g < 160).astype(np.uint8) * 255
    for w in words:
        pad = 3
        ink[max(0, w["y"] - pad):w["y"] + w["h"] + pad, max(0, w["x"] - pad):w["x"] + w["w"] + pad] = 0
    ink = cv2.dilate(ink, np.ones((5, 5), np.uint8))  # close small door gaps / hairline breaks
    free = cv2.bitwise_not(ink)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    limit = 0.25 * img.shape[0] * img.shape[1]
    for r in rooms:
        cx, cy = r["center"]
        lab = cc[min(cy, cc.shape[0] - 1), min(cx, cc.shape[1] - 1)]
        px = int(stats[lab, cv2.CC_STAT_AREA]) if lab > 0 else 0
        r["px_area"] = px if 0 < px < limit else None


def calibrate(rooms):
    """Median scale (m per pixel) from rooms whose stated area and pixel area are both known."""
    ratios = [np.sqrt(r["area_m2"] / r["px_area"]) for r in rooms if r.get("px_area") and r["area_m2"]]
    if len(ratios) < 3:
        return None
    med = float(np.median(ratios))
    keep = [x for x in ratios if abs(x - med) / med < 0.35]
    return float(np.median(keep)) if len(keep) >= 3 else med


# ----------------------------------------------------------------------------- main
def analyse(path, height=3.0, slab=0.15, floors=1):
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        raise RuntimeError("Tesseract OCR is not installed. Install it first (see README).")

    original = load_image(path)
    img, deg = auto_rotate(original)
    words = ocr_words(img)
    lines = group_lines(words)

    rooms = []
    for grp in cluster_labels(lines):
        r = parse_group(grp)
        text = " ".join(l["text"] for l in grp)
        if (r is None or r["dims"] is None) and (KEYWORDS.search(text) or re.search(r"\d+[.,]\d", text)):
            area2, dims2 = refine_group(img, grp)
            if r is None and (dims2 or area2):
                name = re.sub(r"[^A-Za-z0-9 \-/&.]", "", " ".join(
                    l["text"] for l in grp if not re.search(r"\d", l["text"]))).strip(" -.")
                x1 = min(l["x1"] for l in grp); x2 = max(l["x2"] for l in grp)
                y1 = min(l["y1"] for l in grp); y2 = max(l["y2"] for l in grp)
                r = dict(name=fix_name(name).title() or "Unnamed space", dims=dims2, stated_area=area2,
                         bbox=(x1, y1, x2, y2), center=((x1 + x2) // 2, (y1 + y2) // 2))
            elif r is not None:
                r["dims"] = dims2 or r["dims"]
                r["stated_area"] = r["stated_area"] or area2
        if r and r["dims"] is None and r["stated_area"] is None:
            r = None
        if r:
            r["boxes"] = label_boxes(grp)
            rooms.append(r)
    rooms.sort(key=lambda r: (r["center"][1], r["center"][0]))

    for i, r in enumerate(rooms, 1):
        r["id"] = i
        calc = r["dims"][0] * r["dims"][1] if r["dims"] else None
        st = r["stated_area"]
        r["calc_area"] = round(calc, 2) if calc else None
        r["area_m2"] = round(calc if calc else st, 2)
        if calc and st:
            diff = abs(calc - st) / st
            if diff <= 0.03:
                r["check"] = "ok"
            else:  # OCR probably misread a dimension: trust the printed area, flag for review
                r["check"] = "mismatch"
                r["area_m2"] = round(st, 2)
        else:
            r["check"] = "single source"

    measure_regions(img, words, rooms)
    scale = calibrate(rooms)
    for r in rooms:
        r["measured_area"] = round(r["px_area"] * scale ** 2, 1) if (scale and r.get("px_area")) else None
        r["volume_m3"] = round(r["area_m2"] * height, 2)
        r["slab_m3"] = round(r["area_m2"] * slab, 2)

    def enc(im, width=1900):
        k = width / max(im.shape[:2])
        if k < 1:
            im = cv2.resize(im, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return "data:image/jpeg;base64," + base64.b64encode(buf).decode()

    H, W = img.shape[:2]
    for r in rooms:
        r["boxes"] = [dict(kind=b["kind"], text=b["text"],
                           x=round((b["x1"] - 6) / W, 5), y=round((b["y1"] - 6) / H, 5),
                           w=round((b["x2"] - b["x1"] + 12) / W, 5), h=round((b["y2"] - b["y1"] + 12) / H, 5))
                      for b in r["boxes"]]
    preview, original_b64, aspect = enc(img), enc(original, 1100), W / H

    total_area = round(sum(r["area_m2"] for r in rooms), 2)
    for r in rooms:
        r.pop("bbox"); r.pop("center"); r.pop("px_area", None)
    return dict(
        rooms=rooms,
        rotation=deg,
        scale_m_per_px=round(scale, 5) if scale else None,
        totals=dict(
            area_m2=total_area,
            volume_m3=round(total_area * height, 2),
            slab_m3=round(total_area * slab, 2),
            floors=floors,
            volume_all_floors_m3=round(total_area * height * floors, 2),
            slab_all_floors_m3=round(total_area * slab * floors, 2),
        ),
        params=dict(height=height, slab=slab, floors=floors),
        preview=preview,
        original=original_b64,
        aspect=aspect,
    )
