"""
prepare_dataset_v2.py — 2-Klassen-YOLO-Datensatz (datamatrix + text) in dataset_v2/.

Quellen:
1. Handgelabelte Bilder aus dem Label-Studio-Export (temp_ls_export/). Fehlt der Export, bleiben die
   handgelabelten Bilder des bestehenden dataset_v2 unverändert erhalten.
2. training_data/ und auto_training_data/images/ mit automatischen Labels: DataMatrix-Ecken über zxing-cpp,
   sonst über den Modul-Decoder. Bilder ohne gefundenen Code werden ausgelassen – keine Platzhalter-Boxen mehr
   (die hatten YOLO bei schweren Bildern übergroße Boxen in der Bildmitte beigebracht).

field_data/ bleibt als unabhängiger Testsatz draußen. Bereits vorhandene Bilder behalten ihre Train/Val-Zuordnung.
"""

import glob
import hashlib
import logging
import os
import shutil

import cv2
import numpy as np
import zxingcpp

from scanner.dmx_module_reader import read_dmx_modules

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("PrepareDatasetV2")
logging.getLogger("scanner").setLevel(logging.WARNING)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_LS_DIR = os.path.join(PROJECT_DIR, "temp_ls_export")
TRAIN_DATA_DIR = os.path.join(PROJECT_DIR, "training_data")
AUTO_TRAIN_IMAGES_DIR = os.path.join(PROJECT_DIR, "auto_training_data", "images")
FIELD_DATA_DIR = os.path.join(PROJECT_DIR, "field_data")
DATASET_V2_DIR = os.path.join(PROJECT_DIR, "dataset_v2")
VAL_PERCENT = 15
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def boxes_from_corners(corners, w_img: int, h_img: int):
    """
    Normierte YOLO-Boxen aus den DataMatrix-Ecken: DMX mit 10 % Rand; die Klarschrift beginnt direkt unter
    dem Code, ist ~0,5 Codehöhen hoch, so breit wie der Code und leicht nach rechts versetzt (30 Handlabels).
    """
    xs = [float(p[0]) for p in corners]
    ys = [float(p[1]) for p in corners]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    cw, ch = x2 - x1, y2 - y1
    pad = max(10.0, max(cw, ch) * 0.10)
    dx1, dy1 = max(0.0, x1 - pad), max(0.0, y1 - pad)
    dx2, dy2 = min(float(w_img), x2 + pad), min(float(h_img), y2 + pad)

    tcx = (x1 + x2) / 2 + 0.06 * cw
    tx1, tx2 = max(0.0, tcx - 0.51 * cw), min(float(w_img), tcx + 0.51 * cw)
    ty1 = min(float(h_img) - 1, max(0.0, y2 - 0.03 * ch))
    ty2 = min(float(h_img), ty1 + 0.55 * ch)

    dmx_norm = ((dx1 + dx2) / 2 / w_img, (dy1 + dy2) / 2 / h_img, (dx2 - dx1) / w_img, (dy2 - dy1) / h_img)
    txt_norm = ((tx1 + tx2) / 2 / w_img, (ty1 + ty2) / 2 / h_img, (tx2 - tx1) / w_img, (ty2 - ty1) / h_img)
    return dmx_norm, txt_norm


def detect_label_boxes(image: np.ndarray):
    """DataMatrix-Ecken über zxing-cpp, sonst Modul-Decoder → (DMX-Box, Text-Box, Quelle) oder None."""
    h_img, w_img = image.shape[:2]
    zx_res = zxingcpp.read_barcode(image)
    if zx_res and zx_res.position:
        pos = zx_res.position
        corners = [(p.x, p.y) for p in (pos.top_left, pos.top_right, pos.bottom_right, pos.bottom_left)]
        return (*boxes_from_corners(corners, w_img, h_img), "zxing")
    module = read_dmx_modules(image)
    if module["tier"] is not None and module["quad"] is not None:
        return (*boxes_from_corners(module["quad"], w_img, h_img), f"modul_{module['tier']}")
    return None


def _label_text(dmx_box, text_box) -> str:
    return (f"0 {dmx_box[0]:.6f} {dmx_box[1]:.6f} {dmx_box[2]:.6f} {dmx_box[3]:.6f}\n"
            f"1 {text_box[0]:.6f} {text_box[1]:.6f} {text_box[2]:.6f} {text_box[3]:.6f}")


def _hash_split(name: str) -> str:
    return "val" if int(hashlib.md5(name.encode("utf-8")).hexdigest(), 16) % 100 < VAL_PERCENT else "train"


def _images_in(directory: str) -> list[str]:
    if not os.path.isdir(directory):
        return []
    return sorted(p for p in glob.glob(os.path.join(directory, "*")) if p.lower().endswith(IMAGE_EXTENSIONS))


def _existing_items() -> dict[str, tuple[str, str, str]]:
    """Bestehender Datensatz: Dateiname → (Split, Bildpfad, Label-Inhalt)."""
    items = {}
    for split in ("train", "val"):
        for img_path in _images_in(os.path.join(DATASET_V2_DIR, "images", split)):
            name = os.path.basename(img_path)
            lbl_path = os.path.join(DATASET_V2_DIR, "labels", split, os.path.splitext(name)[0] + ".txt")
            label = open(lbl_path, encoding="utf-8").read().strip() if os.path.exists(lbl_path) else ""
            items[name] = (split, img_path, label)
    return items


def _hand_labeled_items(existing: dict) -> list[tuple[str, str, str, str]]:
    """Label-Studio-Bilder → [(Dateiname, Split, Bildpfad, Label-Inhalt)]."""
    ls_images = os.path.join(TEMP_LS_DIR, "images")
    ls_labels = os.path.join(TEMP_LS_DIR, "labels")
    items = []
    if os.path.isdir(ls_images) and os.path.isdir(ls_labels):
        for img_path in _images_in(ls_images):
            name = os.path.basename(img_path)
            lbl_path = os.path.join(ls_labels, os.path.splitext(name)[0] + ".txt")
            if os.path.exists(lbl_path):
                label = open(lbl_path, encoding="utf-8").read().strip()
                if label:
                    split = existing[name][0] if name in existing else _hash_split(name)
                    items.append((name, split, img_path, label))
        logger.info(f"{len(items)} handgelabelte Bilder aus dem Label-Studio-Export.")
    else:
        # Label-Studio-Dateinamen tragen ein Hash-Präfix ("110cc7af-train_data_...jpg")
        items = [(name, split, path, label) for name, (split, path, label) in existing.items() if "-" in name and label]
        logger.info(f"Kein Label-Studio-Export gefunden – {len(items)} handgelabelte Bilder aus dataset_v2 übernommen.")
    return items


def prepare():
    logger.info("Vorbereitung des 2-Klassen-Datensatzes in dataset_v2 ...")
    existing = _existing_items()
    items = _hand_labeled_items(existing)
    hand_originals = {name.split("-", 1)[-1] for name, *_ in items} | {name for name, *_ in items}
    field_names = {os.path.basename(p) for p in _images_in(FIELD_DATA_DIR)}

    sources = {"zxing": 0}
    skipped = []
    for img_path in _images_in(TRAIN_DATA_DIR) + _images_in(AUTO_TRAIN_IMAGES_DIR):
        name = os.path.basename(img_path)
        if name in hand_originals or name in field_names:
            continue
        image = cv2.imread(img_path)
        if image is None:
            skipped.append(name)
            continue
        detected = detect_label_boxes(image)
        if detected is None:
            skipped.append(name)
            continue
        dmx_box, text_box, source = detected
        sources[source] = sources.get(source, 0) + 1
        split = existing[name][0] if name in existing else _hash_split(name)
        items.append((name, split, img_path, _label_text(dmx_box, text_box)))
    logger.info(f"Automatische Labels: {sources}; ohne gefundenen Code ausgelassen: {len(skipped)} {skipped}")

    # In einen Staging-Ordner schreiben und dann austauschen (handgelabelte Bilder liegen im alten Ordner)
    staging = DATASET_V2_DIR + ".staging"
    shutil.rmtree(staging, ignore_errors=True)
    for split in ("train", "val"):
        os.makedirs(os.path.join(staging, "images", split))
        os.makedirs(os.path.join(staging, "labels", split))
    for name, split, img_path, label in items:
        shutil.copy2(img_path, os.path.join(staging, "images", split, name))
        with open(os.path.join(staging, "labels", split, os.path.splitext(name)[0] + ".txt"), "w", encoding="utf-8") as f:
            f.write(label + "\n")
    for sub in ("images", "labels"):
        shutil.rmtree(os.path.join(DATASET_V2_DIR, sub), ignore_errors=True)
        shutil.move(os.path.join(staging, sub), os.path.join(DATASET_V2_DIR, sub))
    shutil.rmtree(staging, ignore_errors=True)

    with open(os.path.join(DATASET_V2_DIR, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {DATASET_V2_DIR.replace(os.sep, '/')}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write("names:\n  0: datamatrix\n  1: text\n")

    n_val = sum(1 for _, split, _, _ in items if split == "val")
    logger.info(f"Datensatz fertig: {len(items)} Bilder (Train {len(items) - n_val}, Val {n_val}).")


if __name__ == "__main__":
    prepare()
