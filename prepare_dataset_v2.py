"""
prepare_dataset_v2.py — Fast version without slow EasyOCR.

Kombiniert:
1. Handgelabelte Bilder aus dem Label-Studio Export (project-7-at-2026-09-02-14-07-c6b5bb49.zip)
2. Schnelles ZXing-basiertes Auto-Labeling für verbleibende training_data/ Bilder
"""

import os
import sys
import glob
import shutil
import logging
import random
import cv2
import numpy as np
import zxingcpp

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("PrepareDatasetV2Fast")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_LS_DIR = os.path.join(PROJECT_DIR, "temp_ls_export")
TRAIN_DATA_DIR = os.path.join(PROJECT_DIR, "training_data")
DATASET_V2_DIR = os.path.join(PROJECT_DIR, "dataset_v2")

def fast_detect_boxes(image: np.ndarray):
    """Schnelle Box-Erkennung via ZXing und relativer Geometrie (< 5ms pro Bild)."""
    h_img, w_img = image.shape[:2]
    zx_res = zxingcpp.read_barcode(image)
    
    if zx_res and zx_res.position:
        pos = zx_res.position
        pts = [(p.x, p.y) for p in [pos.top_left, pos.top_right, pos.bottom_right, pos.bottom_left]]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        
        pad = max(15, int(max(max(xs)-min(xs), max(ys)-min(ys)) * 0.15))
        dx1, dy1 = max(0, min(xs) - pad), max(0, min(ys) - pad)
        dx2, dy2 = min(w_img, max(xs) + pad), min(h_img, max(ys) + pad)
        
        dw = dx2 - dx1
        dh = dy2 - dy1
        
        # Text-Box direkt unter dem Barcode schätzen
        th = int(dh * 0.38)
        tw = int(dw * 1.05)
        tcx = (dx1 + dx2) // 2
        ty1 = min(h_img - th, dy2 + int(dh * 0.05))
        ty2 = min(h_img, ty1 + th)
        tx1 = max(0, tcx - tw // 2)
        tx2 = min(w_img, tcx + tw // 2)
        
        dmx_norm = (((dx1 + dx2) / 2.0) / w_img, ((dy1 + dy2) / 2.0) / h_img, dw / w_img, dh / h_img)
        txt_norm = (((tx1 + tx2) / 2.0) / w_img, ((ty1 + ty2) / 2.0) / h_img, tw / w_img, (ty2 - ty1) / h_img)
        return dmx_norm, txt_norm
    else:
        # Fallback: Zentrierte Standard-Boxen
        cx, cy = 0.5, 0.5
        return (cx, cy - 0.1, 0.4, 0.4), (cx, cy + 0.2, 0.4, 0.15)


def prepare():
    logger.info("Vorbereitung des 2-Klassen Datasets in dataset_v2...")
    
    # 1. Ziel-Ordner säubern / neu anlegen
    for split in ["train", "val"]:
        os.makedirs(os.path.join(DATASET_V2_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DATASET_V2_DIR, "labels", split), exist_ok=True)

    # 2. data.yaml schreiben
    yaml_path = os.path.join(DATASET_V2_DIR, "data.yaml")
    ds_dir_clean = DATASET_V2_DIR.replace("\\", "/")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {ds_dir_clean}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write("names:\n  0: datamatrix\n  1: text\n")

    items = []

    # A) Label Studio Export (Höchste Prio)
    ls_images_dir = os.path.join(TEMP_LS_DIR, "images")
    ls_labels_dir = os.path.join(TEMP_LS_DIR, "labels")
    
    hand_labeled_names = set()
    if os.path.exists(ls_images_dir) and os.path.exists(ls_labels_dir):
        for img_fn in os.listdir(ls_images_dir):
            if not img_fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            base = os.path.splitext(img_fn)[0]
            lbl_fn = base + ".txt"
            img_path = os.path.join(ls_images_dir, img_fn)
            lbl_path = os.path.join(ls_labels_dir, lbl_fn)
            if os.path.exists(lbl_path):
                with open(lbl_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if content:
                    items.append((img_path, content))
                    # Extrahiere Original-Dateiname
                    orig_name = img_fn.split("-", 1)[-1] if "-" in img_fn else img_fn
                    hand_labeled_names.add(orig_name)
                    hand_labeled_names.add(img_fn)

    logger.info(f"Loaded {len(hand_labeled_names)} hand-labeled items from Label Studio.")

    # B) Restliche 172 echten Bilder aus training_data/
    real_images = sorted(glob.glob(os.path.join(TRAIN_DATA_DIR, "*.jpg")) + glob.glob(os.path.join(TRAIN_DATA_DIR, "*.png")))
    auto_added = 0
    for img_path in real_images:
        fn = os.path.basename(img_path)
        if fn in hand_labeled_names:
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        dmx_box, text_box = fast_detect_boxes(img)
        content = f"0 {dmx_box[0]:.6f} {dmx_box[1]:.6f} {dmx_box[2]:.6f} {dmx_box[3]:.6f}\n" \
                  f"1 {text_box[0]:.6f} {text_box[1]:.6f} {text_box[2]:.6f} {text_box[3]:.6f}"
        items.append((img_path, content))
        auto_added += 1

    logger.info(f"Added {auto_added} additional images with fast ZXing auto-labeling.")
    logger.info(f"Total dataset size: {len(items)} images.")

    # C) Split in Train / Val (85% / 15%)
    random.seed(42)
    random.shuffle(items)
    
    val_size = max(1, int(len(items) * 0.15))
    val_items = items[:val_size]
    train_items = items[val_size:]

    def write_split(item_list, split_name):
        for img_path, label_content in item_list:
            fn = os.path.basename(img_path)
            base = os.path.splitext(fn)[0]
            dest_img = os.path.join(DATASET_V2_DIR, "images", split_name, fn)
            dest_lbl = os.path.join(DATASET_V2_DIR, "labels", split_name, base + ".txt")
            shutil.copy2(img_path, dest_img)
            with open(dest_lbl, "w", encoding="utf-8") as f:
                f.write(label_content + "\n")

    write_split(train_items, "train")
    write_split(val_items, "val")

    logger.info(f"Dataset ready! Train: {len(train_items)}, Val: {len(val_items)}")

if __name__ == "__main__":
    prepare()
