import os
import sys
import glob
import cv2
import json
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("renumber_new")

APP_DIR = r"c:\Users\kremidas\Documents\DataDetector"
sys.path.insert(0, APP_DIR)

import benchmark_suite
import scanner
from ultralytics import YOLO

TRAINING_DIR = os.path.join(APP_DIR, "training_data")
GT_PATH = os.path.join(APP_DIR, "ground_truth.json")


def main():
    # Load existing ground_truth.json
    gt_map = {}
    if os.path.exists(GT_PATH):
        with open(GT_PATH, "r", encoding="utf-8") as f:
            gt_map = json.load(f)

    # Find existing numbered files (001.jpg .. 127.jpg)
    existing_numbered = []
    new_files = []

    for fname in os.listdir(TRAINING_DIR):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        base = os.path.splitext(fname)[0]
        if base.isdigit():
            existing_numbered.append(fname)
        else:
            new_files.append(fname)

    existing_numbered.sort()
    new_files.sort()

    start_num = len(existing_numbered) + 1
    logger.info(f"Found {len(existing_numbered)} existing numbered images (up to {len(existing_numbered):03d}.jpg).")
    logger.info(f"Found {len(new_files)} new images to renumber (starting from {start_num:03d}.jpg).")

    if not new_files:
        logger.info("No new un-numbered images found!")
        return

    # Load YOLO model for GT detection on new images
    model_path = os.path.join(APP_DIR, "runs", "detect", "training_runs", "horde_model", "weights", "best.pt")
    model = YOLO(model_path)

    renamed_records = []
    for idx, old_fname in enumerate(new_files, start_num):
        new_fname = f"{idx:03d}.jpg"
        old_path = os.path.join(TRAINING_DIR, old_fname)
        new_path = os.path.join(TRAINING_DIR, new_fname)

        os.rename(old_path, new_path)

        # Run scanner to auto-detect ground truth label
        image = cv2.imread(new_path)
        code = "?"
        if image is not None:
            try:
                cropped, yolo_conf = benchmark_suite._detect_and_crop(model, image)
                scan_res = scanner.scan(cropped)
                if scan_res.get("success") and scan_res.get("result"):
                    code = scan_res["result"]
            except Exception as e:
                logger.warning(f"Error scanning {new_fname}: {e}")

        gt_map[new_fname] = code
        renamed_records.append((old_fname, new_fname, code))
        logger.info(f"Renamed: {old_fname:<35} -> {new_fname:<8} (Auto-GT: {code})")

    # Save updated ground_truth.json
    with open(GT_PATH, "w", encoding="utf-8") as f:
        json.dump(gt_map, f, indent=2, ensure_ascii=False)

    logger.info(f"\n==========================================")
    logger.info(f"Renumbered {len(new_files)} new images: {start_num:03d}.jpg to {start_num + len(new_files) - 1:03d}.jpg")
    logger.info(f"Total dataset size now: {len(gt_map)} images.")
    logger.info(f"Updated ground_truth.json saved to: {GT_PATH}")
    logger.info(f"==========================================")


if __name__ == "__main__":
    main()
