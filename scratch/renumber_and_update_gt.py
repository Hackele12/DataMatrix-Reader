import os
import sys
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("renumber")

APP_DIR = r"c:\Users\kremidas\Documents\DataDetector"
TRAINING_DIR = os.path.join(APP_DIR, "training_data")
GT_PATH = os.path.join(APP_DIR, "ground_truth.json")


def main():
    if not os.path.exists(TRAINING_DIR):
        raise FileNotFoundError(f"Directory {TRAINING_DIR} not found.")

    # Load existing ground_truth.json
    old_gt_map = {}
    if os.path.exists(GT_PATH):
        with open(GT_PATH, "r", encoding="utf-8") as f:
            old_gt_map = json.load(f)
        logger.info(f"Loaded existing ground truth with {len(old_gt_map)} entries.")

    valid_exts = (".jpg", ".jpeg", ".png")
    old_files = sorted([f for f in os.listdir(TRAINING_DIR) if os.path.splitext(f)[1].lower() in valid_exts])
    total = len(old_files)

    logger.info(f"Found {total} images in {TRAINING_DIR} to renumber.")

    # Map old names to new names
    rename_mapping = []  # list of (old_name, temp_path, final_name, final_path)
    new_gt_map = {}

    # Step 1: Pass 1 - Rename to temporary names to avoid collision
    temp_records = []
    for idx, old_name in enumerate(old_files, 1):
        ext = os.path.splitext(old_name)[1].lower()
        new_name = f"{idx:03d}.jpg"

        old_path = os.path.join(TRAINING_DIR, old_name)
        temp_name = f"__temp_{idx:04d}{ext}"
        temp_path = os.path.join(TRAINING_DIR, temp_name)

        os.rename(old_path, temp_path)
        temp_records.append((temp_path, new_name, old_name))

    # Step 2: Pass 2 - Rename from temp to final 001.jpg ... 127.jpg
    for temp_path, new_name, old_name in temp_records:
        final_path = os.path.join(TRAINING_DIR, new_name)
        os.rename(temp_path, final_path)

        # Map GT entry
        if old_name in old_gt_map:
            new_gt_map[new_name] = old_gt_map[old_name]
        else:
            new_gt_map[new_name] = "?"

        logger.info(f"Renamed: {old_name:<12} -> {new_name:<10} (GT: {new_gt_map[new_name]})")

    # Save updated ground_truth.json
    with open(GT_PATH, "w", encoding="utf-8") as f:
        json.dump(new_gt_map, f, indent=2, ensure_ascii=False)

    logger.info(f"Successfully renumbered {total} images (001.jpg to {total:03d}.jpg).")
    logger.info(f"Updated ground_truth.json saved with {len(new_gt_map)} entries.")


if __name__ == "__main__":
    main()
