import os
import sys
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("instant_renumber")

APP_DIR = r"c:\Users\kremidas\Documents\DataDetector"
TRAINING_DIR = os.path.join(APP_DIR, "training_data")
GT_PATH = os.path.join(APP_DIR, "ground_truth.json")


def main():
    gt_map = {}
    if os.path.exists(GT_PATH):
        with open(GT_PATH, "r", encoding="utf-8") as f:
            gt_map = json.load(f)

    # Separate already numbered images (001.jpg .. 130.jpg) and un-numbered images
    numbered_files = []
    unnumbered_files = []

    for fn in os.listdir(TRAINING_DIR):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        base = os.path.splitext(fn)[0]
        if base.isdigit():
            numbered_files.append(fn)
        else:
            unnumbered_files.append(fn)

    numbered_files.sort()
    unnumbered_files.sort()

    start_idx = len(numbered_files) + 1
    logger.info(f"Existing numbered images: {len(numbered_files)} (up to {len(numbered_files):03d}.jpg)")
    logger.info(f"Un-numbered new images: {len(unnumbered_files)}")

    for idx, old_fn in enumerate(unnumbered_files, start_idx):
        new_fn = f"{idx:03d}.jpg"
        old_path = os.path.join(TRAINING_DIR, old_fn)
        new_path = os.path.join(TRAINING_DIR, new_fn)

        os.rename(old_path, new_path)

        if new_fn not in gt_map:
            gt_map[new_fn] = "?"

        logger.info(f"Instantly Renamed: {old_fn} -> {new_fn}")

    # Write clean ground_truth.json sorted by filename
    sorted_gt = {k: gt_map.get(k, "?") for k in sorted(gt_map.keys())}
    with open(GT_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted_gt, f, indent=2, ensure_ascii=False)

    logger.info(f"Updated ground_truth.json saved with {len(sorted_gt)} entries.")

    # Recreate split folders in training_data_splits
    sys.path.insert(0, APP_DIR)
    import run_batch_benchmarks
    folder_paths = run_batch_benchmarks.prepare_split_folders(batch_size=30)
    logger.info(f"Split folders updated: {len(folder_paths)} split folders created.")


if __name__ == "__main__":
    main()
