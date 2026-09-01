import os
import glob
import json

APP_DIR = r"c:\Users\kremidas\Documents\DataDetector"
HORDE_DIR = os.path.join(APP_DIR, "horden_db")
GT_PATH = os.path.join(APP_DIR, "ground_truth.json")

with open(GT_PATH, "r", encoding="utf-8") as f:
    gt_map = json.load(f)

# Get all valid Ground Truth codes
valid_gt_codes = set(gt_map.values())

horde_files = glob.glob(os.path.join(HORDE_DIR, "*.jpg"))
print(f"Total files in horden_db before audit: {len(horde_files)}")
print(f"Total unique valid GT codes: {len(valid_gt_codes)}")

# Audit each file in horden_db: does the code name exist in GT?
invalid_count = 0
for h_file in horde_files:
    code = os.path.basename(h_file).replace(".jpg", "")
    if code not in valid_gt_codes:
        print(f"Removing invalid/unverified horde_db file: {os.path.basename(h_file)} (Code '{code}' not in GT)")
        os.remove(h_file)
        invalid_count += 1

print(f"\nAudit complete. Removed {invalid_count} invalid files from horden_db.")
print(f"Remaining clean files in horden_db: {len(glob.glob(os.path.join(HORDE_DIR, '*.jpg')))}")
