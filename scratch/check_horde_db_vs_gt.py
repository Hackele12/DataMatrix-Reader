import os
import json
import glob
import cv2

APP_DIR = r"c:\Users\kremidas\Documents\DataDetector"
HORDE_DIR = os.path.join(APP_DIR, "horden_db")
GT_PATH = os.path.join(APP_DIR, "ground_truth.json")

with open(GT_PATH, "r", encoding="utf-8") as f:
    gt_map = json.load(f)

horde_files = glob.glob(os.path.join(HORDE_DIR, "*.jpg"))
print(f"Total files in horden_db: {len(horde_files)}")

# Check horde_db filenames vs GT
# Notice: horde_db filenames are saved as <CODE>.jpg (e.g. W001.jpg, B182.jpg)
# But ground_truth.json maps image filenames (e.g. "001.jpg": "B182") to GT codes!

print("\n--- AUDITING HORDEN_DB CONTENTS ---")
for h_file in horde_files:
    code_name = os.path.basename(h_file).replace(".jpg", "")
    print(f"Horde DB File: {os.path.basename(h_file)} -> Code: {code_name}")
