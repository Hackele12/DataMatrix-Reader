import os
import json
import glob
import cv2

APP_DIR = r"c:\Users\kremidas\Documents\DataDetector"
GT_PATH = os.path.join(APP_DIR, "ground_truth.json")
HORDE_DIR = os.path.join(APP_DIR, "horden_db")
LOG_PATH = r"C:\Users\kremidas\.gemini\antigravity-ide\brain\bf448dd0-c328-4d2a-8261-302090026ebd\.system_generated\tasks\task-1285.log"

with open(GT_PATH, "r", encoding="utf-8") as f:
    gt_map = json.load(f)

print(f"Total entries in ground_truth.json: {len(gt_map)}")

# Extract all mismatch lines from task-1285.log
mismatches = []
success_matches = []
failures = []

if os.path.exists(LOG_PATH):
    with open(LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "Match=False" in line:
                mismatches.append(line.strip())
            elif "Match=True" in line:
                success_matches.append(line.strip())
            elif "Res='Kein Code erkannt.'" in line:
                failures.append(line.strip())

print(f"\n--- LOG ANALYSIS SUMMARY ---")
print(f"Total Match=True (Correct GT): {len(success_matches)}")
print(f"Total Match=False (GT Mismatch): {len(mismatches)}")
print(f"Total Failures (No Code Detected): {len(failures)}")

print("\n--- SAMPLE MISMATCHES (GT vs Scanner Result) ---")
for line in mismatches[:25]:
    # Example line: [part_01 1/30] 001.jpg -> Res='Kein Code erkannt.' GT='B182' Match=False ...
    print(line)
