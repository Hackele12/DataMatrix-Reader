"""Inspect DMX inner grid matching for 131044 across reference codes."""
import sys, os
sys.path.insert(0, '.')
import cv2
import numpy as np
import scanner

img_path = "training_data/train_data_20260820_131044.jpg"
img = cv2.imread(img_path)

# Extract DMX crop
yolo_res = scanner._get_yolo_model()(img, verbose=False) if hasattr(scanner, '_get_yolo_model') else None
# Get DMX box
box = (676, 400, 854, 609)
crop = scanner.deskew_crop(img, box, padding=30)

# Run _scan_datamatrix_pipeline with debug
# Let's inspect the observed grid
gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
# Let's see how _scan_datamatrix_pipeline extracts grid
res = scanner._scan_datamatrix_pipeline(crop)
print("DMX result:", res)

if res and "observed_grid" in res and res["observed_grid"] is not None:
    obs = res["observed_grid"]
    print("Observed grid shape:", obs.shape)
    
    # Compare observed grid against ALL reference codes
    matches = []
    # Load all ref grids from GT or generate for known codes
    import json
    with open("ground_truth_cleanroom.json", "r") as f:
        gt = json.load(f)
    
    all_codes = sorted(list(set(gt.values())))
    for code in all_codes:
        ref = scanner._get_cached_reference_grid(code)
        if ref is not None and ref.shape == obs.shape:
            score = np.sum(obs == ref)
            matches.append((code, score))
    
    matches.sort(key=lambda x: x[1], reverse=True)
    print("\nTop 15 reference grid matches for 131044:")
    for code, score in matches[:15]:
        print(f"  Code: {code:6s} | Score: {score}/64 ({score/64*100:.1f}%)")
