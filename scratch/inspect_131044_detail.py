"""Detailed inspection of train_data_20260820_131044.jpg (44.jpg)."""
import sys, os
sys.path.insert(0, '.')
import cv2
import numpy as np
import scanner

img_path = "training_data/train_data_20260820_131044.jpg"
img = cv2.imread(img_path)
print(f"Loaded {img_path}: shape {img.shape if img is not None else None}")

# 1. YOLO detections
from ultralytics import YOLO

model_path = os.path.join("runs", "detect", "training_runs_v2", "horde_2class", "weights", "best.pt")
yolo_res = YOLO(model_path)(img, verbose=False)
boxes = yolo_res[0].boxes
print(f"YOLO detections: {len(boxes)}")
for b in boxes:
    cls_id = int(b.cls[0])
    conf = float(b.conf[0])
    xyxy = map(int, b.xyxy[0])
    print(f"  Class {cls_id} ({'dmx' if cls_id==0 else 'text'}): conf={conf:.3f}, box={list(xyxy)}")

# 2. Test text crops with various preprocessings
for b in boxes:
    if int(b.cls[0]) == 1: # text
        box = tuple(map(int, b.xyxy[0]))
        for pad in [30, 50, 80]:
            crop = scanner.deskew_crop(img, box, padding=pad)
            cv2.imwrite(f"scratch/crop_131044_pad{pad}.png", crop)
            print(f"\n--- Crop pad={pad} shape={crop.shape} ---")
            
            # Try OCR on crop
            ocr_res = scanner._read_ocr_with_status(crop)
            print(f"  OCR on crop: {ocr_res}")
            
            # Try PACC on crop
            pacc_t, pacc_c = scanner._predict_pacc(crop)
            print(f"  PACC on crop: text={pacc_t}, conf={pacc_c:.3f}")

# 3. Test DataMatrix on DataMatrix crops
for b in boxes:
    if int(b.cls[0]) == 0: # dmx
        box = tuple(map(int, b.xyxy[0]))
        for pad in [20, 40, 60]:
            crop = scanner.deskew_crop(img, box, padding=pad)
            cv2.imwrite(f"scratch/crop_dmx_131044_pad{pad}.png", crop)
            print(f"\n--- DMX Crop pad={pad} shape={crop.shape} ---")
            dmx_res = scanner._scan_datamatrix_pipeline(crop)
            print(f"  DMX result: {dmx_res}")
