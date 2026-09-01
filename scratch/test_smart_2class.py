import os
import sys
import cv2
import json

sys.path.insert(0, ".")
import scanner
from ultralytics import YOLO

MODEL_PATH = os.path.join("runs", "detect", "training_runs_v2", "horde_2class", "weights", "best.pt")
model = YOLO(MODEL_PATH)

gt_path = "ground_truth_cleanroom.json"
with open(gt_path, "r", encoding="utf-8") as f:
    gt = json.load(f)

print("Starting test with lowered YOLO conf threshold (0.15)...")

correct = 0
total = 0
for img_name, expected in gt.items():
    img_path = os.path.join("training_data", img_name)
    if not os.path.exists(img_path):
        img_path = os.path.join("dataset_v2", "images", "train", img_name)
    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    total += 1
    detections = []
    yolo_res = model(img, verbose=False)
    if yolo_res and len(yolo_res[0].boxes) > 0:
        for box in yolo_res[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if conf > 0.15:  # lowered threshold
                detections.append({"cls": cls_id, "box": (x1, y1, x2, y2), "conf": conf})

    res = scanner.scan_2class(img, detections) if detections else scanner.scan(img)
    code = res.get("result") if res.get("success") else "FEHLER"
    
    if code == expected:
        correct += 1
    else:
        print(f"FAILED {img_name}: expected {expected}, got {code} (method={res.get('method')})")

print(f"\nTOTAL: {correct}/{total} ({correct/total*100:.1f}%)")
