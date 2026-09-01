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

files_to_check = [
    "SCN-20260820-130950-0021.jpg",
    "SCN-20260820-131030-0022.jpg",
    "SCN-20260820-131052-0023.jpg",
    "SCN-20260820-131103-0024.jpg",
    "train_data_20260820_130933.jpg",
    "train_data_20260820_131044.jpg"
]

print("--- Testing DMX Multi-Pass Enhancement on W031 images ---")
for fname in files_to_check:
    img_path = os.path.join("training_data", fname)
    if not os.path.exists(img_path):
        img_path = os.path.join("dataset_v2", "images", "train", fname)
    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)
    yolo_res = model(img, verbose=False)
    detections = []
    if yolo_res and len(yolo_res[0].boxes) > 0:
        for box in yolo_res[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = tuple(map(int, box.xyxy[0]))
            if conf > 0.15:
                detections.append({"cls": cls_id, "box": xyxy, "conf": conf})

    dmx_det = next((d for d in detections if d["cls"] == 0), None)
    if dmx_det:
        crop = scanner.deskew_crop(img, dmx_det["box"], padding=50)
        res_raw = scanner.scan_datamatrix(crop)
        print(f"{fname}: dmx_det conf={dmx_det['conf']:.2f}, raw scan_datamatrix result: {res_raw.get('text')}")
    else:
        print(f"{fname}: No DMX detection!")
