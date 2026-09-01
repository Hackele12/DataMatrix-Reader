import os
import sys
import glob
import subprocess

APP_DIR = r"c:\Users\kremidas\Documents\DataDetector"
PART_02_DIR = os.path.join(APP_DIR, "training_data_splits", "part_02")

images = sorted(glob.glob(os.path.join(PART_02_DIR, "*.jpg")))

print(f"Testing {len(images)} images in part_02 individually...")

for img_path in images:
    fname = os.path.basename(img_path)
    cmd = [
        sys.executable,
        "-c",
        f"""
import cv2, os, sys
sys.path.insert(0, r'{APP_DIR}')
import scanner
from ultralytics import YOLO

model_path = os.path.join(r'{APP_DIR}', 'runs', 'detect', 'training_runs', 'horde_model', 'weights', 'best.pt')
model = YOLO(model_path)
image = cv2.imread(r'{img_path}')
results = model.predict(image, conf=0.15, verbose=False)
boxes = results[0].boxes
if len(boxes) > 0:
    x1, y1, x2, y2 = map(int, boxes[0].xyxy[0])
    cropped = scanner.deskew_crop(image, (x1, y1, x2, y2), padding=60)
else:
    cropped = image
res = scanner.scan(cropped)
print('SUCCESS:', res.get('success'), 'CODE:', res.get('result'))
"""
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"FAILED on image {fname} with returncode {res.returncode}: {res.stderr[-200:]}")
    else:
        print(f"OK: {fname} -> {res.stdout.strip()}")
