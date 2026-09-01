"""Schnelltest der 5 fehlenden Bilder nach Gamma-Fix."""
import os, sys, cv2, json
sys.stdout = open("test_5_output.txt", "w", encoding="utf-8")

import scanner
from ultralytics import YOLO

FAILING = {
    "SCN-20260820-130950-0021.jpg": "W031",
    "SCN-20260820-131030-0022.jpg": "W031",
    "SCN-20260820-131052-0023.jpg": "W031",
    "SCN-20260824-130855-0011.jpg": "W003",
    "train_data_20260820_131044.jpg": "W031",
}

model_path = os.path.join("runs", "detect", "training_runs_v2", "horde_2class", "weights", "best.pt")
model = YOLO(model_path)

for img_name, expected in FAILING.items():
    img = cv2.imread(os.path.join("training_data", img_name))
    if img is None:
        print(f"SKIP: {img_name}")
        continue

    # YOLO
    yolo_res = model(img, verbose=False)
    dets = []
    for box in yolo_res[0].boxes:
        c = int(box.cls[0]); cf = float(box.conf[0])
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        if cf > 0.3:
            dets.append({"cls":c,"box":(x1,y1,x2,y2),"conf":cf})

    res = scanner.scan_2class(img, dets) if dets else scanner.scan(img)
    code = res.get("result") if res.get("success") else "FEHLER"
    ok = "OK" if code == expected else "FAIL"
    print(f"{img_name:40s} | Soll={expected} | Erkannt={code:8s} | {ok} | method={res.get('method')} | conf={res.get('confidence'):.2f}")

sys.stdout.close()
sys.stdout = sys.__stdout__
print("Test fertig -> test_5_output.txt")
