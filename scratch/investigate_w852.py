import os
import sys
APP_DIR = r"c:\Users\kremidas\Documents\DataDetector"
sys.path.insert(0, APP_DIR)

import cv2
import zxingcpp
import ultralytics
import benchmark_suite
import scanner
model_path = os.path.join(APP_DIR, "runs", "detect", "training_runs", "horde_model", "weights", "best.pt")
model = ultralytics.YOLO(model_path)

test_imgs = ["026.jpg", "048.jpg", "049.jpg", "050.jpg", "051.jpg", "052.jpg", "095.jpg", "096.jpg", "097.jpg", "098.jpg", "099.jpg", "100.jpg"]

print("======================================================================")
print("              EXAKTE UNTERSUCHUNG VON W052 vs W852                    ")
print("======================================================================")

for fn in test_imgs:
    for sub in ["training_data", "training_data_splits/part_01", "training_data_splits/part_02", "training_data_splits/part_04"]:
        path = os.path.join(APP_DIR, sub, fn)
        if os.path.exists(path):
            img = cv2.imread(path)
            cropped, _ = benchmark_suite._detect_and_crop(model, img)
            
            # 1. Direct zxing-cpp
            zx_raw = zxingcpp.read_barcode(cropped, formats=zxingcpp.BarcodeFormat.DataMatrix)
            zx_text = zx_raw.text if (zx_raw and zx_raw.valid) else None
            
            # 2. Scanner scan()
            res = scanner.scan(cropped)
            
            print(f"Datei: {fn:<8} | Path: {sub:<28} | zxingcpp DMX: {str(zx_text):<8} | Scanner Result: {res['result']:<6} | Method: {res['method']:<12} | Verified: {res['verified']}")
