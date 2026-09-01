"""Test DataMatrix decoding with various image preprocessing on the 5 failing images."""
import sys, os
sys.path.insert(0, '.')
import cv2
import numpy as np
import scanner
import zxingcpp

FAILING = [
    "SCN-20260820-130950-0021.jpg",
    "SCN-20260820-131030-0022.jpg",
    "SCN-20260820-131052-0023.jpg",
    "SCN-20260824-130855-0011.jpg",
    "train_data_20260820_131044.jpg"
]

model_path = os.path.join("runs", "detect", "training_runs_v2", "horde_2class", "weights", "best.pt")
from ultralytics import YOLO
model = YOLO(model_path)

for img_name in FAILING:
    img_path = os.path.join("training_data", img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue
    
    print(f"\n==================== {img_name} ====================")
    
    # YOLO box for dmx
    yolo_res = model(img, verbose=False)
    dmx_boxes = [box for box in yolo_res[0].boxes if int(box.cls[0]) == 0]
    
    crops = []
    if dmx_boxes:
        box = tuple(map(int, dmx_boxes[0].xyxy[0]))
        for pad in [10, 20, 40, 60]:
            crop = scanner.deskew_crop(img, box, padding=pad)
            crops.append((f"crop_pad{pad}", crop))
    crops.append(("full_frame", img))
    
    decoded = False
    for crop_name, crop_img in crops:
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY) if len(crop_img.shape) == 3 else crop_img
        
        # Preprocessings:
        preprocessed_list = [("raw", gray)]
        
        for g in [0.2, 0.3, 0.4, 0.5, 0.6, 1.5, 2.0]:
            lut = np.array([((i / 255.0) ** g) * 255 for i in range(256)]).astype("uint8")
            preprocessed_list.append((f"gamma_{g}", cv2.LUT(gray, lut)))
            
        for clip in [2.0, 4.0, 8.0, 15.0, 20.0]:
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
            preprocessed_list.append((f"clahe_{clip}", clahe.apply(gray)))
            
            # Gamma + CLAHE
            lut03 = np.array([((i / 255.0) ** 0.3) * 255 for i in range(256)]).astype("uint8")
            bright = cv2.LUT(gray, lut03)
            preprocessed_list.append((f"gamma03_clahe_{clip}", clahe.apply(bright)))
            
        for prep_name, prep_img in preprocessed_list:
            # 1. zxing-cpp
            res = zxingcpp.read_barcodes(prep_img)
            if res:
                for r in res:
                    if r.text:
                        print(f"  [SUCCESS] {crop_name} | {prep_name} | zxing-cpp -> '{r.text}'")
                        decoded = True
            
            # 2. _read_datamatrix
            dmtx_text = scanner._read_datamatrix(prep_img)
            if dmtx_text:
                print(f"  [SUCCESS] {crop_name} | {prep_name} | scanner._read_datamatrix -> '{dmtx_text}'")
                decoded = True
                
            if decoded:
                break
        if decoded:
            break
            
    if not decoded:
        print("  [FAILED] No DataMatrix decode found with any preprocessing.")
