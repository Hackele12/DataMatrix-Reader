"""Debug script to inspect 0 vs 6 OCR misreads in 131044 and 131052."""
import sys, os
sys.path.insert(0, '.')
import cv2
import numpy as np
import scanner

for img_name in ["train_data_20260820_131044.jpg", "SCN-20260820-131052-0023.jpg", "SCN-20260820-130950-0021.jpg", "SCN-20260820-131030-0022.jpg"]:
    img_path = f"training_data/{img_name}"
    img = cv2.imread(img_path)
    if img is None:
        continue
    
    print(f"\n==================== {img_name} ====================")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Try different gamma values and inspect OCR character confidences
    reader = scanner._load_ocr()
    for g in [0.25, 0.3, 0.35, 0.4]:
        lut = np.array([((i / 255.0) ** g) * 255 for i in range(256)]).astype("uint8")
        bright = cv2.LUT(gray, lut)
        
        # crop lower 65% as in _read_ocr_with_status
        h, w = bright.shape[:2]
        ocr_zone = bright[int(h * 0.35):, :]
        ocr_zone = cv2.copyMakeBorder(ocr_zone, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        
        results = reader.readtext(ocr_zone, detail=1, paragraph=False, beamWidth=1, allowlist=scanner.ALLOWED_CHARS)
        print(f"Gamma {g}: {results}")
