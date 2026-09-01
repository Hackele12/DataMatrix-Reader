"""Diagnose der 5 fehlenden Bilder - Ausgabe in Datei."""
import os, sys, cv2, json, numpy as np

# Output in Datei umleiten
out_path = "diagnose_output.txt"
sys.stdout = open(out_path, "w", encoding="utf-8")

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
    img_path = os.path.join("training_data", img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"SKIP: {img_name}")
        continue

    print(f"\n{'='*90}")
    print(f"BILD: {img_name} | SOLL: {expected} | {img.shape[1]}x{img.shape[0]}")
    print(f"{'='*90}")

    # YOLO
    yolo_res = model(img, verbose=False)
    dets = []
    for box in yolo_res[0].boxes:
        c = int(box.cls[0]); cf = float(box.conf[0])
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        cn = "dmx" if c==0 else "txt"
        print(f"  YOLO: {cn} conf={cf:.3f} box=({x1},{y1},{x2},{y2})")
        dets.append({"cls":c,"box":(x1,y1,x2,y2),"conf":cf})

    # Full-frame scan
    res = scanner.scan(img)
    print(f"  scan(): success={res.get('success')} result={res.get('result')} method={res.get('method')} conf={res.get('confidence')}")

    # OCR full-frame
    ocr = scanner._read_ocr_with_status(img)
    print(f"  OCR full: text={ocr.get('text')} conf={ocr.get('confidence'):.2f} partial={ocr.get('partial_display')} raw={ocr.get('raw_candidate')}")

    # PACC full-frame
    pt, pc = scanner._predict_pacc(img)
    print(f"  PACC full: text={pt} conf={pc:.3f}")

    # Text crops mit verschiedenen Paddings
    for det in dets:
        if det["cls"] == 1:
            for pad in [30, 60, 100, 150]:
                crop = scanner.deskew_crop(img, det["box"], padding=pad)
                ocr_c = scanner._read_ocr_with_status(crop)
                pt2, pc2 = scanner._predict_pacc(crop)
                print(f"  TXT crop pad={pad}: OCR={ocr_c.get('text')} conf={ocr_c.get('confidence'):.2f} partial={ocr_c.get('partial_display')} | PACC={pt2} conf={pc2:.3f}")

    # Kontrastverstärkung
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))
    enh = clahe.apply(gray)
    ocr_e = scanner._read_ocr_with_status(enh)
    print(f"  OCR CLAHE: text={ocr_e.get('text')} conf={ocr_e.get('confidence'):.2f} partial={ocr_e.get('partial_display')}")

    # Gamma 0.4
    lut = np.array([((i/255.0)**0.4)*255 for i in range(256)]).astype("uint8")
    bright = cv2.LUT(gray, lut)
    ocr_b = scanner._read_ocr_with_status(bright)
    print(f"  OCR Gamma0.4: text={ocr_b.get('text')} conf={ocr_b.get('confidence'):.2f} partial={ocr_b.get('partial_display')}")

    # Gamma 0.3
    lut2 = np.array([((i/255.0)**0.3)*255 for i in range(256)]).astype("uint8")
    bright2 = cv2.LUT(gray, lut2)
    ocr_b2 = scanner._read_ocr_with_status(bright2)
    print(f"  OCR Gamma0.3: text={ocr_b2.get('text')} conf={ocr_b2.get('confidence'):.2f} partial={ocr_b2.get('partial_display')}")

sys.stdout.close()
sys.stdout = sys.__stdout__
print(f"Diagnose fertig -> {out_path}")
