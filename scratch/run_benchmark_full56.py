"""Run cleanroom 56-image benchmark and print detailed statistics."""
import sys, os, json, time
sys.path.insert(0, '.')
import cv2
import scanner
from ultralytics import YOLO

# Load GT
with open("ground_truth_cleanroom.json", "r", encoding="utf-8") as f:
    gt_data = json.load(f)

print(f"Starte Cleanroom Benchmark mit {len(gt_data)} Bildern...\n")

model_path = os.path.join("runs", "detect", "training_runs_v2", "horde_2class", "weights", "best.pt")
model = YOLO(model_path)

correct = 0
errors = 0
misreads = 0
total = len(gt_data)

details = []
t_start = time.time()

for img_name, expected in gt_data.items():
    img_path = os.path.join("training_data", img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"WARNUNG: {img_name} nicht gefunden!")
        continue
    
    t0 = time.time()
    # YOLO 2-class detection
    yolo_res = model(img, verbose=False)
    dets = []
    for box in yolo_res[0].boxes:
        c = int(box.cls[0])
        cf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if cf > 0.3:
            dets.append({"cls": c, "box": (x1, y1, x2, y2), "conf": cf})
            
    res = scanner.scan_2class(img, dets) if dets else scanner.scan(img)
    dt_ms = int((time.time() - t0) * 1000)
    
    result_code = res.get("result") if res.get("success") else "FEHLER"
    method = res.get("method", "Fehler")
    conf = res.get("confidence", 0.0)
    
    if result_code == expected:
        status = "OK"
        correct += 1
    elif result_code == "FEHLER" or not res.get("success"):
        status = "FEHLER (Sauber)"
        errors += 1
    else:
        status = "MISREAD (FEHLLESUNG!)"
        misreads += 1
        
    details.append((img_name, expected, result_code, status, method, conf, dt_ms))

t_total = time.time() - t_start

print("=" * 110)
print(f"{'Bildname':38s} | {'GT':6s} | {'Erkannt':8s} | {'Status':22s} | {'Methode':12s} | {'Conf':5s} | {'Zeit'}")
print("=" * 110)

for img_name, expected, result_code, status, method, conf, dt_ms in details:
    print(f"{img_name:38s} | {expected:6s} | {result_code:8s} | {status:22s} | {method:12s} | {conf:5.2f} | {dt_ms}ms")

print("=" * 110)
print(f"ZUSAMMENFASSUNG:")
print(f"  Gesamtbilder: {total}")
print(f"  Korrekte Erkennungen: {correct} / {total} ({correct/total*100:.1f}%)")
print(f"  Saubere Fehler (Abgelehnt): {errors} / {total} ({errors/total*100:.1f}%)")
print(f"  Fehllesungen (Falscher Code!): {misreads} / {total} ({misreads/total*100:.1f}%)")
print(f"  Gesamtzeit: {t_total:.2f}s (Durchschnitt: {t_total/total*1000:.0f}ms pro Bild)")
print("=" * 110)
