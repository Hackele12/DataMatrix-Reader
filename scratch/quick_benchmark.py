"""
quick_benchmark.py — Schneller v16 Scanner Benchmark
"""

import os
import sys
import glob
import json
import time
import cv2
import numpy as np

# Set environment
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, app_dir)
import scanner

def run_quick_benchmark():
    app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    images_dir = os.path.join(app_dir, "training_data")
    gt_path = os.path.join(app_dir, "ground_truth.json")
    
    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)
        
    image_files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    
    from ultralytics import YOLO
    model_path = os.path.join(app_dir, "runs", "detect", "training_runs", "horde_model", "weights", "best.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(app_dir, "yolov10n.pt")
    model = YOLO(model_path)
    
    print(f"==================================================")
    print(f"  v16 SCANNER BENCHMARK ({len(image_files)} Testbilder)")
    print(f"==================================================\n")
    
    correct = 0
    total = len(image_files)
    total_time_ms = 0
    methods = {}
    
    for i, img_path in enumerate(image_files, 1):
        fn = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # YOLO Crop
        results = model.predict(img, conf=0.15, verbose=False)
        boxes = results[0].boxes
        if len(boxes) > 0:
            x1, y1, x2, y2 = map(int, boxes[0].xyxy[0])
            cropped = scanner.deskew_crop(img, (x1, y1, x2, y2), padding=60)
        else:
            cropped = img
            
        t0 = time.time()
        scan_res = scanner.scan(cropped)
        duration_ms = int((time.time() - t0) * 1000)
        
        gt = ground_truth.get(fn, "?")
        success = scan_res.get("success", False)
        result = scan_res.get("result", "")
        method = scan_res.get("method", "Fehler")
        
        is_correct = (success and result == gt)
        if is_correct:
            correct += 1
            
        total_time_ms += duration_ms
        methods[method] = methods.get(method, 0) + 1
        
        status = "OK" if is_correct else "FAIL"
        print(f"[{i:02d}/{total}] {fn:<12} -> {result:<6} (Soll: {gt:<4}) | Status: {status:<4} | Method: {method:<14} | {duration_ms}ms")
        
    acc_pct = (correct / total) * 100.0 if total > 0 else 0
    avg_dur = total_time_ms / total if total > 0 else 0
    
    print(f"\n==================================================")
    print(f"  v16 BENCHMARK ERGEBNISSE:")
    print(f"==================================================")
    print(f"  Bilder gesamt:       {total}")
    print(f"  Korrekt erkannt:     {correct}/{total} ({acc_pct:.1f}%)")
    print(f"  Durchschnitts-Dauer: {avg_dur:.1f} ms")
    print(f"\n  Methoden-Aufschlüsselung:")
    for m, c in sorted(methods.items(), key=lambda x: -x[1]):
        print(f"    - {m:<18}: {c:>2}x ({100*c/total:.1f}%)")
    print(f"==================================================\n")

if __name__ == "__main__":
    run_quick_benchmark()
