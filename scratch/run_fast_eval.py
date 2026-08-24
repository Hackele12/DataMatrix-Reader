"""
run_fast_eval.py — Echtzeit Benchmark-Auswertung für v16
"""

import os
import sys
import glob
import json
import time
import cv2
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, app_dir)

import scanner
from ultralytics import YOLO

def main():
    images_dir = os.path.join(app_dir, "training_data")
    gt_path = os.path.join(app_dir, "ground_truth.json")
    
    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)
        
    image_files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    
    model_path = os.path.join(app_dir, "runs", "detect", "training_runs", "horde_model", "weights", "best.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(app_dir, "yolov10n.pt")
    model = YOLO(model_path)
    
    print(f"\n==================================================", flush=True)
    print(f"  v16 SCANNER EVALUATION BENCHMARK ({len(image_files)} Testbilder)", flush=True)
    print(f"==================================================\n", flush=True)
    
    results = []
    correct = 0
    
    for i, img_path in enumerate(image_files, 1):
        fn = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        cropped, yolo_conf = scanner.deskew_crop(img, (0, 0, img.shape[1], img.shape[0]), padding=0) if len(model.predict(img, conf=0.15, verbose=False)[0].boxes) == 0 else (scanner.deskew_crop(img, map(int, model.predict(img, conf=0.15, verbose=False)[0].boxes[0].xyxy[0]), padding=60), float(model.predict(img, conf=0.15, verbose=False)[0].boxes[0].conf[0]))
        
        t0 = time.time()
        scan_res = scanner.scan(cropped)
        dur_ms = int((time.time() - t0) * 1000)
        
        gt = ground_truth.get(fn, "?")
        success = scan_res.get("success", False)
        result = scan_res.get("result", "")
        method = scan_res.get("method", "Fehler")
        is_correct = (success and result == gt)
        if is_correct:
            correct += 1
            
        results.append({
            "filename": fn,
            "expected": gt,
            "result": result,
            "success": success,
            "correct": is_correct,
            "method": method,
            "dur_ms": dur_ms,
        })
        
        sym = "[OK]" if is_correct else "[FAIL]"
        print(f"[{i:02d}/{len(image_files)}] {fn:<10} | GT: {gt:<4} | Res: {result:<6} | {sym:<6} | {method:<14} | {dur_ms}ms", flush=True)
        
    print(f"\n==================================================", flush=True)
    print(f"  BENCHMARK ERGEBNISSE:", flush=True)
    print(f"  Gesamtzahl:         {len(results)}", flush=True)
    print(f"  Genauigkeit:        {correct}/{len(results)} ({100*correct/len(results):.1f}%)", flush=True)
    print(f"  Durchschnittszeit:  {sum(r['dur_ms'] for r in results)/len(results):.1f} ms", flush=True)
    print(f"==================================================\n", flush=True)

if __name__ == "__main__":
    main()
