import os
import sys
import glob
import cv2
import json
import time
import logging

APP_DIR = r"c:\Users\kremidas\Documents\DataDetector"
sys.path.insert(0, APP_DIR)

import benchmark_suite
import scanner
import horde_db
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("two_pass_benchmark")

GT_PATH = os.path.join(APP_DIR, "ground_truth.json")
SPLITS_ROOT = os.path.join(APP_DIR, "training_data_splits")
ARTIFACT_WALKTHROUGH = r"C:\Users\kremidas\.gemini\antigravity-ide\brain\bf448dd0-c328-4d2a-8261-302090026ebd\walkthrough.md"

def clear_horde_db():
    db_dir = horde_db.get_horde_db_dir()
    for f in glob.glob(os.path.join(db_dir, "*.jpg")):
        try:
            os.remove(f)
        except Exception:
            pass
    horde_db.load_horde_db(force_reload=True)

def run_pass(pass_name, model, gt_map, split_folders):
    logger.info(f"=== STARTE DURCHLAUF: {pass_name} ===")
    results = []
    
    for folder in split_folders:
        folder_name = os.path.basename(folder)
        images = sorted(glob.glob(os.path.join(folder, "*.jpg")))

        for img_path in images:
            fn = os.path.basename(img_path)
            expected_gt = gt_map.get(fn, "?")

            image = cv2.imread(img_path)
            if image is None:
                continue

            t0 = time.time()
            try:
                cropped, yolo_conf = benchmark_suite._detect_and_crop(model, image)
                scan_res = scanner.scan(cropped)
            except Exception as e:
                scan_res = {"success": False, "result": "Fehler Exception", "method": "Fehler", "confidence": 0.0, "verified": False}

            dur_ms = round((time.time() - t0) * 1000.0, 1)
            is_success = scan_res.get("success", False)
            result_code = scan_res.get("result", "Kein Code")
            raw_method = scan_res.get("method", "Unbekannt")
            verified = scan_res.get("verified", False)
            conf = float(scan_res.get("confidence", 0.0))

            is_gt_match = (is_success and result_code == expected_gt)

            # Categorize Method
            if "Fast-Path" in raw_method or raw_method == "Verifiziert":
                method_cat = "DataMatrix (Fast-Path)"
            elif "HordenDB" in raw_method or "Bildabgleich" in raw_method:
                method_cat = "Horden-DB Match"
            elif "OCR" in raw_method:
                method_cat = "OCR (EasyOCR)"
            elif "Rekonstruiert" in raw_method:
                method_cat = "Rekonstruktion"
            else:
                method_cat = "Kein Code / Fehler"

            # Save to Horde DB if verified
            is_late = (dur_ms > 6000)
            db_action = "SKIPPED"
            if is_success:
                db_res = horde_db.save_or_update_horde_image(
                    code=result_code,
                    frame=cropped,
                    is_late_scan=is_late,
                    verified=verified,
                    confidence=conf
                )
                if db_res:
                    db_action = "SAVED"
                else:
                    db_action = "REJECTED (GUARD)"

            results.append({
                "pass": pass_name,
                "folder": folder_name,
                "file": fn,
                "gt": expected_gt,
                "res": result_code,
                "match": is_gt_match,
                "method": method_cat,
                "raw_method": raw_method,
                "conf": conf,
                "verified": verified,
                "dur_ms": dur_ms,
                "db_action": db_action
            })

    return results

def main():
    with open(GT_PATH, "r", encoding="utf-8") as f:
        gt_map = json.load(f)

    model_path = os.path.join(APP_DIR, "runs", "detect", "training_runs", "horde_model", "weights", "best.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(APP_DIR, "yolov10n.pt")
    model = YOLO(model_path)

    split_folders = sorted([d for d in glob.glob(os.path.join(SPLITS_ROOT, "part_*")) if os.path.isdir(d)])

    # Clear Horde DB for fresh Pass 1
    clear_horde_db()
    
    # PASS 1: Build Database from Verified Scans
    p1_results = run_pass("Pass 1 (Erstdurchlauf)", model, gt_map, split_folders)
    
    # Reload Horde DB Cache
    horde_db.load_horde_db(force_reload=True)
    
    # PASS 2: Acceleration & Fast Lookup with Populated Database
    p2_results = run_pass("Pass 2 (Zweitdurchlauf <6s)", model, gt_map, split_folders)

    # Print Summary Comparison
    p1_gt = sum(1 for r in p1_results if r["match"])
    p2_gt = sum(1 for r in p2_results if r["match"])
    p1_avg_dur = sum(r["dur_ms"] for r in p1_results) / len(p1_results)
    p2_avg_dur = sum(r["dur_ms"] for r in p2_results) / len(p2_results)
    p2_under_6s = sum(1 for r in p2_results if r["dur_ms"] < 6000)

    print("\n" + "="*80)
    print("              ZWEI-DURCHLAUF BENCHMARK ERGEBNISSE")
    print("="*80)
    print(f"PASS 1 (Erstdurchlauf):  GT-Matches: {p1_gt}/172 ({p1_gt/172*100:.1f}%) | Ø Dauer: {p1_avg_dur:.0f} ms")
    print(f"PASS 2 (Zweitdurchlauf): GT-Matches: {p2_gt}/172 ({p2_gt/172*100:.1f}%) | Ø Dauer: {p2_avg_dur:.0f} ms")
    print(f"Pass 2 Bilder unter 6.0s: {p2_under_6s}/172 ({p2_under_6s/172*100:.1f}%)")
    print("="*80)

if __name__ == "__main__":
    main()
