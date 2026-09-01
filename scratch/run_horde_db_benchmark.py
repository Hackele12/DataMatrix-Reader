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
logger = logging.getLogger("horde_db_benchmark")

GT_PATH = os.path.join(APP_DIR, "ground_truth.json")
HORDE_DIR = os.path.join(APP_DIR, "horden_db")
SPLITS_ROOT = os.path.join(APP_DIR, "training_data_splits")

def main():
    if not os.path.exists(GT_PATH):
        raise FileNotFoundError(f"GT path {GT_PATH} not found!")

    with open(GT_PATH, "r", encoding="utf-8") as f:
        gt_map = json.load(f)

    # 1. Existing Horde DB status
    horde_files = glob.glob(os.path.join(HORDE_DIR, "*.jpg"))
    existing_horde_codes = [os.path.basename(f).replace(".jpg", "") for f in horde_files]
    logger.info(f"--- HORDEN-DB INITIALISIERUNG ---")
    logger.info(f"Vorhandene verifizierte Vorlagen in horden_db: {len(horde_files)} {existing_horde_codes}")

    # Load YOLO model
    model_path = os.path.join(APP_DIR, "runs", "detect", "training_runs", "horde_model", "weights", "best.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(APP_DIR, "yolov10n.pt")
    model = YOLO(model_path)

    split_folders = sorted([d for d in glob.glob(os.path.join(SPLITS_ROOT, "part_*")) if os.path.isdir(d)])
    
    total_images = 0
    total_gt_matches = 0
    total_fast_path = 0
    total_refimg_matches = 0

    saved_to_db = 0
    rejected_by_guard = 0

    part_results = []

    for folder in split_folders:
        folder_name = os.path.basename(folder)
        images = sorted(glob.glob(os.path.join(folder, "*.jpg")))
        
        folder_img_count = len(images)
        folder_gt_matches = 0
        folder_ref_matches = 0

        for img_path in images:
            fn = os.path.basename(img_path)
            expected_gt = gt_map.get(fn, "?")

            image = cv2.imread(img_path)
            if image is None:
                continue

            h, w = image.shape[:2]
            if h < 300 or w < 300:
                continue

            t0 = time.time()
            try:
                cropped, yolo_conf = benchmark_suite._detect_and_crop(model, image)
                scan_res = scanner.scan(cropped)
            except Exception as e:
                scan_res = {"success": False, "result": "Fehler Exception", "method": "Fehler", "confidence": 0.0}

            dur_ms = (time.time() - t0) * 1000.0
            is_success = scan_res.get("success", False)
            result_code = scan_res.get("result", "")
            method = scan_res.get("method", "Unbekannt")
            verified = scan_res.get("verified", False)
            conf = scan_res.get("confidence", 0.0)

            is_match = (is_success and result_code == expected_gt)

            if is_match:
                folder_gt_matches += 1
            if "Fast-Path" in method or dur_ms < 100:
                total_fast_path += 1
            if "RefImg" in method:
                folder_ref_matches += 1

            # Test DB guard
            is_late = (dur_ms > 6000)
            if is_success:
                db_res = horde_db.save_or_update_horde_image(
                    code=result_code,
                    frame=cropped,
                    is_late_scan=is_late,
                    verified=verified,
                    confidence=conf
                )
                if db_res:
                    saved_to_db += 1
                else:
                    rejected_by_guard += 1

        total_images += folder_img_count
        total_gt_matches += folder_gt_matches
        total_refimg_matches += folder_ref_matches

        part_results.append({
            "part": folder_name,
            "total": folder_img_count,
            "matches": folder_gt_matches,
            "acc": (folder_gt_matches / folder_img_count * 100.0) if folder_img_count > 0 else 0.0,
            "ref_matches": folder_ref_matches
        })

    print("\n=========================================================================================================")
    print("                      HORDEN-DB BENCHMARK EVALUIERUNG UND GESAMTAUSWERTUNG")
    print("=========================================================================================================")
    print(f"{'Ordner':<12} | {'Bilder':<8} | {'GT-Match':<12} | {'Genauigkeit':<12} | {'RefImg-Treffer':<15}")
    print("---------------------------------------------------------------------------------------------------------")
    for r in part_results:
        print(f"{r['part']:<12} | {r['total']:<8} | {r['matches']:<12} | {r['acc']:>6.1f}%      | {r['ref_matches']:<15}")
    print("=========================================================================================================")
    overall_acc = (total_gt_matches / total_images * 100.0) if total_images > 0 else 0.0
    print(f"GESAMT       | {total_images:<8} | {total_gt_matches:<12} | {overall_acc:>6.1f}%      | {total_refimg_matches:<15}")
    print("=========================================================================================================")
    print(f"\n--- HORDEN-DATENBANK SCHUTZ-AUDIT BILANZ ---")
    print(f"  Verifizierte Vorlagen in horden_db gespeichert/aktualisiert: {saved_to_db}x")
    print(f"  Unverifizierte Ergebnisse von horden_db ABGELEHNT:         {rejected_by_guard}x")
    print("=========================================================================================================\n")

if __name__ == "__main__":
    main()
