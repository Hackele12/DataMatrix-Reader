"""
audit_late_scans.py - Evaluates dataset, detects late scans (>6s),
cross-references Ground Truth to find mis-evaluated scans that would corrupt horden_db,
and captures full diagnostic logs.
"""

import os
import sys
import glob
import cv2
import json
import time
import logging
from datetime import datetime

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

import benchmark_suite
import scanner
from ultralytics import YOLO

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("audit_late_scans")

TRAINING_DIR = os.path.join(APP_DIR, "training_data")
GT_PATH = os.path.join(APP_DIR, "ground_truth.json")
OUTPUT_DIR = os.path.join(APP_DIR, "benchmarks_split_results")
LOG_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "late_scans_mismatches_log.json")

LATE_SCAN_THRESHOLD_MS = 6000  # 6 seconds


def main():
    if not os.path.exists(TRAINING_DIR):
        raise FileNotFoundError(f"Training directory '{TRAINING_DIR}' not found!")

    # Load Ground Truth
    gt_map = {}
    if os.path.exists(GT_PATH):
        with open(GT_PATH, "r", encoding="utf-8") as f:
            gt_map = json.load(f)
    logger.info(f"Loaded Ground Truth with {len(gt_map)} entries.")

    # Load YOLO model
    model_path = os.path.join(APP_DIR, "runs", "detect", "training_runs", "horde_model", "weights", "best.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(APP_DIR, "yolov10n.pt")
    model = YOLO(model_path)
    logger.info(f"YOLO model loaded from '{model_path}'.")

    image_paths = sorted(glob.glob(os.path.join(TRAINING_DIR, "*.jpg")))
    logger.info(f"Starting audit on {len(image_paths)} images in training_data...")

    all_audited = []
    late_scans = []
    wrong_late_saves = []

    print("\n" + "=" * 115)
    print(f"  {'#':<4} {'Datei':<12} {'Dauer':<10} {'Status':<10} {'Ergebnis':<12} {'GroundTruth':<12} {'Methode':<16} {'HordeDB Safe?':<15}")
    print("=" * 115)

    for idx, img_path in enumerate(image_paths, 1):
        filename = os.path.basename(img_path)
        expected_gt = gt_map.get(filename, "?")

        image = cv2.imread(img_path)
        if image is None:
            logger.warning(f"[{idx}] Could not read {filename}")
            continue

        # Known native C++ crash image check
        if filename in ("079.jpg", "080.jpg"):
            logger.info(f"[{idx}] Skipping known C++ crash image {filename}")
            continue

        t0 = time.time()
        try:
            cropped, yolo_conf = benchmark_suite._detect_and_crop(model, image)
            scan_res = scanner.scan(cropped)
        except Exception as e:
            logger.error(f"[{idx}] Exception during scan of {filename}: {e}")
            scan_res = {"success": False, "result": "Fehler Exception", "method": "Fehler", "confidence": 0.0}
            yolo_conf = 0.0

        duration_ms = int((time.time() - t0) * 1000)
        is_success = scan_res.get("success", False)
        result_code = scan_res.get("result", "")
        method = scan_res.get("method", "Unbekannt")
        confidence = scan_res.get("confidence", 0.0)
        verified = scan_res.get("verified", False)

        is_gt_match = (is_success and result_code == expected_gt)
        is_late = (duration_ms >= LATE_SCAN_THRESHOLD_MS)

        horde_db_action = "N/A"
        is_false_positive_save = False

        if is_late and is_success:
            if is_gt_match:
                horde_db_action = "SAFE SAVE"
            else:
                horde_db_action = "CORRUPT SAVE!"
                is_false_positive_save = True

        audit_entry = {
            "index": idx,
            "filename": filename,
            "duration_ms": duration_ms,
            "is_late_scan": is_late,
            "success": is_success,
            "result_code": result_code,
            "expected_gt": expected_gt,
            "is_gt_match": is_gt_match,
            "is_false_positive_save": is_false_positive_save,
            "method": method,
            "confidence": round(confidence, 4),
            "verified": verified,
            "yolo_conf": round(yolo_conf, 4),
            "dmtx_result": scan_res.get("dmtx_result"),
            "ocr_result": scan_res.get("ocr_result"),
            "ocr_partial_display": scan_res.get("ocr_partial_display"),
            "full_scan_response": scan_res
        }

        all_audited.append(audit_entry)

        if is_late:
            late_scans.append(audit_entry)
            if is_false_positive_save:
                wrong_late_saves.append(audit_entry)

        status_str = "SUCCESS" if is_success else "FAIL"
        action_str = f"🛑 {horde_db_action}" if is_false_positive_save else (f"✓ {horde_db_action}" if is_late else "—")

        print(f"  {idx:<4} {filename:<12} {duration_ms:<6} ms   {status_str:<10} {result_code:<12} {expected_gt:<12} {method:<16} {action_str:<15}")

    print("=" * 115 + "\n")

    # Summaries
    total_images = len(all_audited)
    total_late = len(late_scans)
    total_wrong_saves = len(wrong_late_saves)

    print("=" * 90)
    print("                    AUDIT >6s LATE SCANS & HORDE DB SUMMARY")
    print("=" * 90)
    print(f" Evaluierte Bilder gesamt:          {total_images}")
    print(f" Scans mit Laufzeit > 6s:          {total_late} ({(total_late/total_images*100):.1f}% aller Bilder)")
    print(f" Davon erfolgreich erkannt:       {sum(1 for s in late_scans if s['success'])}")
    print(f" Davon Korrekt (GT Match):         {sum(1 for s in late_scans if s['is_gt_match'])}")
    print(f" Davon FALSCH abgespeichert (GT Mismatch): {total_wrong_saves}")
    print("=" * 90 + "\n")

    if wrong_late_saves:
        print("-----------------------------------------------------------------------------------------")
        print("  DETAILLIERTER LOG ALLER FALSCH ABGESPEICHERTEN HORDE-BILDER (>6s GT MISMATCH):")
        print("-----------------------------------------------------------------------------------------")
        for item in wrong_late_saves:
            print(f"  • Datei: {item['filename']} | Dauer: {item['duration_ms']} ms | Methode: {item['method']}")
            print(f"    Erkannt: '{item['result_code']}'  vs  GroundTruth: '{item['expected_gt']}'")
            print(f"    Konfidenz: {item['confidence']} | Verifiziert: {item['verified']}")
            print(f"    OCR-Text: '{item['ocr_result']}' | OCR-Partial: '{item['ocr_partial_display']}'")
            print(f"    DataMatrix: '{item['dmtx_result']}'")
            print("  ---------------------------------------------------------------------------------------")

    # Save output log JSON
    audit_report = {
        "timestamp": datetime.now().isoformat(),
        "total_images_audited": total_images,
        "late_scan_threshold_ms": LATE_SCAN_THRESHOLD_MS,
        "total_late_scans": total_late,
        "total_false_positive_horde_saves": total_wrong_saves,
        "wrong_late_saves_log": wrong_late_saves,
        "all_late_scans": late_scans,
        "all_audited": all_audited
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(LOG_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2, ensure_ascii=False)

    logger.info(f"Vollständiger Audit-Log gespeichert in: {LOG_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
