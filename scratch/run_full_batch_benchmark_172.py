import os
import sys
import glob
import cv2
import json
import time
import logging
from datetime import datetime

APP_DIR = r"c:\Users\kremidas\Documents\DataDetector"
sys.path.insert(0, APP_DIR)

import benchmark_suite
import scanner
import horde_db
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("full_benchmark_172")

SPLITS_ROOT = os.path.join(APP_DIR, "training_data_splits")
GT_PATH = os.path.join(APP_DIR, "ground_truth.json")
OUTPUT_DIR = os.path.join(APP_DIR, "benchmarks_split_results")


def main():
    if not os.path.exists(GT_PATH):
        raise FileNotFoundError(f"Ground truth file {GT_PATH} not found!")

    with open(GT_PATH, "r", encoding="utf-8") as f:
        gt_map = json.load(f)

    model_path = os.path.join(APP_DIR, "runs", "detect", "training_runs", "horde_model", "weights", "best.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(APP_DIR, "yolov10n.pt")
    model = YOLO(model_path)

    split_folders = sorted([d for d in glob.glob(os.path.join(SPLITS_ROOT, "part_*")) if os.path.isdir(d)])
    logger.info(f"Found {len(split_folders)} split folders for evaluation: {[os.path.basename(f) for f in split_folders]}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    folder_summaries = []
    all_details = []
    report_paths = []

    grand_total_images = 0
    grand_total_successes = 0
    grand_total_gt_matches = 0
    grand_total_duration_ms = 0.0

    horde_db_saved_count = 0
    horde_db_rejected_count = 0

    for idx, folder in enumerate(split_folders, 1):
        folder_name = os.path.basename(folder)
        images = sorted(glob.glob(os.path.join(folder, "*.jpg")))
        total_img = len(images)

        logger.info(f"\n--- Evaluierung von {folder_name} ({total_img} Bilder) ---")

        folder_success = 0
        folder_gt_matches = 0
        folder_duration_sum = 0.0

        folder_details = []

        for i, img_path in enumerate(images, 1):
            fn = os.path.basename(img_path)
            expected_gt = gt_map.get(fn, "?")

            image = cv2.imread(img_path)
            if image is None:
                continue

            # Skip small pre-cropped thumbnail test snippets (<300px) that cause C++ bounds crashes
            h, w = image.shape[:2]
            if h < 300 or w < 300:
                logger.info(f"[{folder_name} {i}/{total_img}] Skipping small pre-cropped thumbnail {fn} ({w}x{h} px)")
                continue

            t0 = time.time()
            try:
                cropped, yolo_conf = benchmark_suite._detect_and_crop(model, image)
                scan_res = scanner.scan(cropped)
            except Exception as e:
                scan_res = {"success": False, "result": "Fehler Exception", "method": "Fehler", "confidence": 0.0}
                yolo_conf = 0.0

            dur_ms = (time.time() - t0) * 1000.0
            is_success = scan_res.get("success", False)
            result_code = scan_res.get("result", "")
            method = scan_res.get("method", "Unbekannt")
            confidence = scan_res.get("confidence", 0.0)
            verified = scan_res.get("verified", False)

            is_match = (is_success and result_code == expected_gt)

            if is_success:
                folder_success += 1
            if is_match:
                folder_gt_matches += 1

            folder_duration_sum += dur_ms

            # Check Horde DB Guard behavior
            is_late = (dur_ms > 6000)
            horde_save_path = None
            if is_success:
                horde_save_path = horde_db.save_or_update_horde_image(
                    code=result_code,
                    frame=cropped if cropped is not None else image,
                    is_late_scan=is_late,
                    verified=verified,
                    confidence=confidence
                )

            horde_db_status = "SAVED" if horde_save_path is not None else ("REJECTED (GUARD)" if is_late else "NOT SAVED")
            if horde_save_path is not None:
                horde_db_saved_count += 1
            elif is_late and is_success:
                horde_db_rejected_count += 1

            detail_item = {
                "filename": fn,
                "folder": folder_name,
                "success": is_success,
                "result_code": result_code,
                "expected_gt": expected_gt,
                "is_match": is_match,
                "method": method,
                "confidence": round(confidence, 4),
                "verified": verified,
                "duration_ms": round(dur_ms, 1),
                "horde_db_status": horde_db_status
            }
            folder_details.append(detail_item)
            all_details.append(detail_item)

            logger.info(f"[{folder_name} {i}/{total_img}] {fn:<8} -> Res='{result_code:<4}' GT='{expected_gt:<4}' Match={str(is_match):<5} ({dur_ms:.0f}ms, HordeDB: {horde_db_status})")

        avg_dur = folder_duration_sum / total_img if total_img > 0 else 0.0
        succ_pct = (folder_success / total_img * 100) if total_img > 0 else 0.0
        acc_pct = (folder_gt_matches / total_img * 100) if total_img > 0 else 0.0

        grand_total_images += total_img
        grand_total_successes += folder_success
        grand_total_gt_matches += folder_gt_matches
        grand_total_duration_ms += folder_duration_sum

        folder_summary = {
            "folder": folder_name,
            "total_images": total_img,
            "success_count": folder_success,
            "success_rate_pct": round(succ_pct, 1),
            "gt_matches": folder_gt_matches,
            "gt_total": total_img,
            "accuracy_pct": round(acc_pct, 1),
            "avg_duration_ms": round(avg_dur, 1)
        }
        folder_summaries.append(folder_summary)

        # Write part json
        part_json_path = os.path.join(OUTPUT_DIR, f"benchmark_{folder_name}.json")
        with open(part_json_path, "w", encoding="utf-8") as f:
            json.dump({"summary": folder_summary, "details": folder_details}, f, indent=2, ensure_ascii=False)
        report_paths.append(part_json_path)

    # Calculate grand total metrics
    tot_succ_pct = (grand_total_successes / grand_total_images * 100) if grand_total_images > 0 else 0.0
    tot_acc_pct = (grand_total_gt_matches / grand_total_images * 100) if grand_total_images > 0 else 0.0
    tot_avg_dur = (grand_total_duration_ms / grand_total_images) if grand_total_images > 0 else 0.0

    print("\n" + "=" * 105)
    print("                      NEUER BATCH BENCHMARK GESAMTBERICHT (172 BILDER)")
    print("=" * 105)
    print(f"{'Ordner':<12} | {'Bilder':<8} | {'Erfolge':<10} | {'Erfolgs-%':<10} | {'GT-Genauigkeit':<15} | {'Ø Dauer / Bild':<12}")
    print("-" * 105)

    for item in folder_summaries:
        gt_str = f"{item['gt_matches']}/{item['gt_total']} ({item['accuracy_pct']:.1f}%)"
        print(f"{item['folder']:<12} | {item['total_images']:<8} | {item['success_count']:<10} | {item['success_rate_pct']:.1f}%     | {gt_str:<15} | {item['avg_duration_ms']:.0f} ms")

    print("=" * 105)
    tot_gt_str = f"{grand_total_gt_matches}/{grand_total_images} ({tot_acc_pct:.1f}%)"
    print(f"{'GESAMT (SUMME)':<12} | {grand_total_images:<8} | {grand_total_successes:<10} | {tot_succ_pct:.1f}%     | {tot_gt_str:<15} | {tot_avg_dur:.0f} ms")
    print("=" * 105 + "\n")

    print("--- HORDEN-DATENBANK SPEICHER-SCHUTZ BILANZ ---")
    print(f"  Erfolgreich in horden_db/ gespeichert (Sicher / Verifiziert): {horde_db_saved_count}x")
    print(f"  Vom Datenbank-Schutz ABGELEHNT (>6s unverifiziert):           {horde_db_rejected_count}x")
    print("=" * 105 + "\n")

    # Save summary json
    summary_path = os.path.join(OUTPUT_DIR, "combined_batch_benchmark_summary.json")
    summary_report = {
        "timestamp": datetime.now().isoformat(),
        "total_images": grand_total_images,
        "total_successes": grand_total_successes,
        "overall_success_rate_pct": round(tot_succ_pct, 1),
        "total_gt_matches": grand_total_gt_matches,
        "overall_accuracy_pct": round(tot_acc_pct, 1),
        "overall_avg_duration_ms": round(tot_avg_dur, 1),
        "horde_db_saved_count": horde_db_saved_count,
        "horde_db_rejected_count": horde_db_rejected_count,
        "folder_summaries": folder_summaries,
        "all_details": all_details
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)

    logger.info(f"Gesamtbericht erfolgreich gespeichert in: {summary_path}")


if __name__ == "__main__":
    main()
