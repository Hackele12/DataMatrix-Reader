import os
import sys
import glob
import json
import time
import subprocess
from datetime import datetime

APP_DIR = r"c:\Users\kremidas\Documents\DataDetector"
sys.path.insert(0, APP_DIR)

import benchmark_suite

PART_02_DIR = os.path.join(APP_DIR, "training_data_splits", "part_02")
OUTPUT_JSON = os.path.join(APP_DIR, "benchmarks_split_results", "benchmark_part_02.json")

print("Safe part_02 benchmark execution starting...")

gt_map = benchmark_suite._load_ground_truth(APP_DIR)
image_paths = sorted(glob.glob(os.path.join(PART_02_DIR, "*.jpg")))
print(f"Loaded {len(image_paths)} images from {PART_02_DIR}")

results = []
success_count = 0
accuracy_count = 0
total_gt_count = 0
total_duration_ms = 0

failure_categories = {}
method_distribution = {}
ocr_partial_breakdown = {"4/4": 0, "3/4": 0, "2/4": 0, "1/4": 0, "0/4": 0}

for idx, path in enumerate(image_paths, 1):
    filename = os.path.basename(path)
    source_dir = os.path.basename(os.path.dirname(path))

    # Single-image subprocess to prevent native C++ crashes from killing the runner
    cmd = [
        sys.executable,
        "-c",
        f"""
import cv2, os, sys, json
sys.path.insert(0, r'{APP_DIR}')
import benchmark_suite
import scanner
from ultralytics import YOLO

model_path = os.path.join(r'{APP_DIR}', 'runs', 'detect', 'training_runs', 'horde_model', 'weights', 'best.pt')
model = YOLO(model_path)
image = cv2.imread(r'{path}')
if image is None:
    print(json.dumps({{"error": "imread_failed"}}))
    sys.exit(0)

cropped, yolo_conf = benchmark_suite._detect_and_crop(model, image)
scan_res = scanner.scan(cropped)
scan_res['yolo_conf'] = yolo_conf
print(json.dumps(scan_res, ensure_ascii=False))
"""
    ]

    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=APP_DIR)
    duration_ms = int((time.time() - t0) * 1000)
    total_duration_ms += duration_ms

    if res.returncode != 0 or not res.stdout.strip():
        print(f"[{idx}/30] {filename}: NATIVE CRASH (exit code {res.returncode}) ({duration_ms}ms)")
        scan_res = {"success": False, "result": "Kein Code erkannt.", "method": "Fehler"}
        yolo_conf = 0.0
    else:
        try:
            # Parse json from last line of stdout
            lines = [l for l in res.stdout.strip().split("\n") if l.startswith("{") and l.endswith("}")]
            scan_res = json.loads(lines[-1])
            yolo_conf = scan_res.get("yolo_conf", 0.0)
        except Exception as e:
            print(f"[{idx}/30] {filename}: JSON parse error: {e}")
            scan_res = {"success": False, "result": "Kein Code erkannt.", "method": "Fehler"}
            yolo_conf = 0.0

    is_success = scan_res.get("success", False)
    result_code = scan_res.get("result", "")
    method = scan_res.get("method", "Unbekannt")
    expected_code = gt_map.get(filename, "?")

    partial_text, ocr_chars = benchmark_suite._analyze_ocr_partial(scan_res)
    if is_success:
        ocr_chars = 4
        partial_display = result_code
    else:
        partial_display = partial_text if partial_text else "—"

    key_chars = f"{ocr_chars}/4"
    ocr_partial_breakdown[key_chars] = ocr_partial_breakdown.get(key_chars, 0) + 1

    is_match = False
    if expected_code != "?":
        total_gt_count += 1
        if is_success and result_code == expected_code:
            is_match = True
            accuracy_count += 1

    if is_success:
        success_count += 1
        method_distribution[method] = method_distribution.get(method, 0) + 1
        fail_reason = "—"
    else:
        if is_success and not is_match and expected_code != "?":
            fail_reason = "MISMATCH"
        elif res.returncode != 0:
            fail_reason = "DMTX_AND_OCR_FAILED"
        else:
            fail_reason = benchmark_suite._diagnose_failure_reason(scan_res, yolo_conf)
        failure_categories[fail_reason] = failure_categories.get(fail_reason, 0) + 1

    print(f"[{idx}/30] {filename}: OK={is_success} Res={result_code} Expected={expected_code} ({duration_ms}ms)")

    results.append({
        "index": idx,
        "filename": filename,
        "source_dir": source_dir,
        "full_path": path,
        "success": is_success,
        "result_code": result_code,
        "expected_code": expected_code,
        "is_match": is_match,
        "method": method,
        "confidence": scan_res.get("confidence", 0.0),
        "yolo_conf": yolo_conf,
        "duration_ms": duration_ms,
        "fail_reason": fail_reason,
        "ocr_result": scan_res.get("ocr_result"),
        "ocr_partial_text": partial_text,
        "ocr_chars_recognized": ocr_chars,
        "dmtx_result": scan_res.get("dmtx_result"),
    })

total_images = len(image_paths)
avg_duration = total_duration_ms / total_images if total_images > 0 else 0
success_pct = (success_count / total_images) * 100 if total_images > 0 else 0
accuracy_pct = (accuracy_count / total_gt_count) * 100 if total_gt_count > 0 else 0

summary_data = {
    "total_images": total_images,
    "success_count": success_count,
    "success_rate_pct": round(success_pct, 1),
    "accuracy_gt_count": f"{accuracy_count}/{total_gt_count}",
    "accuracy_pct": round(accuracy_pct, 1),
    "avg_duration_ms": round(avg_duration, 1),
    "ocr_partial_breakdown": ocr_partial_breakdown,
    "method_distribution": method_distribution,
    "failure_categories": failure_categories,
}

report_data = {
    "timestamp": datetime.now().isoformat(),
    "summary": summary_data,
    "details": results,
}

os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(report_data, f, indent=2, ensure_ascii=False)

print(f"Saved part_02 report to {OUTPUT_JSON} successfully!")
