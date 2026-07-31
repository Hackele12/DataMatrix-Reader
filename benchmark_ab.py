"""
benchmark_ab.py — A/B-Benchmark-Skript für Horden-Scanner

Vergleicht die Performance des Scanners in 2 Modi:
  Mode A (Baseline):  Standard scanner.py (nur pylibdmtx + EasyOCR)
  Mode B (Verbessert): scanner.py mit zxing-cpp Fast-Path Integration

Nutzung:
    .venv/Scripts/python.exe benchmark_ab.py
"""

import os
import sys
import json
import time
import glob
import logging
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

try:
    import scanner
except Exception as e:
    logger.error(f"Scanner Importfehler: {e}")
    sys.exit(1)


def _load_yolo_model():
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics nicht installiert.")
        return None

    app_dir = os.path.dirname(os.path.abspath(__file__))
    trained_path = os.path.join(
        app_dir, "runs", "detect", "training_runs", "horde_model", "weights", "best.pt"
    )
    if os.path.exists(trained_path):
        return YOLO(trained_path)

    base_path = os.path.join(app_dir, "yolov10n.pt")
    if os.path.exists(base_path):
        return YOLO(base_path)

    return None


def _detect_and_crop(model, image: np.ndarray) -> tuple[np.ndarray, float]:
    results = model.predict(image, conf=0.15, verbose=False)
    boxes = results[0].boxes
    if len(boxes) == 0:
        return image, 0.0

    best_box = boxes[0]
    conf = float(best_box.conf[0])
    x1, y1, x2, y2 = map(int, best_box.xyxy[0])
    padding = 20
    fh, fw = image.shape[:2]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(fw, x2 + padding)
    y2 = min(fh, y2 + padding)

    return image[y1:y2, x1:x2], conf


def evaluate_mode(model, image_files: list[str], ground_truth: dict, mode_name: str, use_zxing: bool) -> list[dict]:
    # Set scanner mode
    scanner.USE_ZXING_FASTPATH = use_zxing

    if hasattr(scanner, "_recent_scans"):
        scanner._recent_scans = []

    results = []
    print(f"\n\033[96m\033[1m=== Starte Testlauf: {mode_name} ===\033[0m\n")

    for i, img_path in enumerate(image_files, 1):
        fn = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            continue

        cropped, yolo_conf = _detect_and_crop(model, img)
        t0 = time.time()
        scan_res = scanner.scan(cropped)
        dur_ms = int((time.time() - t0) * 1000)

        gt = ground_truth.get(fn, "?")
        success = scan_res.get("success", False)
        res_text = scan_res.get("result", "")
        method = scan_res.get("method", "")
        conf = scan_res.get("confidence", 0.0)
        is_correct = (success and res_text == gt)

        results.append({
            "filename": fn,
            "expected": gt,
            "success": success,
            "result": res_text,
            "correct": is_correct,
            "method": method,
            "confidence": conf,
            "duration_ms": dur_ms,
        })

        status_symbol = "\033[92m✓\033[0m" if is_correct else "\033[91m✗\033[0m"
        print(f"  [{i:>2}/{len(image_files)}] {fn:<32} {status_symbol} Res='{res_text:<6}' (Soll='{gt:<4}') [{method:<12}, {dur_ms:>4} ms]")

    return results


def print_comparison_table(results_a: list[dict], results_b: list[dict]):
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    print(f"\n{BOLD}{'═' * 110}{RESET}")
    print(f"{BOLD}  A/B BENCHMARK COMPARISON TABLE (Baseline vs. Verbessert with zxing-cpp){RESET}")
    print(f"{BOLD}{'═' * 110}{RESET}\n")

    header = f"  {'Datei':<32} {'Soll':<6} | {'MODE A (Baseline)':<32} | {'MODE B (zxing-cpp)':<32}"
    print(f"{BOLD}{header}{RESET}")
    print(f"  {'─' * 106}")

    count_a_correct = sum(1 for r in results_a if r["correct"])
    count_b_correct = sum(1 for r in results_b if r["correct"])
    dur_a_total = sum(r["duration_ms"] for r in results_a)
    dur_b_total = sum(r["duration_ms"] for r in results_b)

    for ra, rb in zip(results_a, results_b):
        fn = ra["filename"]
        gt = ra["expected"]

        # Mode A string
        symbol_a = f"{GREEN}✓{RESET}" if ra["correct"] else f"{RED}✗{RESET}"
        res_a = f"{symbol_a} {ra['result']:<5} ({ra['method']:<11}, {ra['duration_ms']:>4}ms)"

        # Mode B string
        symbol_b = f"{GREEN}✓{RESET}" if rb["correct"] else f"{RED}✗{RESET}"
        res_b = f"{symbol_b} {rb['result']:<5} ({rb['method']:<11}, {rb['duration_ms']:>4}ms)"

        print(f"  {fn:<32} {gt:<6} | {res_a:<41} | {res_b:<41}")

    total = len(results_a)
    avg_dur_a = dur_a_total / total if total > 0 else 0
    avg_dur_b = dur_b_total / total if total > 0 else 0

    print(f"\n{BOLD}{'─' * 106}{RESET}")
    print(f"  {BOLD}ZUSAMMENFASSUNG / SUMMARY:{RESET}")
    print(f"  Mode A (Baseline):   Genauigkeit: {GREEN if count_a_correct==total else YELLOW}{count_a_correct}/{total} ({100*count_a_correct/total:.1f}%){RESET} | Ø Dauer: {avg_dur_a:.0f} ms")
    print(f"  Mode B (zxing-cpp):  Genauigkeit: {GREEN if count_b_correct==total else YELLOW}{count_b_correct}/{total} ({100*count_b_correct/total:.1f}%){RESET} | Ø Dauer: {avg_dur_b:.0f} ms")

    diff_dur = avg_dur_b - avg_dur_a
    speedup = f"{abs(diff_dur):.0f} ms {'schneller' if diff_dur < 0 else 'langsamer'}"
    print(f"  {BOLD}Geschwindigkeits-Differenz:{RESET} {GREEN if diff_dur <= 0 else RED}{speedup}{RESET}")
    print(f"{BOLD}{'═' * 110}{RESET}\n")


def main():
    app_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(app_dir, "training_data")
    gt_path = os.path.join(app_dir, "ground_truth.json")

    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    image_files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))

    model = _load_yolo_model()
    if model is None:
        logger.error("YOLO Model not found.")
        sys.exit(1)

    # 1. Mode A: Baseline
    results_a = evaluate_mode(model, image_files, ground_truth, "Mode A (Baseline: pylibdmtx + EasyOCR)", use_zxing=False)

    # 2. Mode B: Improved (zxing-cpp Fast-Path)
    results_b = evaluate_mode(model, image_files, ground_truth, "Mode B (Verbessert: zxing-cpp Fast-Path)", use_zxing=True)

    # 3. Print side by side
    print_comparison_table(results_a, results_b)

    # 4. Save JSON report
    report = {
        "timestamp": datetime.now().isoformat(),
        "baseline": results_a,
        "improved": results_b,
    }
    report_path = os.path.join(app_dir, "benchmark_ab_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"A/B Benchmark Report gespeichert in {report_path}")


if __name__ == "__main__":
    main()
