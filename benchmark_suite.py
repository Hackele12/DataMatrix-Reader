"""
benchmark_suite.py — Unified Benchmark & Regression Suite for DataDetector

Automatisierte Evaluierung und Schwachstellenanalyse aller Testbilder aus:
1. 'training_data' (lokales Projektverzeichnis)
2. 'U:\\Temp\\DataMatrixReader.logFiles' (Reale Fehlerbilder & Feld-Logs)

Erweiterte Funktionen (Schritt 1.2):
- Automatische Vorher-/Nachher-Vergleichsfunktion (--compare baseline.json)
- Detaillierte Erfassung von OCR-Teilergebnissen (z. B. 3/4 Zeichen erkannt: "W03?")
- Konsolentabelle mit Differenzanzeige (+ / -) & Visualisierung neuer Treffer/Regresse
- JSON-Export mit OCR-Teilergebnis-Metriken

Nutzung:
    .venv/Scripts/python.exe benchmark_suite.py
    .venv/Scripts/python.exe benchmark_suite.py --output run1.json --compare benchmark_baseline.json
"""

import os
import sys
import json
import time
import glob
import logging
from datetime import datetime
from pathlib import Path

# --- KMP Multithreading Fix ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# --- Scanner Modul importieren ---
try:
    import scanner
except Exception as e:
    logger.error(f"Scanner Importfehler: {e}")
    sys.exit(1)


# ANSI Farbcodes
_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


DEFAULT_DATA_DIRS = [
    "training_data",
    r"U:\Temp\DataMatrixReader.logFiles\images",
    r"U:\Temp\DataMatrixReader.logFiles\cam1\images",
    r"U:\Temp\DataMatrixReader.logFiles\cam2\images",
]

LOGFILES_ROOT = r"U:\Temp\DataMatrixReader.logFiles"


def _load_ground_truth(app_dir: str) -> dict[str, str]:
    """
    Lädt die Ground-Truth-Datenbank aus ground_truth.json und erweitert sie
    automatisch um bekannte Log-Referenzen aus scans.jsonl Dateien.
    """
    gt_map: dict[str, str] = {}

    gt_path = os.path.join(app_dir, "ground_truth.json")
    if os.path.exists(gt_path):
        try:
            with open(gt_path, "r", encoding="utf-8") as f:
                gt_map.update(json.load(f))
            logger.info(f"Ground Truth geladen aus {gt_path} ({len(gt_map)} Einträge).")
        except Exception as e:
            logger.warning(f"Fehler beim Laden von {gt_path}: {e}")

    scan_jsonl_paths = [
        os.path.join(LOGFILES_ROOT, "scans.jsonl"),
        os.path.join(LOGFILES_ROOT, "cam1", "scans.jsonl"),
        os.path.join(LOGFILES_ROOT, "cam2", "scans.jsonl"),
    ]

    log_gt_hints = {
        "SCN-20260803-133134-0001.jpg": "W032",
        "SCN-20260803-133456-0002.jpg": "W032",
        "SCN-20260803-130603-0004.jpg": "W032",
        "SCN-20260803-132635-0006.jpg": "W032",
        "SCN-20260803-132738-0007.jpg": "W032",
        "SCN-20260803-132747-0008.jpg": "W032",
        "SCN-20260803-132834-0009.jpg": "W032",
        "SCN-20260803-132939-0010.jpg": "W032",
        "SCN-20260803-133019-0011.jpg": "W032",
        "SCN-20260803-133051-0012.jpg": "W032",
        "SCN-20260803-133530-0001.jpg": "W032",
        "SCN-20260803-133616-0002.jpg": "W032",
        "SCN-20260803-133657-0003.jpg": "W032",
        "SCN-20260803-133727-0004.jpg": "W032",
        "SCN-20260803-133829-0007.jpg": "W032",
        "SCN-20260803-133942-0009.jpg": "P999",
        "SCN-20260803-134036-0010.jpg": "P999",
        "SCN-20260803-134112-0011.jpg": "W032",
        "SCN-20260803-134244-0013.jpg": "W032",
        "SCN-20260803-134500-0016.jpg": "W032",
        "SCN-20260803-134641-0021.jpg": "W032",
        "SCN-20260803-134654-0023.jpg": "W032",
        "SCN-20260803-134728-0024.jpg": "W032",
    }

    for fname, code in log_gt_hints.items():
        if fname not in gt_map:
            gt_map[fname] = code

    for jpath in scan_jsonl_paths:
        if os.path.exists(jpath):
            try:
                with open(jpath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        data = json.loads(line)
                        img_info = data.get("image", {})
                        rel_path = img_info.get("path")
                        res_info = data.get("result", {})
                        code = res_info.get("code")
                        if rel_path and code and res_info.get("success"):
                            fn = os.path.basename(rel_path)
                            if fn not in gt_map:
                                gt_map[fn] = code
            except Exception as e:
                logger.debug(f"Hinweis beim Lesen von {jpath}: {e}")

    return gt_map


def _load_yolo_model():
    """Lädt das beste verfügbare YOLOv10-Modell."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics nicht installiert. Bitte: pip install ultralytics")
        return None

    app_dir = os.path.dirname(os.path.abspath(__file__))
    path_2class = os.path.join(
        app_dir, "runs", "detect", "training_runs_v2",
        "horde_2class", "weights", "best.pt"
    )
    if os.path.exists(path_2class):
        logger.info(f"Lade trainiertes 2-Klassen YOLO-Modell: {path_2class}")
        return YOLO(path_2class)

    path_1class = os.path.join(
        app_dir, "runs", "detect", "training_runs",
        "horde_model", "weights", "best.pt"
    )
    if os.path.exists(path_1class):
        logger.info(f"Lade trainiertes 1-Klassen YOLO-Modell: {path_1class}")
        return YOLO(path_1class)

    base_path = os.path.join(app_dir, "yolov10n.pt")
    if os.path.exists(base_path):
        logger.warning(f"Kein trainiertes Modell gefunden, nutze Base-Modell: {base_path}")
        return YOLO(base_path)

    logger.error("Kein YOLO-Modell (best.pt oder yolov10n.pt) gefunden.")
    return None


def _detect_and_crop(model, image: np.ndarray) -> tuple[np.ndarray, float]:
    """Segmentiert das Etikett im Bild via YOLO und schneidet es zu."""
    results = model.predict(image, conf=0.15, verbose=False)
    boxes = results[0].boxes

    if len(boxes) == 0:
        return image, 0.0

    best_box = boxes[0]
    conf = float(best_box.conf[0])
    x1, y1, x2, y2 = map(int, best_box.xyxy[0])

    cropped = scanner.deskew_crop(image, (x1, y1, x2, y2), padding=60)
    return cropped, conf


def _diagnose_failure_reason(scan_res: dict, yolo_conf: float) -> str:
    """Bestimmt die genaue Ausfallursache eines fehlerhaften Scans."""
    if yolo_conf == 0.0:
        return "YOLO_NO_LABEL"
    
    ocr_res = scan_res.get("ocr_result")
    dmtx_res = scan_res.get("dmtx_result")
    method = scan_res.get("method", "")
    
    if not ocr_res and not dmtx_res:
        return "DMTX_AND_OCR_FAILED"
    elif ocr_res and not dmtx_res:
        return "RECONSTRUCTION_REJECTED"
    elif dmtx_res and not ocr_res:
        return "OCR_FAILED"
    elif method == "Fehler":
        return "CONFIDENCE_TOO_LOW"
    else:
        return "UNKNOWN_FAIL"


def _analyze_ocr_partial(scan_res: dict) -> tuple[str | None, int]:
    """
    Analysiert OCR-Teilergebnisse und ermittelt:
    - partial_text: z. B. 'W03?' oder 'P999'
    - chars_recognized: Anzahl der gültigen/erkannten Zeichen (0 bis 4)
    """
    ocr_raw = scan_res.get("ocr_result")
    ocr_partial = scan_res.get("ocr_partial_display")
    
    candidate_str = None
    if ocr_raw and isinstance(ocr_raw, str) and len(ocr_raw.strip()) > 0:
        candidate_str = ocr_raw.strip()
    elif ocr_partial and isinstance(ocr_partial, str) and len(ocr_partial.strip()) > 0:
        candidate_str = ocr_partial.strip()
        
    if not candidate_str:
        result_text = scan_res.get("result", "")
        if "Teilweise erkannt:" in result_text:
            candidate_str = result_text.replace("Teilweise erkannt:", "").strip()

    if not candidate_str:
        return None, 0

    valid_chars = sum(1 for c in candidate_str if c != '?' and c != ' ')
    chars_recognized = min(4, valid_chars)
    return candidate_str, chars_recognized


def _compare_with_baseline(current_summary: dict, current_details: list[dict], baseline_path: str):
    """Vergleicht den aktuellen Run mit einem Baseline-JSON-Bericht und gibt Differenzen aus."""
    if not os.path.exists(baseline_path):
        logger.warning(f"Baseline-Datei nicht gefunden für Vergleich: {baseline_path}")
        return

    try:
        with open(baseline_path, "r", encoding="utf-8") as f:
            base_data = json.load(f)
    except Exception as e:
        logger.error(f"Fehler beim Lesen der Baseline-Datei {baseline_path}: {e}")
        return

    base_summary = base_data.get("summary", {})
    base_details_list = base_data.get("details", [])
    base_map = {item["filename"]: item for item in base_details_list}

    cur_success = current_summary["success_rate_pct"]
    cur_acc = current_summary["accuracy_pct"]
    cur_dur = current_summary["avg_duration_ms"]

    base_success = base_summary.get("success_rate_pct", 0.0)
    base_acc = base_summary.get("accuracy_pct", 0.0)
    base_dur = base_summary.get("avg_duration_ms", 0.0)

    diff_success = cur_success - base_success
    diff_acc = cur_acc - base_acc
    diff_dur = cur_dur - base_dur

    newly_solved = []
    regressions = []

    for item in current_details:
        fn = item["filename"]
        cur_ok = item["success"]
        base_item = base_map.get(fn)
        if base_item:
            base_ok = base_item.get("success", False)
            if not base_ok and cur_ok:
                newly_solved.append((fn, item["result_code"], item["method"]))
            elif base_ok and not cur_ok:
                regressions.append((fn, base_item.get("result_code", ""), item["fail_reason"]))

    print(f"\n{_CYAN}{_BOLD}{'═' * 80}{_RESET}")
    print(f"{_CYAN}{_BOLD}  VORHER / NACHHER VERGLEICH (vs. {os.path.basename(baseline_path)}){_RESET}")
    print(f"{_CYAN}{_BOLD}{'═' * 80}{_RESET}")

    s_color = _GREEN if diff_success > 0 else (_RED if diff_success < 0 else _RESET)
    a_color = _GREEN if diff_acc > 0 else (_RED if diff_acc < 0 else _RESET)
    d_color = _GREEN if diff_dur < 0 else (_RED if diff_dur > 0 else _RESET)

    print(f"  Erfolgsrate:     Baseline {base_success:.1f}%  →  Aktuell {cur_success:.1f}% ({s_color}{diff_success:+.1f}%{_RESET})")
    print(f"  Genauigkeit GT:  Baseline {base_acc:.1f}%  →  Aktuell {cur_acc:.1f}% ({a_color}{diff_acc:+.1f}%{_RESET})")
    print(f"  Ø Dauer/Bild:    Baseline {base_dur:.0f} ms  →  Aktuell {cur_dur:.0f} ms ({d_color}{diff_dur:+.0f} ms{_RESET})")

    print(f"\n  {_BOLD}Neu gelöste Bilder (Verbesserungen: {len(newly_solved)}):{_RESET}")
    if newly_solved:
        for fn, code, method in newly_solved:
            print(f"    {_GREEN}+ {fn:<32} -> {code:<8} (via {method}){_RESET}")
    else:
        print("    (Keine neuen Bilder gelöst)")

    print(f"\n  {_BOLD}Regresse / Verschlechterungen ({len(regressions)}):{_RESET}")
    if regressions:
        for fn, prev_code, reason in regressions:
            print(f"    {_RED}- {fn:<32} (Vorher: {prev_code} -> Nun FAIL: {reason}){_RESET}")
    else:
        print(f"    {_GREEN}✓ Keine Regresse!{_RESET}")

    print(f"{_CYAN}{_BOLD}{'═' * 80}{_RESET}\n")


def run_benchmark(
    custom_dirs: list[str] = None,
    output_report_path: str = "benchmark_baseline.json",
    compare_baseline_path: str = None
):
    app_dir = os.path.dirname(os.path.abspath(__file__))
    gt_map = _load_ground_truth(app_dir)

    dirs_to_scan = custom_dirs if custom_dirs else DEFAULT_DATA_DIRS
    image_paths: list[str] = []

    for d in dirs_to_scan:
        target_dir = d if os.path.isabs(d) else os.path.join(app_dir, d)
        if os.path.exists(target_dir):
            files = sorted(
                glob.glob(os.path.join(target_dir, "*.jpg"))
                + glob.glob(os.path.join(target_dir, "*.jpeg"))
                + glob.glob(os.path.join(target_dir, "*.png"))
            )
            image_paths.extend(files)
            logger.info(f"Ordner '{target_dir}': {len(files)} Bilder gefunden.")
        else:
            logger.warning(f"Ordner '{target_dir}' nicht gefunden — wird übersprungen.")

    unique_paths = list(dict.fromkeys(image_paths))
    total_images = len(unique_paths)

    if total_images == 0:
        logger.error("Keine Testbilder in den angegebenen Verzeichnissen gefunden!")
        return

    logger.info(f"\n{_BOLD}Gesamt-Test-Set: {total_images} Bilder werden evaluiert.{_RESET}\n")

    model = _load_yolo_model()
    if model is None:
        logger.error("YOLO-Modell konnte nicht geladen werden. Abbruch.")
        sys.exit(1)

    if hasattr(scanner, "_recent_scans"):
        scanner._recent_scans = []

    results = []
    success_count = 0
    accuracy_count = 0
    total_gt_count = 0
    total_duration_ms = 0

    failure_categories: dict[str, int] = {}
    method_distribution: dict[str, int] = {}
    ocr_partial_breakdown: dict[str, int] = {"4/4": 0, "3/4": 0, "2/4": 0, "1/4": 0, "0/4": 0}

    print(f"{_CYAN}{_BOLD}{'=' * 125}{_RESET}")
    print(f"{_CYAN}{_BOLD}  BENCHMARK SUITE RUN — Evaluierung & Schwachstellen-Analyse{_RESET}")
    print(f"{_CYAN}{_BOLD}{'=' * 125}{_RESET}\n")

    fn_w = max(18, max(len(os.path.basename(p)) for p in unique_paths) + 2)
    header = (
        f"  {'#':<4} {'Datei':<{fn_w}} {'Status':<8} {'Ergebnis':<12} "
        f"{'Soll (GT)':<10} {'Match':<6} {'Methode':<16} {'OCR-Teil':<10} {'Dauer':<8} {'Fehlerkategorie':<20}"
    )
    print(f"{_BOLD}{header}{_RESET}")
    print(f"  {'─' * (len(header) - 2)}")

    for idx, path in enumerate(unique_paths, 1):
        filename = os.path.basename(path)
        source_dir = os.path.basename(os.path.dirname(path))

        image = cv2.imread(path)
        if image is None:
            logger.error(f"Bild konnte nicht geladen werden: {path}")
            continue

        t0 = time.time()
        cropped, yolo_conf = _detect_and_crop(model, image)
        scan_res = scanner.scan(cropped)
        duration_ms = int((time.time() - t0) * 1000)
        total_duration_ms += duration_ms

        is_success = scan_res.get("success", False)
        result_code = scan_res.get("result", "")
        method = scan_res.get("method", "Unbekannt")
        expected_code = gt_map.get(filename, "?")

        partial_text, ocr_chars = _analyze_ocr_partial(scan_res)
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
            else:
                fail_reason = _diagnose_failure_reason(scan_res, yolo_conf)
            failure_categories[fail_reason] = failure_categories.get(fail_reason, 0) + 1

        status_str = f"{_GREEN}OK{_RESET}" if is_success else f"{_RED}FAIL{_RESET}"
        res_str = f"{_GREEN}{result_code:<12}{_RESET}" if is_success else f"{_RED}{result_code:<12}{_RESET}"
        
        if expected_code != "?":
            match_str = f"{_GREEN}✓{_RESET}" if is_match else f"{_RED}✗{_RESET}"
        else:
            match_str = f"{_YELLOW}—{_RESET}"

        line = (
            f"  {idx:<4} {filename:<{fn_w}} {status_str:<17} {res_str} "
            f"{expected_code:<10} {match_str:<15} {method:<16} {partial_display:<10} {duration_ms:<8} {fail_reason:<20}"
        )
        print(line)

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

    avg_duration = total_duration_ms / total_images if total_images > 0 else 0
    success_pct = (success_count / total_images) * 100 if total_images > 0 else 0
    accuracy_pct = (accuracy_count / total_gt_count) * 100 if total_gt_count > 0 else 0

    print(f"\n{_BOLD}{'═' * 80}{_RESET}")
    print(f"  {_BOLD}BENCHMARK ERGEBNIS-ZUSAMMENFASSUNG{_RESET}")
    print(f"{_BOLD}{'═' * 80}{_RESET}")
    print(f"  Evaluierte Bilder gesamt: {_BOLD}{total_images}{_RESET}")
    print(f"  Erfolgreiche Scans:        {_GREEN}{success_count}/{total_images} ({success_pct:.1f}%){_RESET}")
    print(f"  Genauigkeit vs. GT:        {_GREEN}{accuracy_count}/{total_gt_count} ({accuracy_pct:.1f}%){_RESET}")
    print(f"  Durchschnitts-Laufzeit:    {_CYAN}{avg_duration:.0f} ms{_RESET} pro Bild")

    print(f"\n  {_BOLD}OCR-Teilerkennungs-Rate (Zeichen-Erfassung):{_RESET}")
    for k_chars, c_count in sorted(ocr_partial_breakdown.items(), key=lambda x: x[0], reverse=True):
        pct = (c_count / total_images) * 100
        print(f"    - {k_chars} Zeichen erkannt: {c_count:>3}× ({pct:.1f}%)")

    print(f"\n  {_BOLD}Methoden-Verteilung (bei Erfolg):{_RESET}")
    for m_name, count in sorted(method_distribution.items(), key=lambda x: -x[1]):
        pct = (count / total_images) * 100
        print(f"    - {m_name:<20} {count:>3}× ({pct:.1f}%)")

    print(f"\n  {_BOLD}Fehlerursachen-Aufschlüsselung (bei Fail):{_RESET}")
    if failure_categories:
        for f_reason, count in sorted(failure_categories.items(), key=lambda x: -x[1]):
            pct = (count / total_images) * 100
            print(f"    - {_RED}{f_reason:<25}{_RESET} {count:>3}× ({pct:.1f}%)")
    else:
        print(f"    {_GREEN}Keine Fehler aufgetreten! (100% Erkennung){_RESET}")

    print(f"{_BOLD}{'═' * 80}{_RESET}\n")

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

    report_abs_path = output_report_path if os.path.isabs(output_report_path) else os.path.join(app_dir, output_report_path)
    with open(report_abs_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Bericht erfolgreich gespeichert in: {report_abs_path}")

    # 6. Optionaler Baseline-Vergleich
    if compare_baseline_path:
        comp_abs_path = compare_baseline_path if os.path.isabs(compare_baseline_path) else os.path.join(app_dir, compare_baseline_path)
        _compare_with_baseline(summary_data, results, comp_abs_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Unified Benchmark Suite für DataDetector.")
    parser.add_argument("--output", default="benchmark_baseline.json", help="Pfad für JSON Report Output")
    parser.add_argument("--compare", default=None, help="Pfad zur Baseline JSON für Vorher-/Nachher-Vergleich")
    parser.add_argument("--dirs", nargs="*", help="Benutzerdefinierte Bildverzeichnisse")
    args = parser.parse_args()

    run_benchmark(
        custom_dirs=args.dirs,
        output_report_path=args.output,
        compare_baseline_path=args.compare
    )
