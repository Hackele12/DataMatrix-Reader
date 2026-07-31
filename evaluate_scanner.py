"""
evaluate_scanner.py — Offline-Evaluierungsskript für den Horden-Scanner

Liest Bilder aus dem Ordner 'training_data', schneidet per YOLO das Etikett zu
und führt die Scan-Pipeline von scanner.py aus. Gibt eine formatierte Ergebnistabelle
auf der Konsole aus und speichert einen detaillierten Bericht als JSON.

Nutzung:
    .venv/Scripts/python.exe evaluate_scanner.py
    .venv/Scripts/python.exe evaluate_scanner.py --images-dir mein_ordner
    .venv/Scripts/python.exe evaluate_scanner.py --ground-truth ground_truth.json
"""

import os
import sys
import json
import time
import glob
import logging
from datetime import datetime

# --- Umgebungsvariablen VOR allen anderen Imports ---
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

# --- Scanner Modul laden ---
try:
    import scanner
except Exception as e:
    logger.error(f"Scanner Importfehler: {e}")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
#  YOLO-Modell laden (gleiche Logik wie vision_app.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_yolo_model():
    """
    Lädt das YOLO-Modell.
    Sucht zuerst nach dem trainierten Modell, dann nach dem Standardmodell.

    Returns:
        YOLO-Modell-Instanz oder None bei Fehler.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error(
            "ultralytics konnte nicht importiert werden. "
            "Bitte installieren: pip install ultralytics"
        )
        return None

    app_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Trainiertes Modell suchen
    trained_path = os.path.join(
        app_dir, "runs", "detect", "training_runs",
        "horde_model", "weights", "best.pt",
    )
    if os.path.exists(trained_path):
        logger.info(f"Lade trainiertes YOLO-Modell: {trained_path}")
        return YOLO(trained_path)

    # 2. Fallback: Standard-Modell
    base_path = os.path.join(app_dir, "yolov10n.pt")
    if os.path.exists(base_path):
        logger.warning(
            f"Kein trainiertes Modell gefunden, nutze Standard: {base_path}"
        )
        return YOLO(base_path)

    logger.error("Kein YOLO-Modell gefunden (weder trainiert noch Standard).")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  YOLO-Inferenz & Zuschnitt
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_and_crop(model, image: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Führt YOLO-Inferenz durch und schneidet das erkannte Etikett zu.

    Args:
        model: Das geladene YOLO-Modell.
        image: Das Eingangsbild (BGR).

    Returns:
        (zugeschnittenes Bild, YOLO-Konfidenz).
        Falls kein Etikett erkannt: (Originalbild, 0.0).
    """
    results = model.predict(image, conf=0.15, verbose=False)
    boxes = results[0].boxes

    if len(boxes) == 0:
        return image, 0.0

    # Beste Box (höchste Konfidenz)
    best_box = boxes[0]
    conf = float(best_box.conf[0])
    x1, y1, x2, y2 = map(int, best_box.xyxy[0])

    # Mit Begradigung (Deskewing) ausschneiden
    cropped = scanner.deskew_crop(image, (x1, y1, x2, y2), padding=60)
    return cropped, conf


# ═══════════════════════════════════════════════════════════════════════════════
#  Einzelnes Bild scannen
# ═══════════════════════════════════════════════════════════════════════════════

def _evaluate_single_image(
    model, image_path: str,
) -> dict:
    """
    Lädt ein Bild, erkennt das Etikett per YOLO und scannt es.

    Returns:
        dict mit allen Ergebnis-Feldern.
    """
    filename = os.path.basename(image_path)

    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Bild konnte nicht geladen werden: {image_path}")
        return {
            "filename": filename,
            "success": False,
            "result": "LADEFEHLER",
            "method": "Fehler",
            "confidence": 0.0,
            "yolo_conf": 0.0,
            "duration_ms": 0,
            "ocr_result": None,
            "dmtx_result": None,
            "verified": False,
        }

    t0 = time.time()

    # YOLO-Zuschnitt
    cropped, yolo_conf = _detect_and_crop(model, image)
    crop_h, crop_w = cropped.shape[:2]

    if yolo_conf > 0:
        logger.info(
            f"[{filename}] Etikett erkannt (Konfidenz: {yolo_conf:.2f}), "
            f"Zuschnitt: {crop_w}x{crop_h}"
        )
    else:
        logger.warning(f"[{filename}] Kein Etikett erkannt, scanne gesamtes Bild.")

    # Scanner-Pipeline
    scan_result = scanner.scan(cropped)
    duration_ms = int((time.time() - t0) * 1000)

    return {
        "filename": filename,
        "success": scan_result.get("success", False),
        "result": scan_result.get("result", ""),
        "method": scan_result.get("method", ""),
        "confidence": scan_result.get("confidence", 0.0),
        "yolo_conf": yolo_conf,
        "duration_ms": duration_ms,
        "ocr_result": scan_result.get("ocr_result"),
        "dmtx_result": scan_result.get("dmtx_result"),
        "verified": scan_result.get("verified", False),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Konsolentabelle ausgeben
# ═══════════════════════════════════════════════════════════════════════════════

# ANSI-Farbcodes
_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _print_results_table(
    results: list[dict],
    ground_truth: dict | None,
):
    """Gibt eine formatierte Ergebnistabelle auf der Konsole aus."""

    has_gt = ground_truth is not None and len(ground_truth) > 0

    # Kopfzeile
    print()
    print(f"{_BOLD}{'═' * 100}{_RESET}")
    print(f"{_BOLD}  EVALUIERUNGSBERICHT — Horden-Scanner{_RESET}")
    print(f"{_BOLD}{'═' * 100}{_RESET}")
    print()

    # Spaltenbreiten
    fn_w = max(len(r["filename"]) for r in results) + 2
    header = (
        f"  {'Datei':<{fn_w}} {'Erfolg':<8} {'Ergebnis':<14} "
        f"{'Methode':<16} {'Konf.':<8} {'YOLO':<8} {'Dauer':<8}"
    )
    if has_gt:
        header += f" {'Soll':<8} {'Match':<6}"
    print(f"{_BOLD}{header}{_RESET}")
    print(f"  {'─' * (len(header) - 2)}")

    correct = 0
    total_with_gt = 0
    success_count = 0
    total_duration = 0
    method_counts: dict[str, int] = {}

    for r in results:
        fn = r["filename"]
        ok = r["success"]
        res = r["result"] if r["result"] else "—"
        method = r["method"] if r["method"] else "—"
        conf = r["confidence"]
        yolo = r["yolo_conf"]
        dur = r["duration_ms"]

        # Statistiken sammeln
        if ok:
            success_count += 1
        total_duration += dur
        method_counts[method] = method_counts.get(method, 0) + 1

        # Farbcodierung
        ok_str = f"{_GREEN}✓ Ja{_RESET}" if ok else f"{_RED}✗ Nein{_RESET}"
        res_str = f"{_GREEN}{res:<14}{_RESET}" if ok else f"{_RED}{res:<14}{_RESET}"
        yolo_str = f"{yolo:.2f}" if yolo > 0 else f"{_YELLOW}—{_RESET}    "

        line = (
            f"  {fn:<{fn_w}} {ok_str:<17} {res_str} "
            f"{method:<16} {conf:<8.2f} {yolo_str:<8} {dur:<8}"
        )

        if has_gt:
            expected = ground_truth.get(fn, "?")
            total_with_gt += 1 if expected != "?" else 0
            is_match = ok and r["result"] == expected
            if is_match:
                correct += 1
            match_str = (
                f"{_GREEN}✓{_RESET}" if is_match
                else f"{_RED}✗{_RESET}" if expected != "?" and ok
                else f"{_YELLOW}—{_RESET}"
            )
            line += f" {expected:<8} {match_str}"

        print(line)

    # Zusammenfassung
    print()
    print(f"{_BOLD}{'─' * 60}{_RESET}")
    total = len(results)
    avg_dur = total_duration / total if total > 0 else 0

    print(f"  {_BOLD}Gesamt:{_RESET}          {total} Bilder")
    print(
        f"  {_BOLD}Erfolgsrate:{_RESET}     "
        f"{_GREEN if success_count == total else _YELLOW}"
        f"{success_count}/{total} ({100 * success_count / total:.0f}%){_RESET}"
    )

    if has_gt and total_with_gt > 0:
        acc = 100 * correct / total_with_gt
        color = _GREEN if acc == 100 else _YELLOW if acc >= 80 else _RED
        print(
            f"  {_BOLD}Genauigkeit:{_RESET}     "
            f"{color}{correct}/{total_with_gt} ({acc:.0f}%){_RESET}"
        )

    print(f"  {_BOLD}Ø Dauer:{_RESET}         {avg_dur:.0f} ms pro Bild")

    # Methodenverteilung
    print(f"\n  {_BOLD}Methodenverteilung:{_RESET}")
    for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total
        print(f"    {method:<20} {count:>3}× ({pct:.0f}%)")

    print(f"\n{_BOLD}{'═' * 100}{_RESET}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Bericht speichern
# ═══════════════════════════════════════════════════════════════════════════════

def _save_report(results: list[dict], ground_truth: dict | None, report_path: str):
    """Speichert den detaillierten Evaluierungsbericht als JSON."""
    total = len(results)
    success_count = sum(1 for r in results if r["success"])
    total_duration = sum(r["duration_ms"] for r in results)

    correct = 0
    total_with_gt = 0
    if ground_truth:
        for r in results:
            expected = ground_truth.get(r["filename"])
            if expected:
                total_with_gt += 1
                if r["success"] and r["result"] == expected:
                    correct += 1

    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_images": total,
            "successful_scans": success_count,
            "success_rate_pct": round(100 * success_count / total, 1) if total > 0 else 0,
            "accuracy_pct": round(100 * correct / total_with_gt, 1) if total_with_gt > 0 else None,
            "correct_of_labeled": f"{correct}/{total_with_gt}" if total_with_gt > 0 else None,
            "avg_duration_ms": round(total_duration / total, 1) if total > 0 else 0,
        },
        "details": results,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"Evaluierungsbericht gespeichert: {report_path}")


def _save_ground_truth_template(results: list[dict], template_path: str):
    """
    Erstellt ein Ground-Truth-Template mit den aktuell erkannten Codes.
    Der Nutzer kann dieses Template korrigieren und als ground_truth.json verwenden.
    """
    template = {}
    for r in results:
        if r["success"]:
            template[r["filename"]] = r["result"]
        else:
            template[r["filename"]] = "MANUELL_EINTRAGEN"

    with open(template_path, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Ground-Truth-Template erstellt: {template_path}\n"
        f"  → Bitte korrigiere falsche Einträge und benenne die Datei in "
        f"'ground_truth.json' um."
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Hauptprogramm
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Offline-Evaluierung des Horden-Scanners auf Testbildern."
    )
    parser.add_argument(
        "--images-dir",
        default="training_data",
        help="Ordner mit Testbildern (Standard: training_data)",
    )
    parser.add_argument(
        "--ground-truth",
        default="ground_truth.json",
        help="Pfad zur Ground-Truth-Datei (Standard: ground_truth.json)",
    )
    parser.add_argument(
        "--report",
        default="evaluation_report.json",
        help="Pfad für den Ausgabebericht (Standard: evaluation_report.json)",
    )
    args = parser.parse_args()

    app_dir = os.path.dirname(os.path.abspath(__file__))

    # Bilderordner auflösen
    images_dir = args.images_dir
    if not os.path.isabs(images_dir):
        images_dir = os.path.join(app_dir, images_dir)

    if not os.path.isdir(images_dir):
        logger.error(f"Bilderordner nicht gefunden: {images_dir}")
        sys.exit(1)

    # Bilder sammeln
    image_files = sorted(
        glob.glob(os.path.join(images_dir, "*.jpg"))
        + glob.glob(os.path.join(images_dir, "*.jpeg"))
        + glob.glob(os.path.join(images_dir, "*.png"))
    )

    if not image_files:
        logger.error(f"Keine Bilder gefunden in: {images_dir}")
        sys.exit(1)

    logger.info(f"{len(image_files)} Bilder gefunden in: {images_dir}")

    # Ground Truth laden (optional)
    gt_path = args.ground_truth
    if not os.path.isabs(gt_path):
        gt_path = os.path.join(app_dir, gt_path)

    ground_truth = None
    if os.path.exists(gt_path):
        try:
            with open(gt_path, "r", encoding="utf-8") as f:
                ground_truth = json.load(f)
            logger.info(
                f"Ground Truth geladen: {gt_path} "
                f"({len(ground_truth)} Einträge)"
            )
        except Exception as e:
            logger.warning(f"Ground Truth konnte nicht geladen werden: {e}")
    else:
        logger.info(
            f"Keine Ground-Truth-Datei gefunden ({gt_path}). "
            f"Template wird nach dem Scan erstellt."
        )

    # YOLO-Modell laden
    model = _load_yolo_model()
    if model is None:
        logger.error("YOLO-Modell konnte nicht geladen werden. Abbruch.")
        sys.exit(1)

    # Historie zurücksetzen vor Beginn der Evaluation
    if hasattr(scanner, "_recent_scans"):
        scanner._recent_scans = []

    # Alle Bilder evaluieren
    print(f"\n{_CYAN}{_BOLD}Starte Evaluierung (Pass 1) von {len(image_files)} Bildern...{_RESET}\n")

    results = []
    for i, image_path in enumerate(image_files, 1):
        fn = os.path.basename(image_path)
        print(f"  [Pass 1][{i:>2}/{len(image_files)}] {fn} ... ", end="", flush=True)

        result = _evaluate_single_image(model, image_path)
        results.append(result)

        # Fortschrittsanzeige
        if result["success"]:
            print(f"{_GREEN}✓ {result['result']}{_RESET} ({result['method']}, {result['duration_ms']}ms)")
        else:
            print(f"{_RED}✗ {result['result']}{_RESET} ({result['duration_ms']}ms)")

    # Pass 2: Re-scan any images that failed or were not verified
    print(f"\n{_CYAN}{_BOLD}Starte Pass 2 (Korrektur von Fehlern / nicht verifizierten Scans)...{_RESET}\n")
    for i, image_path in enumerate(image_files):
        result = results[i]
        if not result["success"] or result["method"] != "Verifiziert":
            fn = os.path.basename(image_path)
            print(f"  [Pass 2][{i+1:>2}/{len(image_files)}] {fn} (Vorher: {result['result']} - {result['method']}) ... ", end="", flush=True)
            
            # Re-evaluate
            new_result = _evaluate_single_image(model, image_path)
            
            # Overwrite if successful and improved
            if new_result["success"]:
                results[i] = new_result
                print(f"{_GREEN}✓ {new_result['result']}{_RESET} ({new_result['method']}, {new_result['duration_ms']}ms)")
            else:
                print(f"{_RED}✗ {new_result['result']}{_RESET} ({new_result['duration_ms']}ms)")

    # Ergebnistabelle ausgeben
    _print_results_table(results, ground_truth)

    # Bericht speichern
    report_path = args.report
    if not os.path.isabs(report_path):
        report_path = os.path.join(app_dir, report_path)
    _save_report(results, ground_truth, report_path)

    # Ground-Truth-Template erstellen falls noch keines existiert
    if ground_truth is None:
        template_path = os.path.join(app_dir, "ground_truth_template.json")
        _save_ground_truth_template(results, template_path)


if __name__ == "__main__":
    main()
