"""
benchmark_gui.py — Benchmark & Ground-Truth-Tool für DataDetector (helles Design wie Log Analyzer und DataMatrixReader).

Reiter:
  Übersicht     Kennzahlen und Kachelansicht der Testbilder (Klick öffnet das Bild im Ground-Truth-Editor)
  Ground Truth  Code je Bild erfassen: Zoom, Format-Prüfung, Filter „Offen“ / „Fehler“, Tastaturbedienung
  Benchmark     Lauf starten oder abbrechen (alle Bilder oder nur die Fehler des angezeigten Laufs), Live-Protokoll
  Ergebnisse    Kennzahlen, Tabelle mit Sortierung und Filter, Vergleich mit einem früheren Lauf, CSV-Export
  Detail        Pipeline je Bild: YOLO-Boxen, Crops, Zwischenergebnisse

Die Seitenleiste listet die gespeicherten Läufe (benchmark_reports/); ein Klick zeigt den Lauf in Ergebnisse und Detail.
Der Benchmark nutzt dieselbe Pipeline wie die Produktion: YOLO-Detektion → scanner.scan_2class().
Ohne Oberfläche: `benchmark_gui.py --headless [--fail-on-regression]`; anderer Bildordner: --images <Ordner> --gt <datei.json>.
"""

import argparse
import csv
import glob
import json
import logging
import math
import os
import queue
import re
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

# --- KMP Fix ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageDraw, ImageTk

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# --- Scanner & YOLO ---
try:
    import scanner
    import yolo_detector
    from scanner.config import HORDEN_PATTERN
except Exception as e:
    logger.error(f"Scanner Importfehler: {e}")
    sys.exit(1)

# --- Pfade ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(APP_DIR, "training_data")
GROUND_TRUTH_PATH = os.path.join(APP_DIR, "ground_truth.json")
REPORTS_DIR = os.path.join(APP_DIR, "benchmark_reports")
BASELINE_PATH = os.path.join(APP_DIR, "benchmark_baseline.json")
BENCHMARK_VERSION = "2.3.0"
IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
MAX_LISTED_REPORTS = 30

# --- UI Styling (Light Theme, identisch zu Log Analyzer und DataMatrixReader) ---
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

ACCENT       = "#2563EB"
ACCENT_HOVER = "#1D4ED8"
SUCCESS      = "#16A34A"
SUCCESS_HOVER = "#15803D"
WARN         = "#D97706"
DANGER       = "#DC2626"
DANGER_HOVER = "#991B1B"
BG_SIDE      = "#F1F5F9"
BG_CARD      = "#FFFFFF"
BG_MAIN      = "#E2E8F0"
BORDER       = "#CBD5E1"
TXT_DARK     = "#1E293B"
TXT_MID      = "#475569"
TXT_LIGHT    = "#94A3B8"
HOVER_ROW    = "#F1F5F9"
SELECT_ROW   = "#DBEAFE"
HOVER_CARD   = "#F8FAFC"
ROW_REGRESSION = "#FEE2E2"
ROW_IMPROVED   = "#DCFCE7"

OUTCOME_COLORS = {"ok": SUCCESS, "wrong": WARN, "fail": DANGER}
LEVEL_COLORS = {"ok": SUCCESS, "warn": WARN, "fail": DANGER, "muted": TXT_LIGHT}
LEVEL_ICONS = {"ok": "✓", "warn": "!", "fail": "✗", "muted": "–"}
METHOD_COLORS = {"Verifiziert": SUCCESS, "Modulabgleich": "#0891B2", "OCR": ACCENT, "Rekonstruiert": WARN,
                 "Fehler": DANGER}
MENU_STYLE = {"fg_color": BG_SIDE, "text_color": TXT_DARK, "button_color": BORDER, "button_hover_color": TXT_LIGHT}
RESULT_FILTERS = ("Alle", "Nicht gelesen", "Falsch gelesen", "Korrekt / gelesen", "Ohne Ground Truth",
                  "Änderungen zum Vergleich")


# ═══════════════════════════════════════════════════════════════════════════════
#  Hilfsfunktionen
# ═══════════════════════════════════════════════════════════════════════════════

def _natural_key(text: str) -> list:
    """Sortierschlüssel mit Zahlen als Zahlen (2.jpg vor 10.jpg); Text und Zahl wechseln sich strikt ab."""
    return [int(part) if index % 2 else part.lower() for index, part in enumerate(re.split(r"(\d+)", text))]


def _fmt_int(value: float) -> str:
    return f"{int(value):,}".replace(",", ".")


def _fmt_pct(value: float) -> str:
    return f"{value:.1f}".replace(".", ",") + " %"


def _fmt_delta(value: float, unit: str = "", decimals: int = 1) -> str:
    """Vorzeichenbehaftete Differenz mit Komma und Tausenderpunkt, z. B. „+2,9 %-Pkt.“."""
    value = round(value, decimals) or 0.0  # kein „−0,0“
    text = f"{value:+,.{decimals}f}".translate(str.maketrans(",.", ".,"))
    return text.replace("-", "−") + unit


def _fmt_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds} s" if seconds < 60 else f"{seconds // 60}:{seconds % 60:02d} min"


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_image(path: str) -> np.ndarray | None:
    """BGR-Bild laden; cv2.imread scheitert unter Windows an Sonderzeichen im Pfad, daher Fallback über imdecode."""
    image = cv2.imread(path)
    if image is None:
        try:
            data = np.fromfile(path, dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
        except OSError:
            image = None
    return image


def _write_json_atomic(path: str, data, **dump_args):
    """Schreibt über eine Temp-Datei, damit ein Abbruch nie eine halbe Datei hinterlässt."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, **dump_args)
    os.replace(tmp, path)


# ═══════════════════════════════════════════════════════════════════════════════
#  Ground-Truth & Testbilder
# ═══════════════════════════════════════════════════════════════════════════════

def load_ground_truth(path: str) -> dict:
    """Lädt die Ground-Truth-Datei (Dateiname → Code); eine defekte Datei wird beiseitegelegt statt überschrieben."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("kein JSON-Objekt")
    except ValueError as e:
        backup = f"{path}.defekt-{datetime.now():%Y%m%d-%H%M%S}"
        logger.error(f"Ground Truth ist defekt ({e}); Datei nach {backup} verschoben.")
        try:
            os.replace(path, backup)
        except OSError as move_error:
            logger.error(f"Defekte Ground Truth konnte nicht verschoben werden: {move_error}")
        return {}
    except OSError as e:
        logger.warning(f"Fehler beim Laden der Ground Truth: {e}")
        return {}
    return {str(name): str(code) for name, code in data.items() if code}


def save_ground_truth(gt_map: dict, path: str):
    """Speichert die Ground-Truth-Datei (sortiert nach Dateiname)."""
    ordered = dict(sorted(gt_map.items(), key=lambda item: _natural_key(item[0])))
    _write_json_atomic(path, ordered)
    logger.info(f"Ground Truth gespeichert: {len(gt_map)} Einträge.")


def load_image_list(image_dir: str) -> list[str]:
    """Alle Bilder (jpg, jpeg, png, bmp) des Ordners, natürlich sortiert."""
    files = {}
    for pattern in IMAGE_PATTERNS:
        for path in glob.glob(os.path.join(glob.escape(image_dir), pattern)):
            files[os.path.normcase(path)] = path
    return sorted(files.values(), key=lambda p: _natural_key(os.path.basename(p)))


# ═══════════════════════════════════════════════════════════════════════════════
#  Benchmark-Kern (GUI und --headless)
# ═══════════════════════════════════════════════════════════════════════════════

def outcome(entry: dict) -> str:
    """„ok“ = gelesen (mit Ground Truth: richtig), „wrong“ = falsch gelesen, „fail“ = nicht gelesen."""
    if not entry.get("success"):
        return "fail"
    if entry.get("expected") and not entry.get("is_match"):
        return "wrong"
    return "ok"


def display_code(entry: dict) -> str:
    return entry["result_code"] if entry["success"] else "FAIL"


def _failure_reason(detections: list[dict], scan_res: dict) -> str:
    if not detections:
        return "YOLO_NO_DETECTION"
    message = str(scan_res.get("result") or "")
    if message.startswith("Widerspruch"):
        return "CROSSVAL_CONFLICT"
    if message.startswith("Unsicheres Ergebnis"):
        return "CROSSVAL_UNSURE"
    if not scan_res.get("dmtx_result") and not scan_res.get("ocr_result"):
        return "DMTX_AND_OCR_FAILED"
    if scan_res.get("ocr_result") and not scan_res.get("dmtx_result"):
        return "RECONSTRUCTION_REJECTED"
    return "UNKNOWN"


def _failed_entry(base: dict, expected: str | None, reason: str, error: str, duration_ms: int = 0) -> dict:
    """Ergebnis für ein Bild, das nicht ausgewertet werden konnte (unlesbar oder Ausnahme in der Pipeline)."""
    return {
        **base, "success": False, "result_code": "", "expected": expected, "is_match": False,
        "method": "Fehler", "confidence": 0.0, "duration_ms": duration_ms, "fail_reason": reason,
        "detections": [], "ocr_result": None, "dmtx_result": None, "ocr_partial": None,
        "error": error, "scan_res": {},
    }


def benchmark_image(model, path: str, gt_map: dict, index: int) -> dict:
    """Ein Bild durch YOLO + scanner.scan_2class(); Fehler einzelner Bilder brechen den Lauf nicht ab."""
    fname = os.path.basename(path)
    expected = gt_map.get(fname, None)
    base = {"index": index, "filename": fname, "path": path}

    image = read_image(path)
    if image is None:
        return _failed_entry(base, expected, "IMAGE_UNREADABLE", "Bild konnte nicht gelesen werden")

    t0 = time.time()
    try:
        yolo_result = model.predict(image, conf=yolo_detector.PREDICT_CONF, verbose=False)[0]
        detections = yolo_detector.extract_detections(yolo_result)
        scan_res = scanner.scan_2class(image, detections)
    except Exception as e:
        logger.exception(f"{fname}: Fehler in der Scan-Pipeline")
        return _failed_entry(base, expected, "EXCEPTION", f"{type(e).__name__}: {e}", int((time.time() - t0) * 1000))
    duration_ms = int((time.time() - t0) * 1000)

    is_success = scan_res.get("success", False)
    result_code = scan_res.get("result", "")
    return {
        **base,
        "success": is_success,
        "result_code": result_code,
        "expected": expected,
        "is_match": bool(expected) and is_success and result_code == expected,
        "method": scan_res.get("method", "Unbekannt"),
        "method_detail": scan_res.get("method_detail"),
        "confidence": scan_res.get("confidence", 0.0),
        "duration_ms": duration_ms,
        "fail_reason": None if is_success else _failure_reason(detections, scan_res),
        "detections": detections,
        "ocr_result": scan_res.get("ocr_result"),
        "dmtx_result": scan_res.get("dmtx_result"),
        "ocr_partial": scan_res.get("ocr_partial_display"),
        "scan_res": scan_res,
    }


def summarize(results: list[dict], total_images: int, total_time_s: float) -> dict:
    """Kennzahlen eines Benchmark-Laufs (Genauigkeit und Fehllesungen nur für Bilder mit Ground Truth)."""
    success_count = sum(1 for r in results if r["success"])
    with_gt = [r for r in results if r["expected"]]
    accuracy_count = sum(1 for r in with_gt if r["is_match"])
    method_dist = {}
    stage_dist = {}
    failure_cats = {}
    for r in results:
        if r["success"]:
            method_dist[r["method"]] = method_dist.get(r["method"], 0) + 1
            stage = r.get("method_detail") or r["method"]
            stage_dist[stage] = stage_dist.get(stage, 0) + 1
        else:
            reason = r["fail_reason"] or "UNKNOWN"
            failure_cats[reason] = failure_cats.get(reason, 0) + 1

    total_ms = sum(r["duration_ms"] for r in results)
    return {
        "total_images": total_images,
        "success_count": success_count,
        "success_rate_pct": round(success_count / total_images * 100, 1) if total_images > 0 else 0,
        "accuracy_count": accuracy_count,
        "accuracy_total_gt": len(with_gt),
        "accuracy_pct": round(accuracy_count / len(with_gt) * 100, 1) if with_gt else 0,
        "false_read_count": sum(1 for r in with_gt if r["success"] and not r["is_match"]),
        "no_read_count": len(results) - success_count,
        "avg_duration_ms": round(total_ms / total_images, 1) if total_images > 0 else 0,
        "total_time_s": round(total_time_s, 1),
        "method_distribution": method_dist,
        "stage_distribution": dict(sorted(stage_dist.items(), key=lambda item: -item[1])),
        "failure_categories": failure_cats,
    }


def format_result_line(entry: dict, total: int) -> str:
    kind = outcome(entry)
    icon = {"ok": "✓", "wrong": "⚠", "fail": "✗"}[kind]
    extra = ""
    if kind == "wrong":
        extra = f"  ≠ Soll {entry['expected']}"
    elif kind == "fail":
        extra = f"  [{entry['fail_reason']}]"
        if entry.get("error"):
            extra += f" {entry['error']}"
    return (
        f"  {icon} [{entry['index']:>3}/{total}] {entry['filename']:<14} "
        f"→ {display_code(entry):<8} "
        f"({entry['method']}, {entry['duration_ms']}ms){extra}"
    )


def format_summary_lines(summary: dict) -> list[str]:
    s = summary
    title = "BENCHMARK ABGEBROCHEN" if s.get("cancelled") else "BENCHMARK ABGESCHLOSSEN"
    return [
        f"\n{'═' * 60}",
        f"  {title}",
        f"{'═' * 60}",
        f"  Bilder:         {s['total_images']}",
        f"  Erfolgsrate:    {s['success_count']}/{s['total_images']} ({s['success_rate_pct']:.1f}%)",
        f"  Genauigkeit GT: {s['accuracy_count']}/{s['accuracy_total_gt']} ({s['accuracy_pct']:.1f}%)",
        f"  Falsch gelesen: {s['false_read_count']}",
        f"  Nicht gelesen:  {s['no_read_count']}",
        f"  Ø Dauer/Bild:   {s['avg_duration_ms']:.0f} ms",
        f"  Gesamtzeit:     {s['total_time_s']:.1f}s",
        *[f"  Stufe {count:>4}×  {stage}" for stage, count in s.get("stage_distribution", {}).items()],
        f"{'═' * 60}",
    ]


def run_benchmark(model, image_paths: list[str], gt_map: dict, log=print, on_progress=None,
                  should_stop=None, on_start=None) -> tuple[list, dict]:
    """
    Führt den Benchmark über alle Bilder aus.

    Args:
        log: Ausgabe für Protokollzeilen.
        on_progress: Optionaler Callback (bild_index, gesamt, ergebnisse, laufzeit_s) nach jedem Bild.
        should_stop: Optionaler Callback; True beendet den Lauf vor dem nächsten Bild (summary["cancelled"]).
        on_start: Optionaler Callback (bild_index, gesamt, pfad) vor jedem Bild.
    """
    total = len(image_paths)
    results = []
    cancelled = False
    log(f"\n▶ Benchmark gestartet: {total} Bilder\n{'─' * 60}")
    start_time = time.time()

    for idx, path in enumerate(image_paths):
        if should_stop and should_stop():
            cancelled = True
            break
        if on_start:
            on_start(idx, total, path)
        entry = benchmark_image(model, path, gt_map, idx + 1)
        results.append(entry)
        log(format_result_line(entry, total))
        if on_progress:
            on_progress(idx, total, results, time.time() - start_time)

    summary = summarize(results, len(results), time.time() - start_time)
    summary["cancelled"] = cancelled
    return results, summary


def save_report(results: list[dict], summary: dict, image_dir: str, gt_path: str) -> str:
    """Schreibt den Bericht nach benchmark_reports/ (mit Datum, Zeit & Version) und als benchmark_baseline.json."""
    now_dt = datetime.now()
    report = {
        "version": BENCHMARK_VERSION,
        "timestamp": now_dt.isoformat(),
        "image_dir": image_dir,
        "ground_truth_file": gt_path,
        "summary": {k: v for k, v in summary.items() if k != "cancelled"},
        "details": [{k: v for k, v in r.items() if k != "scan_res"} for r in results],
    }
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, f"benchmark_{now_dt.strftime('%Y-%m-%d_%H-%M-%S')}_v{BENCHMARK_VERSION}.json")
    targets = [report_path]
    if os.path.normcase(os.path.abspath(image_dir)) == os.path.normcase(IMAGE_DIR):
        targets.append(BASELINE_PATH)  # Baseline nur für den Standard-Bildsatz
    for path in targets:
        _write_json_atomic(path, report, default=str)
    return report_path


# ═══════════════════════════════════════════════════════════════════════════════
#  Berichte laden, Läufe vergleichen, CSV-Export
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_entry(raw: dict, number: int, image_dir: str | None) -> dict:
    """Bericht-Eintrag mit festen Typen und Standardwerten (ältere Berichte kennen nicht alle Felder)."""
    path = str(raw.get("path") or "")
    filename = str(raw.get("filename") or os.path.basename(path))
    if image_dir and not os.path.isfile(path):
        moved = os.path.join(image_dir, filename)  # Projektordner wurde verschoben
        if os.path.isfile(moved):
            path = moved
    detections = []
    for det in raw.get("detections") or []:
        try:
            box = tuple(int(v) for v in det["box"])
            if len(box) == 4:
                detections.append({"cls": int(det.get("cls", 0)), "conf": _as_float(det.get("conf")), "box": box})
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
    expected = raw.get("expected")
    return {
        "index": int(_as_float(raw.get("index"), number)),
        "filename": filename,
        "path": path,
        "success": bool(raw.get("success")),
        "result_code": str(raw.get("result_code") or ""),
        "expected": str(expected) if expected else None,
        "is_match": bool(raw.get("is_match")),
        "method": str(raw.get("method") or "Unbekannt"),
        "method_detail": raw.get("method_detail") or None,
        "confidence": _as_float(raw.get("confidence")),
        "duration_ms": int(_as_float(raw.get("duration_ms"))),
        "fail_reason": raw.get("fail_reason") or None,
        "detections": detections,
        "ocr_result": raw.get("ocr_result") or None,
        "dmtx_result": raw.get("dmtx_result") or None,
        "ocr_partial": raw.get("ocr_partial") or None,
        "error": raw.get("error") or None,
    }


def read_report(path: str, image_dir: str | None = None) -> dict:
    """Lädt einen Bericht; die Kennzahlen werden aus den Einzelergebnissen neu berechnet (einheitlich über alle Versionen)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("details"), list):
        raise ValueError("Kein Benchmark-Bericht (Feld „details“ fehlt).")
    results = [_normalize_entry(raw, number, image_dir)
               for number, raw in enumerate(data["details"], 1) if isinstance(raw, dict)]
    stored = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    total_time = _as_float(stored.get("total_time_s"), sum(r["duration_ms"] for r in results) / 1000)
    try:
        timestamp = datetime.fromisoformat(data["timestamp"])
    except (KeyError, TypeError, ValueError):
        timestamp = datetime.fromtimestamp(os.path.getmtime(path))
    return {
        "path": path,
        "version": str(data.get("version") or "?"),
        "timestamp": timestamp,
        "image_dir": str(data.get("image_dir") or ""),
        "results": results,
        "summary": summarize(results, len(results), total_time),
    }


def list_reports(image_dir: str | None = None, cache: dict | None = None) -> list[dict]:
    """Berichte aus benchmark_reports/, neueste zuerst; defekte Dateien werden übersprungen."""
    cache = cache if cache is not None else {}
    reports = []
    for path in glob.glob(os.path.join(glob.escape(REPORTS_DIR), "benchmark_*.json")):
        try:
            mtime = os.path.getmtime(path)
            cached = cache.get(path)
            if cached and cached[0] == mtime:
                reports.append(cached[1])
                continue
            report = read_report(path, image_dir)
            cache[path] = (mtime, report)
            reports.append(report)
        except Exception as e:  # eine defekte Datei darf die Liste nicht verhindern
            logger.warning(f"Bericht übersprungen ({os.path.basename(path)}): {e}")
    reports.sort(key=lambda r: r["timestamp"], reverse=True)
    return reports[:MAX_LISTED_REPORTS]


def report_title(report: dict) -> str:
    return report["timestamp"].strftime("%d.%m.%Y  %H:%M:%S")


def report_fits(report: dict, names: set[str]) -> bool:
    """Bericht gehört zu diesen Testbildern (mindestens die Hälfte der Dateinamen stimmt überein)."""
    hits = sum(1 for r in report["results"] if r["filename"] in names)
    return bool(names) and bool(report["results"]) and hits * 2 >= len(report["results"])


def compare_runs(current: list[dict], previous: list[dict]) -> dict[str, str]:
    """
    Vergleicht zwei Läufe je Dateiname → {Datei: "regression" | "improved" | "changed" | "new"}.

    regression = vorher korrekt, jetzt nicht; improved = umgekehrt; changed = beide gelesen, aber anderer Code.
    Unveränderte Bilder fehlen im Ergebnis.
    """
    before = {r["filename"]: r for r in previous}
    diff = {}
    for r in current:
        old = before.get(r["filename"])
        if old is None:
            diff[r["filename"]] = "new"
            continue
        now_ok, was_ok = outcome(r) == "ok", outcome(old) == "ok"
        if was_ok and not now_ok:
            diff[r["filename"]] = "regression"
        elif now_ok and not was_ok:
            diff[r["filename"]] = "improved"
        elif r["success"] and old["success"] and r["result_code"] != old["result_code"]:
            diff[r["filename"]] = "changed"
    return diff


DIFF_TEXT = {"regression": "▼ Regression", "improved": "▲ verbessert", "changed": "↔ Code geändert", "new": "neu"}


def reevaluate(results: list[dict], gt_map: dict) -> list[dict]:
    """Bewertet die Lesungen eines Laufs gegen die aktuelle Ground Truth neu (ohne die Pipeline erneut auszuführen)."""
    rescored = []
    for r in results:
        expected = gt_map.get(r["filename"])
        rescored.append({**r, "expected": expected,
                         "is_match": bool(expected) and r["success"] and r["result_code"] == expected})
    return rescored


def format_compare_lines(diff: dict[str, str], current: list[dict], previous: list[dict], label: str, limit: int = 15) -> list[str]:
    """Textbericht des Vergleichs für die Konsole (--headless)."""
    counts = Counter(diff.values())
    lines = [
        f"\nVergleich mit Lauf {label}:",
        f"  Regressionen: {counts['regression']}   Verbesserungen: {counts['improved']}   "
        f"Geänderte Codes: {counts['changed']}",
    ]
    old = {r["filename"]: r for r in previous}
    now = {r["filename"]: r for r in current}
    shown = 0
    for name, kind in diff.items():
        if kind == "new" or shown >= limit:
            continue
        shown += 1
        wrong = f"  (Soll {now[name]['expected']})" if outcome(now[name]) == "wrong" else ""
        lines.append(f"  {DIFF_TEXT[kind]:<16} {name:<14} {display_code(old[name])} → {display_code(now[name])}{wrong}")
    return lines


CSV_COLUMNS = ("Nr", "Datei", "Status", "Code", "Soll", "Treffer", "Methode", "Konfidenz", "Dauer_ms",
               "Fehlerursache", "DMTX", "OCR", "Fehlermeldung")
STATUS_TEXT = {"ok": "Korrekt", "wrong": "Falsch gelesen", "fail": "Nicht gelesen"}


def status_text(entry: dict) -> str:
    kind = outcome(entry)
    return "Gelesen" if kind == "ok" and not entry.get("expected") else STATUS_TEXT[kind]


def export_results_csv(results: list[dict], path: str):
    """Ergebnistabelle als CSV (Semikolon, UTF-8 mit BOM → öffnet direkt in Excel)."""
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(CSV_COLUMNS)
        for r in results:
            match = "" if not r["expected"] else ("JA" if r["is_match"] else "NEIN")
            writer.writerow([
                r["index"], r["filename"], status_text(r), r["result_code"] if r["success"] else "",
                r["expected"] or "", match, r["method"], f"{r['confidence'] * 100:.1f}%", r["duration_ms"],
                r["fail_reason"] or "", r["dmtx_result"] or "", r["ocr_result"] or "", r.get("error") or "",
            ])


def pipeline_steps(r: dict) -> list[tuple[str, str, str]]:
    """Pipeline-Schritte eines Ergebnisses → [(Schritt, Ergebnis, Stufe)] mit Stufe ok / warn / fail / muted."""
    n = len(r.get("detections") or [])
    steps = [("YOLO-Detektion (Etikett finden)",
              f"{n} Box{'en' if n != 1 else ''} gefunden" if n else "Keine Box erkannt", "ok" if n else "fail")]

    dmtx = r.get("dmtx_result")
    steps.append(("DataMatrix-Dekodierung (zxing / pylibdmtx)",
                  f"Erkannt: {dmtx}" if dmtx else "Nicht dekodierbar", "ok" if dmtx else "fail"))

    ocr, partial = r.get("ocr_result"), r.get("ocr_partial")
    if ocr:
        steps.append(("OCR / Klarschrift-Erkennung", f"Erkannt: {ocr}", "ok"))
    elif partial:
        steps.append(("OCR / Klarschrift-Erkennung", f"Teilweise: {partial}", "warn"))
    elif dmtx:
        steps.append(("OCR / Klarschrift-Erkennung", "Übersprungen (DataMatrix dekodiert)", "muted"))
    else:
        steps.append(("OCR / Klarschrift-Erkennung", "Nicht erkannt", "fail"))

    if r["success"]:
        steps.append(("Fusion & Gegenprobe", f"→ {r['result_code']} via {r['method']}", "ok"))
    else:
        steps.append(("Fusion & Gegenprobe", f"Gescheitert ({r.get('fail_reason') or '?'})", "fail"))

    if r.get("expected"):
        if r["is_match"]:
            steps.append(("Ground-Truth-Vergleich", f"{r['result_code']} = {r['expected']}", "ok"))
        else:
            steps.append(("Ground-Truth-Vergleich", f"{display_code(r)} ≠ {r['expected']}", "fail"))
    return steps


# ═══════════════════════════════════════════════════════════════════════════════
#  Bild-Hilfsfunktionen und eigene Widgets
# ═══════════════════════════════════════════════════════════════════════════════

def cv2_to_pil(cv_img: np.ndarray) -> Image.Image:
    """Konvertiert ein OpenCV-Bild (BGR) in ein PIL-Bild (RGB)."""
    if len(cv_img.shape) == 2:
        return Image.fromarray(cv_img)
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


def fit_image(pil_img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    """Skaliert ein PIL-Bild proportional in den gegebenen Rahmen (höchstens vierfach vergrößert)."""
    w, h = pil_img.size
    ratio = min(max_w / w, max_h / h, 4.0)
    return pil_img.resize((max(1, int(w * ratio)), max(1, int(h * ratio))), Image.Resampling.LANCZOS)


def draw_boxes_on_image(cv_img: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Zeichnet YOLO-Detektions-Boxen auf ein Bild."""
    annotated = cv_img.copy()
    colors = {0: (0, 200, 100), 1: (255, 165, 0)}  # DataMatrix=grün, Text=orange
    labels = {0: "DataMatrix", 1: "Text"}

    for det in detections:
        cls = det.get("cls", 0)
        conf = det.get("conf", 0.0)
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        color = colors.get(cls, (200, 200, 200))
        label = f"{labels.get(cls, '?')} {conf:.0%}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return annotated


def make_badges(height: int) -> dict[str, ImageTk.PhotoImage]:
    """Farbige Status-Plaketten (Haken / Ausrufezeichen / Kreuz) für Tabellen, 4-fach gerendert für glatte Kanten."""
    scale, width = 4, int(height * 1.7)
    badges = {}
    for kind, color in (("ok", SUCCESS), ("wrong", WARN), ("fail", DANGER)):
        image = Image.new("RGBA", (width * scale, height * scale), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((0, 0, width * scale - 1, height * scale - 1), radius=height * scale // 4, fill=color)
        cx, cy, u = width * scale / 2, height * scale / 2, height * scale
        stroke = max(2, int(u * 0.11))
        if kind == "ok":
            draw.line([(cx - .20 * u, cy + .02 * u), (cx - .06 * u, cy + .17 * u), (cx + .22 * u, cy - .15 * u)],
                      fill="white", width=stroke, joint="curve")
        elif kind == "fail":
            draw.line([(cx - .16 * u, cy - .16 * u), (cx + .16 * u, cy + .16 * u)], fill="white", width=stroke)
            draw.line([(cx - .16 * u, cy + .16 * u), (cx + .16 * u, cy - .16 * u)], fill="white", width=stroke)
        else:
            draw.line([(cx, cy - .22 * u), (cx, cy + .04 * u)], fill="white", width=stroke)
            draw.ellipse((cx - .06 * u, cy + .13 * u, cx + .06 * u, cy + .25 * u), fill="white")
        badges[kind] = ImageTk.PhotoImage(image.resize((width, height), Image.Resampling.LANCZOS))
    return badges


class ImageViewer(tk.Canvas):
    """Bildanzeige mit Mausrad-Zoom (wheel() wird von der App aufgerufen), Verschieben per Ziehen und Einpassen per Doppelklick."""

    MAX_ZOOM = 16.0

    def __init__(self, master, on_zoom=None):
        super().__init__(master, bg=BG_MAIN, highlightthickness=0, bd=0, width=1, height=1)
        self._src: Image.Image | None = None
        self._photo = None
        self._message = ""
        self._zoom = self._fit = 1.0
        self._ox = self._oy = 0.0
        self._fitted = True
        self._drag = None
        self._job = None
        self._on_zoom = on_zoom
        self._font = tkfont.Font(family="Segoe UI", size=11)
        self.bind("<Configure>", lambda _e: self._schedule(resized=True))
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._move)
        self.bind("<Double-Button-1>", lambda _e: self.fit())

    def show(self, image: Image.Image | None, message: str = ""):
        self._src, self._message = image, message
        self.fit()

    def fit(self):
        self._fitted = True
        self._apply_fit()
        self._schedule()

    def _apply_fit(self):
        if self._src is None:
            return
        cw, ch = max(1, self.winfo_width()), max(1, self.winfo_height())
        iw, ih = self._src.size
        self._fit = self._zoom = min(cw / iw, ch / ih)
        self._ox, self._oy = (cw - iw * self._zoom) / 2, (ch - ih * self._zoom) / 2
        self.configure(cursor="")
        self._notify()

    def _notify(self):
        if self._on_zoom is not None:
            self._on_zoom(self._zoom * 100 if self._src is not None else None)

    def _schedule(self, resized: bool = False):
        if resized and self._src is not None:
            if self._fitted:
                self._apply_fit()
            else:
                self._clamp()
        if self._job is None:
            self._job = self.after_idle(self._render)

    def _clamp(self):
        cw, ch = self.winfo_width(), self.winfo_height()
        w, h = self._src.width * self._zoom, self._src.height * self._zoom
        self._ox = (cw - w) / 2 if w <= cw else min(0.0, max(cw - w, self._ox))
        self._oy = (ch - h) / 2 if h <= ch else min(0.0, max(ch - h, self._oy))

    def wheel(self, event):
        if self._src is None:
            return
        x, y = event.x_root - self.winfo_rootx(), event.y_root - self.winfo_rooty()
        lowest, highest = self._fit, max(self.MAX_ZOOM, self._fit)
        zoom = min(highest, max(lowest, self._zoom * 1.25 ** (event.delta / 120)))
        if zoom == self._zoom:
            return
        self._ox = x - (x - self._ox) * zoom / self._zoom  # Bildpunkt unter dem Mauszeiger bleibt stehen
        self._oy = y - (y - self._oy) * zoom / self._zoom
        self._zoom = zoom
        self._fitted = zoom <= self._fit * 1.001
        self.configure(cursor="" if self._fitted else "fleur")
        self._clamp()
        self._notify()
        self._schedule()

    def _press(self, event):
        self._drag = (event.x, event.y, self._ox, self._oy)

    def _move(self, event):
        if self._drag is None or self._src is None or self._fitted:
            return
        x0, y0, ox, oy = self._drag
        self._ox, self._oy = ox + event.x - x0, oy + event.y - y0
        self._clamp()
        self._schedule()

    def _render(self):
        self._job = None
        self.delete("all")
        cw, ch = self.winfo_width(), self.winfo_height()
        if self._src is None:
            if self._message:
                self.create_text(cw // 2, ch // 2, text=self._message, font=self._font, fill=TXT_MID,
                                 width=max(50, cw - 40), justify="center")
            return
        z = self._zoom
        iw, ih = self._src.size
        left, top = max(0, int(-self._ox / z)), max(0, int(-self._oy / z))
        right, bottom = min(iw, int((cw - self._ox) / z) + 1), min(ih, int((ch - self._oy) / z) + 1)
        if right <= left or bottom <= top:
            return
        region = self._src.crop((left, top, right, bottom))
        size = (max(1, round((right - left) * z)), max(1, round((bottom - top) * z)))
        if z >= 4:
            shown = region.resize(size, Image.Resampling.NEAREST)  # einzelne Punkte der Dot-Peen-Codes sichtbar
        elif z >= 1:
            shown = region.resize(size, Image.Resampling.BICUBIC)
        else:
            shown = region.resize(size, Image.Resampling.BILINEAR, reducing_gap=2.0)
        self._photo = ImageTk.PhotoImage(shown)
        self.create_image(self._ox + left * z, self._oy + top * z, anchor="nw", image=self._photo)


@dataclass(slots=True)
class Tile:
    path: str
    name: str
    code: str      # Ground Truth ("" = offen)
    outcome: str   # "" (kein Lauf) | "ok" | "wrong" | "fail"


class Gallery(tk.Canvas):
    """Kachelansicht der Testbilder als eine Zeichenfläche (schnell auch bei vielen Bildern); Klick → on_open(pfad)."""

    def __init__(self, master, scaling: float, on_open):
        super().__init__(master, bg=BG_CARD, highlightthickness=0, bd=0, width=1, height=1)
        self._scaling = scaling
        self._on_open = on_open
        self.thumb_size = (int(128 * scaling), int(96 * scaling))
        self._pad = int(6 * scaling)
        self._gap = int(10 * scaling)
        self._caption = int(26 * scaling)
        self._tile_w = self.thumb_size[0] + 2 * self._pad
        self._tile_h = self.thumb_size[1] + self._pad + self._caption
        self._font = tkfont.Font(family="Segoe UI", size=9)
        self._font_bold = tkfont.Font(family="Segoe UI", size=9, weight="bold")
        self._tiles: list[Tile] = []
        self._photos: dict[str, ImageTk.PhotoImage] = {}
        self._image_items: dict[str, int] = {}
        self._rects: list[int] = []
        self._cols = 0
        self._left = self._gap
        self._hover = -1
        self._job = None
        self.configure(yscrollincrement=max(1, (self._tile_h + self._gap) // 4))
        self.bind("<Configure>", self._on_configure)
        self.bind("<Motion>", self._on_motion)
        self.bind("<Leave>", lambda _e: self._set_hover(-1))
        self.bind("<Button-1>", self._on_click)

    def show(self, tiles: list[Tile], reset_scroll: bool = False):
        self._tiles = tiles
        top = 0.0 if reset_scroll else self.yview()[0]
        self._layout(force=True)
        self.yview_moveto(top)

    def set_thumbnail(self, path: str, photo: ImageTk.PhotoImage):
        self._photos[path] = photo
        item = self._image_items.get(path)
        if item is not None:
            self.itemconfigure(item, image=photo)

    def wheel(self, event):
        self.yview_scroll(-int(event.delta / 120) * 4, "units")

    def _on_configure(self, _event):
        if self._job is not None:
            self.after_cancel(self._job)
        self._job = self.after(60, self._layout)

    def _layout(self, force: bool = False):
        self._job = None
        width = self.winfo_width()
        if width <= 1:
            return
        cols = max(1, (width - self._gap) // (self._tile_w + self._gap))
        if cols == self._cols and not force:
            return
        self._cols = cols
        self.delete("all")
        self._image_items, self._rects, self._hover = {}, [], -1
        self._left = max(self._gap, (width - cols * self._tile_w - (cols - 1) * self._gap) // 2)
        for index, tile in enumerate(self._tiles):
            row, col = divmod(index, cols)
            self._draw_tile(tile, self._left + col * (self._tile_w + self._gap),
                            self._gap + row * (self._tile_h + self._gap))
        rows = math.ceil(len(self._tiles) / cols)
        self.configure(scrollregion=(0, 0, width, max(self._gap + rows * (self._tile_h + self._gap), self.winfo_height())))

    def _draw_tile(self, tile: Tile, x: int, y: int):
        self._rects.append(self.create_rectangle(x, y, x + self._tile_w, y + self._tile_h, fill=BG_SIDE, outline=BORDER))
        options = {"image": self._photos[tile.path]} if tile.path in self._photos else {}
        self._image_items[tile.path] = self.create_image(
            x + self._tile_w // 2, y + self._pad + self.thumb_size[1] // 2, **options)
        text_y = y + self._pad + self.thumb_size[1] + self._caption // 2
        self.create_text(x + self._pad + 2, text_y, text=tile.name, anchor="w", font=self._font, fill=TXT_MID)
        code, color = (tile.code, SUCCESS) if tile.code else ("offen", WARN)
        self.create_text(x + self._tile_w - self._pad - 2, text_y, text=code, anchor="e", font=self._font_bold, fill=color)
        if tile.outcome:
            r = int(7 * self._scaling)
            cx, cy = x + self._tile_w - self._pad - r - 4, y + self._pad + r + 4
            self.create_oval(cx - r, cy - r, cx + r, cy + r, fill=OUTCOME_COLORS[tile.outcome], outline="white", width=2)

    def _tile_at(self, x: int, y: int) -> int:
        dx, dy = self.canvasx(x) - self._left, self.canvasy(y) - self._gap
        if dx < 0 or dy < 0 or not self._cols:
            return -1
        col, off_x = divmod(int(dx), self._tile_w + self._gap)
        row, off_y = divmod(int(dy), self._tile_h + self._gap)
        if col >= self._cols or off_x >= self._tile_w or off_y >= self._tile_h:
            return -1
        index = row * self._cols + col
        return index if index < len(self._tiles) else -1

    def _set_hover(self, index: int):
        if index == self._hover:
            return
        if 0 <= self._hover < len(self._rects):
            self.itemconfigure(self._rects[self._hover], outline=BORDER, width=1)
        self._hover = index
        if index >= 0:
            self.itemconfigure(self._rects[index], outline=ACCENT, width=2)
        self.configure(cursor="hand2" if index >= 0 else "")

    def _on_motion(self, event):
        self._set_hover(self._tile_at(event.x, event.y))

    def _on_click(self, event):
        index = self._tile_at(event.x, event.y)
        if index >= 0:
            self._on_open(self._tiles[index].path)


# ═══════════════════════════════════════════════════════════════════════════════
#  Hauptapplikation
# ═══════════════════════════════════════════════════════════════════════════════

class BenchmarkApp(ctk.CTk):
    """Hauptfenster: Seitenleiste mit Testdaten und Berichten, daneben ein Reiter je Aufgabe."""

    def __init__(self, image_dir: str = IMAGE_DIR, gt_path: str = GROUND_TRUTH_PATH):
        super().__init__()
        self.title(f"Benchmark  —  DataDetector v{BENCHMARK_VERSION}")
        self.geometry("1400x860")
        self.minsize(1100, 700)
        self.configure(fg_color=BG_MAIN)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # --- Daten ---
        self.image_dir = image_dir
        self.gt_path = gt_path
        self.image_paths = load_image_list(image_dir)
        self.ground_truth = load_ground_truth(gt_path)
        self._scaling = ctk.ScalingTracker.get_widget_scaling(self)

        # --- Angezeigter Lauf (Bericht oder frischer Lauf) und Vergleich ---
        self.results: list[dict] = []
        self.summary: dict = {}
        self._raw_results: list[dict] = []   # Lauf wie gespeichert; results/summary = ggf. neu bewertet
        self._raw_summary: dict = {}
        self.run_source: dict = {}          # kind: report | live | partial | cancelled, path, timestamp, version
        self._result_by_file: dict[str, dict] = {}
        self._outcomes: dict[str, str] = {}
        self.compare_path: str | None = None
        self.diff: dict[str, str] = {}
        self._compare_by_file: dict[str, dict] = {}
        self._compare_labels: dict[str, str] = {}
        self.reports: list[dict] = []
        self._report_cache: dict = {}
        self._report_rows: dict[str, tuple] = {}
        self._autoload_done = False

        # --- Hintergrund-Threads: nur über die Ereignis-Warteschlange mit der Oberfläche verbunden ---
        self._events: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._reports_lock = threading.Lock()
        self._model = None
        self.run_active = False
        self._closing = False
        self._dirty = {"overview", "annotate", "results", "detail"}

        # --- Zustand der Reiter ---
        self._annotate_paths: list[str] = []
        self._annotate_index = 0
        self._annotate_shown: str | None = None
        self._detail_rows: dict[str, dict] = {}
        self._detail_request = 0
        self._detail_job = None
        self._detail_pending: str | None = None
        self._detail_current: dict | None = None
        self._bench_partial = False
        self._result_rows: list[dict] = []
        self._result_sort: str | None = None
        self._result_sort_desc = False
        self._hover_iid = {}
        self._search_job = None

        self._setup_styles()
        self._build_ui()
        self._update_sidebar_data()
        self._refresh_visible()

        threading.Thread(target=self._thumbnail_worker, args=(list(self.image_paths), self.gallery.thumb_size),
                         daemon=True).start()
        self._refresh_reports()
        self.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        self.bind("<Prior>", lambda _e: self._on_page_key(-1))
        self.bind("<Next>", lambda _e: self._on_page_key(1))
        self.after(40, self._drain_events)

    # ------------------------------------------------------------------ #
    #  Grundlagen                                                          #
    # ------------------------------------------------------------------ #
    def report_callback_exception(self, exc, val, tb):
        logger.error("Fehler in der Oberfläche", exc_info=(exc, val, tb))
        self._set_status(f"Fehler: {val}", DANGER)

    def _on_close(self):
        if self.run_active and not messagebox.askyesno(
                "Benchmark beenden", "Der Benchmark läuft noch. Wirklich beenden?\nDer Lauf wird nicht gespeichert."):
            return
        self._closing = True
        self._stop_event.set()
        self.destroy()

    def _drain_events(self):
        try:
            for _ in range(80):
                kind, payload = self._events.get_nowait()
                try:
                    getattr(self, f"_on_{kind}")(payload)
                except Exception:  # ein fehlerhaftes Ereignis darf die Verarbeitung nicht beenden
                    logger.exception(f"Ereignis '{kind}' konnte nicht verarbeitet werden")
        except queue.Empty:
            pass
        if not self._closing:
            self.after(40, self._drain_events)

    def _set_status(self, text: str, color: str = TXT_MID):
        self.status_label.configure(text=text, text_color=color)

    def _setup_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        self._table_font = tkfont.Font(family="Segoe UI", size=10)
        row_height = self._table_font.metrics("linespace") + 12
        style.configure("Bench.Treeview", background=BG_CARD, fieldbackground=BG_CARD, foreground=TXT_DARK,
                        rowheight=row_height, font=self._table_font, borderwidth=0, relief="flat")
        style.configure("Bench.Treeview.Heading", background=BG_SIDE, foreground=TXT_MID, relief="flat",
                        borderwidth=0, font=("Segoe UI", 10, "bold"), padding=(6, 6))
        style.map("Bench.Treeview", background=[("selected", SELECT_ROW)], foreground=[("selected", TXT_DARK)])
        style.map("Bench.Treeview.Heading", background=[("active", BORDER)])
        style.layout("Bench.Treeview", [("Treeview.treearea", {"sticky": "nswe"})])
        style.layout("Bench.Treeview.Item", [("Treeitem.padding", {"sticky": "nswe", "children": [
            ("Treeitem.image", {"side": "left", "sticky": ""}),
            ("Treeitem.text", {"side": "left", "sticky": ""}),
        ]})])
        self._badges = make_badges(max(14, row_height - 12))

    def _on_mousewheel(self, event):
        """Mausrad für die eigenen Zeichenflächen (Zoom bzw. Scrollen); Treeview und CTk-Widgets scrollen selbst."""
        try:
            widget = self.winfo_containing(event.x_root, event.y_root)
        except (KeyError, tk.TclError):  # geöffnetes Dropdown-Menü
            return
        while widget is not None:
            if widget is self.viewer:
                self.viewer.wheel(event)
                return
            if widget is self.gallery:
                self.gallery.wheel(event)
                return
            widget = widget.master

    def _on_page_key(self, step: int):
        if self.tabview.get() == "Ground Truth":
            self._annotate_move(step)

    # ------------------------------------------------------------------ #
    #  Aufbau: Seitenleiste und Reiter                                     #
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_sidebar()

        self.tabview = ctk.CTkTabview(self, fg_color=BG_CARD, segmented_button_selected_color=ACCENT,
                                      segmented_button_selected_hover_color=ACCENT_HOVER,
                                      command=self._refresh_visible)
        self.tabview.grid(row=0, column=1, padx=16, pady=16, sticky="nsew")
        self.tab_overview = self.tabview.add("Übersicht")
        self.tab_annotate = self.tabview.add("Ground Truth")
        self.tab_benchmark = self.tabview.add("Benchmark")
        self.tab_results = self.tabview.add("Ergebnisse")
        self.tab_detail = self.tabview.add("Detail")
        self._build_overview()
        self._build_annotate()
        self._build_benchmark()
        self._build_results()
        self._build_detail()

        self.btn_open_sidebar = ctk.CTkButton(
            self, text=">", width=36, height=36, corner_radius=8,
            fg_color=ACCENT, text_color="#FFFFFF", hover_color=ACCENT_HOVER,
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"), command=self._toggle_sidebar)

    def _build_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0, fg_color=BG_SIDE,
                                    border_width=1, border_color=BORDER)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)
        self.sidebar.grid_columnconfigure(0, weight=1)
        self.sidebar.grid_rowconfigure(5, weight=1)

        header = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        header.grid(row=0, column=0, padx=(20, 10), pady=(18, 2), sticky="ew")
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(header, text="Benchmark", text_color=ACCENT,
                     font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold")).grid(row=0, column=0, sticky="w")
        ctk.CTkButton(
            header, text="<", width=32, height=32, corner_radius=6,
            fg_color="transparent", text_color=TXT_DARK, hover_color=BORDER,
            font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"), command=self._toggle_sidebar
        ).grid(row=0, column=1, sticky="e")
        ctk.CTkLabel(self.sidebar, text=f"Erkennungstest & Ground Truth v{BENCHMARK_VERSION}",
                     font=ctk.CTkFont(size=11), text_color=TXT_LIGHT).grid(row=1, column=0, padx=20, pady=(0, 14), sticky="w")

        ctk.CTkLabel(self.sidebar, text="Testdaten:", font=ctk.CTkFont(weight="bold", size=13),
                     text_color=TXT_DARK).grid(row=2, column=0, padx=20, pady=(0, 2), sticky="w")
        data = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        data.grid(row=3, column=0, padx=20, pady=(2, 12), sticky="ew")
        data.grid_columnconfigure(0, weight=1)
        self.side_images_label = ctk.CTkLabel(data, text="", anchor="w", font=ctk.CTkFont(size=12), text_color=TXT_DARK)
        self.side_images_label.grid(row=0, column=0, sticky="w")
        self.side_gt_label = ctk.CTkLabel(data, text="", anchor="w", font=ctk.CTkFont(size=12), text_color=TXT_DARK)
        self.side_gt_label.grid(row=1, column=0, sticky="w", pady=(2, 4))
        self.side_gt_bar = ctk.CTkProgressBar(data, height=6, fg_color=BORDER, progress_color=SUCCESS)
        self.side_gt_bar.grid(row=2, column=0, sticky="ew")
        self.side_warn_label = ctk.CTkLabel(data, text="", anchor="w", justify="left", wraplength=240,
                                            font=ctk.CTkFont(size=11), text_color=WARN)
        self.side_warn_label.grid(row=3, column=0, sticky="w", pady=(4, 0))

        ctk.CTkLabel(self.sidebar, text="Läufe:", font=ctk.CTkFont(weight="bold", size=13),
                     text_color=TXT_DARK).grid(row=4, column=0, padx=20, pady=(0, 2), sticky="w")
        self.reports_frame = ctk.CTkScrollableFrame(self.sidebar, fg_color="transparent", label_text="Gespeicherte Berichte")
        self.reports_frame.grid(row=5, column=0, padx=12, pady=(4, 10), sticky="nsew")
        self.reports_empty_label = ctk.CTkLabel(self.reports_frame, text="Noch keine Läufe gespeichert.",
                                                font=ctk.CTkFont(size=11), text_color=TXT_LIGHT)

        self.run_status_label = ctk.CTkLabel(self.sidebar, text="", anchor="w", font=ctk.CTkFont(size=12, weight="bold"),
                                             text_color=ACCENT)
        self.run_status_label.grid(row=6, column=0, padx=20, pady=(0, 2), sticky="w")
        self.run_bar = ctk.CTkProgressBar(self.sidebar, height=6, fg_color=BORDER, progress_color=ACCENT)
        self.run_bar.grid(row=7, column=0, padx=20, pady=(0, 8), sticky="ew")
        self.run_status_label.grid_remove()
        self.run_bar.grid_remove()

        outline = {"height": 30, "fg_color": "transparent", "border_width": 1, "border_color": BORDER,
                   "text_color": TXT_DARK, "hover_color": BORDER}
        ctk.CTkButton(self.sidebar, text="📂  Bildordner öffnen", command=self._open_image_dir, **outline
                      ).grid(row=8, column=0, padx=20, pady=(0, 6), sticky="ew")
        ctk.CTkButton(self.sidebar, text="📄  Bericht öffnen …", command=self._open_report_file, **outline
                      ).grid(row=9, column=0, padx=20, pady=(0, 4), sticky="ew")
        self.status_label = ctk.CTkLabel(self.sidebar, text="Bereit", font=ctk.CTkFont(size=11), text_color=TXT_MID,
                                         justify="left", anchor="w", wraplength=240)
        self.status_label.grid(row=10, column=0, padx=20, pady=(4, 14), sticky="w")

    def _toggle_sidebar(self):
        if self.sidebar.winfo_ismapped():
            self.sidebar.grid_remove()
            self.btn_open_sidebar.place(x=12, y=24)
            self.btn_open_sidebar.lift()
        else:
            self.btn_open_sidebar.place_forget()
            self.sidebar.grid()

    def _open_image_dir(self):
        if os.path.isdir(self.image_dir):
            os.startfile(self.image_dir)
        else:
            messagebox.showinfo("Bildordner", f"Der Ordner existiert nicht:\n{self.image_dir}")

    def _open_report_file(self):
        path = filedialog.askopenfilename(title="Benchmark-Bericht öffnen", initialdir=REPORTS_DIR if os.path.isdir(REPORTS_DIR) else APP_DIR,
                                          filetypes=[("Benchmark-Bericht", "*.json"), ("Alle Dateien", "*.*")])
        if not path:
            return
        try:
            report = read_report(path, self.image_dir)
        except (OSError, ValueError) as e:
            messagebox.showerror("Bericht öffnen", f"Der Bericht konnte nicht gelesen werden:\n{e}")
            return
        self._show_report(report)

    def _update_sidebar_data(self):
        n = len(self.image_paths)
        names = {os.path.basename(p) for p in self.image_paths}
        annotated = len(names & self.ground_truth.keys())
        stale = len(self.ground_truth.keys() - names)
        folder = os.path.basename(os.path.normpath(self.image_dir)) or self.image_dir
        self.side_images_label.configure(text=f"{_fmt_int(n)} Bilder  ·  {folder}" if n else f"Keine Bilder in „{folder}“",
                                         text_color=TXT_DARK if n else WARN)
        self.side_gt_label.configure(text=f"Ground Truth: {annotated} / {n}  ({_fmt_pct(annotated / n * 100 if n else 0)})")
        self.side_gt_bar.set(annotated / n if n else 0)
        self.side_warn_label.configure(text=f"⚠ {stale} Ground-Truth-Einträge ohne Bilddatei" if stale else "")

    # --- Liste der Berichte ---
    def _make_report_row(self, path: str) -> tuple:
        row = ctk.CTkFrame(self.reports_frame, fg_color="transparent", corner_radius=6)
        row.pack(fill="x", pady=2)
        name = ctk.CTkLabel(row, text="", anchor="w", height=20, font=ctk.CTkFont(size=12, weight="bold"), text_color=TXT_DARK)
        name.pack(fill="x", padx=10, pady=(5, 0))
        info = ctk.CTkLabel(row, text="", anchor="w", height=16, font=ctk.CTkFont(size=10), text_color=TXT_MID)
        info.pack(fill="x", padx=10, pady=(0, 5))
        for widget in (row, name, info):
            widget.bind("<Button-1>", lambda _e, p=path: self._select_report(p))
        self._bind_hover(row, "transparent", BG_MAIN, is_active=lambda p=path: p == self.run_source.get("path"))
        return row, name, info

    def _update_report_rows(self):
        keys = [r["path"] for r in self.reports]
        if list(self._report_rows) != keys:
            for row, _name, _info in self._report_rows.values():
                row.destroy()
            self._report_rows = {key: self._make_report_row(key) for key in keys}
        if keys:
            self.reports_empty_label.pack_forget()
        else:
            self.reports_empty_label.pack(pady=10)
        active = self.run_source.get("path")
        for report in self.reports:
            row, name, info = self._report_rows[report["path"]]
            s = report["summary"]
            name.configure(text=report_title(report))
            info.configure(text=f"v{report['version']}  ·  {s['success_count']}/{s['total_images']} gelesen  ·  {s['false_read_count']} falsch")
            row.configure(fg_color=BORDER if report["path"] == active else "transparent")

    def _select_report(self, path: str):
        report = next((r for r in self.reports if r["path"] == path), None)
        if report is not None:
            self._show_report(report)

    @staticmethod
    def _bind_hover(widget, normal: str, hover: str, is_active=lambda: False):
        """Hover-Farbe, die beim Überfahren der enthaltenen Labels nicht flackert."""
        def enter(_event):
            if not is_active():
                widget.configure(fg_color=hover)

        def leave(_event):
            inside = widget.winfo_containing(*widget.winfo_pointerxy())
            if (inside is None or not str(inside).startswith(str(widget))) and not is_active():
                widget.configure(fg_color=normal)
        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)

    def _kpi_card(self, parent, column: int, title: str, color: str) -> tuple:
        card = ctk.CTkFrame(parent, fg_color=BG_SIDE, border_width=1, border_color=BORDER, corner_radius=8)
        card.grid(row=0, column=column, padx=6, pady=(10, 12), sticky="ew")
        ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=11, weight="bold"), text_color=TXT_MID).pack(pady=(10, 0))
        value = ctk.CTkLabel(card, text="—", font=ctk.CTkFont(size=22, weight="bold"), text_color=color)
        value.pack()
        sub = ctk.CTkLabel(card, text="", font=ctk.CTkFont(size=10), text_color=TXT_MID)
        sub.pack(pady=(0, 10))
        self._bind_hover(card, BG_SIDE, HOVER_CARD)
        return value, sub

    @staticmethod
    def _rate_color(pct: float) -> str:
        return SUCCESS if pct >= 95 else (WARN if pct >= 80 else DANGER)

    def _goto_tab(self, name: str):
        self.tabview.set(name)
        self._refresh_visible()

    def _refresh_visible(self, entering: bool = True):
        """Sichtbaren Reiter aktualisieren; entering=False bei reinen Datenänderungen (Eingaben bleiben erhalten)."""
        tab = self.tabview.get()
        if tab == "Übersicht" and "overview" in self._dirty:
            self._refresh_overview()
        elif tab == "Ground Truth":
            self._enter_annotate() if entering else self._annotate_refresh()
        elif tab == "Benchmark":
            self._refresh_benchmark_info()
        elif tab == "Ergebnisse" and "results" in self._dirty:
            self._refresh_results()
        elif tab == "Detail" and "detail" in self._dirty:
            self._refresh_detail_list()

    # ------------------------------------------------------------------ #
    #  Reiter „Übersicht“                                                  #
    # ------------------------------------------------------------------ #
    def _build_overview(self):
        tab = self.tab_overview
        for column in range(5):
            tab.grid_columnconfigure(column, weight=1, uniform="kpi")
        tab.grid_rowconfigure(1, weight=1)
        self.ov_images = self._kpi_card(tab, 0, "Testbilder", TXT_DARK)
        self.ov_gt = self._kpi_card(tab, 1, "Ground Truth", SUCCESS)
        self.ov_rate = self._kpi_card(tab, 2, "Erfolgsrate", SUCCESS)
        self.ov_accuracy = self._kpi_card(tab, 3, "Genauigkeit", ACCENT)
        self.ov_run = self._kpi_card(tab, 4, "Angezeigter Lauf", TXT_DARK)

        card = ctk.CTkFrame(tab, fg_color=BG_CARD, border_width=1, border_color=BORDER, corner_radius=8)
        card.grid(row=1, column=0, columnspan=5, sticky="nsew", padx=6, pady=(0, 6))
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(1, weight=1)
        toolbar = ctk.CTkFrame(card, fg_color="transparent")
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", padx=12, pady=(10, 6))
        ctk.CTkLabel(toolbar, text="Bildvorschau", font=ctk.CTkFont(size=13, weight="bold"), text_color=TXT_DARK).pack(side="left")
        ctk.CTkLabel(toolbar, text="Klick auf ein Bild öffnet es im Ground-Truth-Editor.", font=ctk.CTkFont(size=11),
                     text_color=TXT_LIGHT).pack(side="left", padx=16)
        self.gallery_filter = ctk.CTkSegmentedButton(
            toolbar, values=["Alle", "Ohne GT", "Fehler"], command=lambda _v: self._refresh_gallery(reset_scroll=True),
            selected_color=ACCENT, selected_hover_color=ACCENT_HOVER)
        self.gallery_filter.set("Alle")
        self.gallery_filter.pack(side="right")
        self.gallery_count = ctk.CTkLabel(toolbar, text="", font=ctk.CTkFont(size=11, weight="bold"), text_color=TXT_MID)
        self.gallery_count.pack(side="right", padx=12)

        self.gallery = Gallery(card, self._scaling, self._open_annotation)
        self.gallery.grid(row=1, column=0, sticky="nsew", padx=(8, 0), pady=(0, 8))
        scrollbar = ctk.CTkScrollbar(card, command=self.gallery.yview)
        scrollbar.grid(row=1, column=1, sticky="ns", padx=(0, 4), pady=(0, 8))
        self.gallery.configure(yscrollcommand=scrollbar.set)
        self.gallery_empty = ctk.CTkLabel(card, text="", fg_color=BG_CARD, text_color=TXT_LIGHT, font=ctk.CTkFont(size=13))

    def _refresh_overview(self):
        self._dirty.discard("overview")
        self._update_overview_kpis()
        self._refresh_gallery()

    def _update_overview_kpis(self):
        n = len(self.image_paths)
        names = {os.path.basename(p) for p in self.image_paths}
        annotated = len(names & self.ground_truth.keys())
        pct = annotated / n * 100 if n else 0
        folder = os.path.basename(os.path.normpath(self.image_dir)) or self.image_dir
        self.ov_images[0].configure(text=_fmt_int(n))
        self.ov_images[1].configure(text=folder)
        self.ov_gt[0].configure(text=f"{annotated} / {n}", text_color=SUCCESS if annotated == n and n else WARN)
        self.ov_gt[1].configure(text=f"{_fmt_pct(pct)} annotiert")
        s = self.summary
        if s:
            self.ov_rate[0].configure(text=_fmt_pct(s["success_rate_pct"]), text_color=self._rate_color(s["success_rate_pct"]))
            self.ov_rate[1].configure(text=f"{s['success_count']} von {s['total_images']} gelesen")
            self.ov_accuracy[0].configure(text=_fmt_pct(s["accuracy_pct"]), text_color=self._rate_color(s["accuracy_pct"]))
            self.ov_accuracy[1].configure(
                text=f"{s['false_read_count']} falsch gelesen" if s["false_read_count"] else "keine Fehllesung",
                text_color=DANGER if s["false_read_count"] else TXT_MID)
            when = self.run_source.get("timestamp")
            self.ov_run[0].configure(text=when.strftime("%d.%m. %H:%M") if when else "—")
            self.ov_run[1].configure(text=f"{self._run_label()}  ·  Ø {_fmt_int(s['avg_duration_ms'])} ms")
        else:
            for value, sub in (self.ov_rate, self.ov_accuracy, self.ov_run):
                value.configure(text="—", text_color=TXT_DARK)
                sub.configure(text="kein Lauf geladen", text_color=TXT_MID)

    def _refresh_gallery(self, reset_scroll: bool = False):
        mode = self.gallery_filter.get()
        tiles = []
        for path in self.image_paths:
            name = os.path.basename(path)
            code, kind = self.ground_truth.get(name, ""), self._outcomes.get(name, "")
            if mode == "Ohne GT" and code:
                continue
            if mode == "Fehler" and kind not in ("fail", "wrong"):
                continue
            tiles.append(Tile(path, name, code, kind))
        self.gallery.show(tiles, reset_scroll=reset_scroll)
        self.gallery_count.configure(text=f"{len(tiles)} von {len(self.image_paths)} Bildern")
        if tiles:
            self.gallery_empty.place_forget()
        else:
            if not self.image_paths:
                text = f"Keine Bilder gefunden in\n{self.image_dir}"
            elif mode == "Fehler":
                text = "Keine Fehler im angezeigten Lauf." if self.results else "Kein Lauf geladen."
            else:
                text = "Alle Bilder haben eine Ground Truth."
            self.gallery_empty.configure(text=text)
            self.gallery_empty.place(relx=0.5, rely=0.4, anchor="center")

    def _thumbnail_worker(self, paths: list[str], size: tuple[int, int]):
        for path in paths:
            if self._closing:
                return
            try:
                with Image.open(path) as image:
                    image.draft("RGB", (size[0] * 2, size[1] * 2))
                    thumb = image.convert("RGB")
                thumb.thumbnail(size)
            except Exception as e:
                logger.warning(f"Vorschau für {os.path.basename(path)} nicht möglich: {e}")
                thumb = None
            self._events.put(("thumb", (path, thumb)))

    def _on_thumb(self, payload: tuple):
        path, image = payload
        if image is not None:
            self.gallery.set_thumbnail(path, ImageTk.PhotoImage(image))

    # ------------------------------------------------------------------ #
    #  Berichte, angezeigter Lauf und Vergleich                            #
    # ------------------------------------------------------------------ #
    def _refresh_reports(self):
        threading.Thread(target=self._reports_worker, daemon=True).start()

    def _reports_worker(self):
        with self._reports_lock:
            try:
                reports = list_reports(self.image_dir, self._report_cache)
            except Exception:
                logger.exception("Berichte konnten nicht gelesen werden")
                reports = []
        self._events.put(("reports", reports))

    def _on_reports(self, reports: list[dict]):
        self.reports = reports
        self._update_report_rows()
        if not self._autoload_done:
            self._autoload_done = True
            names = {os.path.basename(p) for p in self.image_paths}
            match = next((r for r in reports if report_fits(r, names)), None)
            if match is not None and not self.results and not self.run_active:
                self._show_report(match, switch=False)
                return
        if self.compare_path and not self._compare_report():
            self._pick_default_compare()
            self._dirty |= {"results", "detail"}
        self._update_compare_menu()

    def _show_report(self, report: dict, switch: bool = True):
        if not report["results"]:
            messagebox.showinfo("Bericht", "Der Bericht enthält keine Ergebnisse.")
            return
        source = {"kind": "report", "path": report["path"], "timestamp": report["timestamp"], "version": report["version"]}
        self._set_displayed_run(report["results"], report["summary"], source)
        self._set_status(f"Bericht geladen: {report_title(report)}", TXT_MID)
        if switch and self.tabview.get() in ("Benchmark", "Ground Truth"):
            self._goto_tab("Ergebnisse")

    @property
    def rescore_enabled(self) -> bool:
        return bool(self.rescore_switch.get())

    def _set_displayed_run(self, results: list[dict], summary: dict, source: dict):
        self._raw_results, self._raw_summary, self.run_source = results, summary, source
        self._recompute_run()
        self._pick_default_compare()
        self._dirty |= {"overview", "results", "detail"}
        self._update_overview_kpis()
        self._update_report_rows()
        self._refresh_visible(entering=False)

    def _recompute_run(self):
        """Angezeigte Ergebnisse = gespeicherter Lauf, auf Wunsch gegen die aktuelle Ground Truth neu bewertet."""
        if self.rescore_enabled and self._raw_results:
            self.results = reevaluate(self._raw_results, self.ground_truth)
            self.summary = summarize(self.results, len(self.results), self._raw_summary.get("total_time_s", 0))
        else:
            self.results, self.summary = self._raw_results, self._raw_summary
        self._result_by_file = {r["filename"]: r for r in self.results}
        self._outcomes = {name: outcome(r) for name, r in self._result_by_file.items()}

    def _pick_default_compare(self):
        """Vergleichslauf = der nächstältere Bericht (bei frischen Läufen der neueste)."""
        own, stamp = self.run_source.get("path"), self.run_source.get("timestamp")
        older = [r for r in self.reports if r["path"] != own and (stamp is None or r["timestamp"] < stamp)]
        self.compare_path = older[0]["path"] if older else None
        self._apply_compare()

    def _compare_report(self) -> dict | None:
        return next((r for r in self.reports if r["path"] == self.compare_path), None)

    def _apply_compare(self):
        report = self._compare_report()
        previous = []
        if report is not None:
            previous = reevaluate(report["results"], self.ground_truth) if self.rescore_enabled else report["results"]
        self._compare_by_file = {r["filename"]: r for r in previous}
        self.diff = compare_runs(self.results, previous) if previous else {}

    def _run_label(self) -> str:
        kind, stamp = self.run_source.get("kind"), self.run_source.get("timestamp")
        if not self.results:
            return "Kein Lauf geladen"
        when = stamp.strftime("%d.%m.%Y %H:%M") if stamp else ""
        return {"report": f"Bericht vom {when}", "live": f"Lauf vom {when}",
                "partial": "Teil-Lauf (nicht gespeichert)", "unsaved": "Lauf (nicht gespeichert)",
                "cancelled": "Abgebrochener Lauf (nicht gespeichert)"}.get(kind, "Lauf")

    def _on_gt_changed(self):
        self._update_sidebar_data()
        self._dirty.add("overview")
        if self.rescore_enabled and self._raw_results:
            self._recompute_run()
            self._apply_compare()
            self._dirty |= {"results", "detail"}
        self._update_overview_kpis()

    # ------------------------------------------------------------------ #
    #  Reiter „Ground Truth“                                               #
    # ------------------------------------------------------------------ #
    def _build_annotate(self):
        tab = self.tab_annotate
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        bar = ctk.CTkFrame(tab, fg_color=BG_SIDE, border_width=1, border_color=BORDER, corner_radius=8)
        bar.grid(row=0, column=0, columnspan=2, sticky="ew", padx=6, pady=(10, 8))
        bar.grid_columnconfigure(2, weight=1)
        self.annotate_filter = ctk.CTkSegmentedButton(
            bar, values=["Alle", "Offen", "Fehler"], command=self._on_annotate_filter,
            selected_color=ACCENT, selected_hover_color=ACCENT_HOVER)
        self.annotate_filter.set("Alle")
        self.annotate_filter.grid(row=0, column=0, padx=(12, 8), pady=10)
        self.annotate_progress_label = ctk.CTkLabel(bar, text="", font=ctk.CTkFont(size=13, weight="bold"),
                                                    text_color=TXT_DARK, width=110, anchor="w")
        self.annotate_progress_label.grid(row=0, column=1, padx=8)
        self.annotate_progress_bar = ctk.CTkProgressBar(bar, height=8, fg_color=BORDER, progress_color=ACCENT)
        self.annotate_progress_bar.set(0)
        self.annotate_progress_bar.grid(row=0, column=2, sticky="ew", padx=8)
        self.annotate_count_label = ctk.CTkLabel(bar, text="", font=ctk.CTkFont(size=12, weight="bold"), text_color=SUCCESS)
        self.annotate_count_label.grid(row=0, column=3, padx=(8, 14))

        view_card = ctk.CTkFrame(tab, fg_color=BG_MAIN, border_width=1, border_color=BORDER, corner_radius=8)
        view_card.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 6))
        view_card.grid_columnconfigure(0, weight=1)
        view_card.grid_rowconfigure(0, weight=1)
        self.viewer = ImageViewer(view_card, on_zoom=self._on_viewer_zoom)
        self.viewer.grid(row=0, column=0, sticky="nsew", padx=6, pady=(6, 0))
        foot = ctk.CTkFrame(view_card, fg_color="transparent")
        foot.grid(row=1, column=0, sticky="ew", padx=12, pady=(4, 6))
        ctk.CTkLabel(foot, text="Mausrad: Zoom  ·  Ziehen: Verschieben  ·  Doppelklick: Einpassen",
                     font=ctk.CTkFont(size=10), text_color=TXT_MID).pack(side="left")
        self.zoom_label = ctk.CTkLabel(foot, text="", font=ctk.CTkFont(size=10, weight="bold"), text_color=TXT_MID)
        self.zoom_label.pack(side="right")

        panel = ctk.CTkFrame(tab, width=330, fg_color=BG_CARD, border_width=1, border_color=BORDER, corner_radius=8)
        panel.grid(row=1, column=1, sticky="ns", padx=(0, 6), pady=(0, 6))
        panel.pack_propagate(False)
        ctk.CTkLabel(panel, text="Enter = Speichern & weiter  ·  Esc = Zurücksetzen\nBild ↑ / Bild ↓ = Zurück / Weiter",
                     font=ctk.CTkFont(size=10), text_color=TXT_LIGHT, justify="left"
                     ).pack(side="bottom", anchor="w", padx=20, pady=(0, 10))  # zuerst packen: bleibt bei wenig Höhe sichtbar
        ctk.CTkLabel(panel, text="BILD", font=ctk.CTkFont(size=12, weight="bold"), text_color=TXT_LIGHT
                     ).pack(anchor="w", padx=20, pady=(14, 2))
        self.annotate_name = ctk.CTkLabel(panel, text="—", font=ctk.CTkFont(size=18, weight="bold"), text_color=TXT_DARK)
        self.annotate_name.pack(anchor="w", padx=20)
        self.annotate_state = ctk.CTkLabel(panel, text="", font=ctk.CTkFont(size=12, weight="bold"), text_color=TXT_MID)
        self.annotate_state.pack(anchor="w", padx=20, pady=(2, 0))
        self.annotate_hint = ctk.CTkLabel(panel, text="", font=ctk.CTkFont(size=11), text_color=TXT_MID,
                                          wraplength=290, justify="left", anchor="w")
        self.annotate_hint.pack(anchor="w", padx=20, pady=(2, 0))

        ctk.CTkLabel(panel, text="Code eingeben:", font=ctk.CTkFont(size=13, weight="bold"), text_color=TXT_DARK
                     ).pack(anchor="w", padx=20, pady=(16, 4))
        self.annotate_entry = ctk.CTkEntry(
            panel, height=52, font=ctk.CTkFont(family="Consolas", size=26, weight="bold"), justify="center",
            corner_radius=8, border_width=2, border_color=BORDER, fg_color=BG_CARD, text_color=TXT_DARK,
            placeholder_text="z. B. W032")
        self.annotate_entry.pack(fill="x", padx=20)
        self.annotate_entry.bind("<Return>", lambda _e: self._annotate_save_and_next())
        self.annotate_entry.bind("<Escape>", lambda _e: self._annotate_show())
        self.annotate_entry.bind("<KeyRelease>", self._on_code_key)
        self.annotate_format = ctk.CTkLabel(panel, text="", font=ctk.CTkFont(size=11), text_color=TXT_LIGHT,
                                            wraplength=290, justify="left", anchor="w")
        self.annotate_format.pack(anchor="w", padx=20, pady=(4, 12))

        self.annotate_save_btn = ctk.CTkButton(
            panel, text="💾  Speichern & weiter", height=44, fg_color=SUCCESS, hover_color=SUCCESS_HOVER,
            font=ctk.CTkFont(size=14, weight="bold"), command=self._annotate_save_and_next)
        self.annotate_save_btn.pack(fill="x", padx=20, pady=(0, 8))
        neutral = {"height": 34, "fg_color": BG_SIDE, "text_color": TXT_DARK, "hover_color": BORDER,
                   "border_width": 1, "border_color": BORDER, "font": ctk.CTkFont(size=12, weight="bold")}
        edit = ctk.CTkFrame(panel, fg_color="transparent")
        edit.pack(fill="x", padx=20, pady=(0, 12))
        edit.grid_columnconfigure((0, 1), weight=1, uniform="edit")
        self.annotate_skip_btn = ctk.CTkButton(edit, text="⏭  Überspringen", command=lambda: self._annotate_step(1), **neutral)
        self.annotate_skip_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.annotate_delete_btn = ctk.CTkButton(
            edit, text="🗑️  Löschen", height=34, fg_color="transparent", text_color=DANGER,
            hover_color=ROW_REGRESSION, border_width=1, border_color=BORDER, font=ctk.CTkFont(size=12, weight="bold"),
            command=self._annotate_delete)
        self.annotate_delete_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        nav = ctk.CTkFrame(panel, fg_color="transparent")
        nav.pack(fill="x", padx=20)
        nav.grid_columnconfigure((0, 1), weight=1, uniform="nav")
        self.annotate_prev_btn = ctk.CTkButton(nav, text="‹  Zurück", command=lambda: self._annotate_step(-1), **neutral)
        self.annotate_prev_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.annotate_next_btn = ctk.CTkButton(nav, text="Weiter  ›", command=lambda: self._annotate_step(1), **neutral)
        self.annotate_next_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.annotate_open_btn = ctk.CTkButton(
            panel, text="Nächste offene  »", height=34, fg_color=ACCENT, hover_color=ACCENT_HOVER,
            font=ctk.CTkFont(size=12, weight="bold"), command=self._annotate_next_open)
        self.annotate_open_btn.pack(fill="x", padx=20, pady=(10, 4))
        self.annotate_external_btn = ctk.CTkButton(
            panel, text="Im Bildbetrachter öffnen", height=28, fg_color="transparent", text_color=ACCENT,
            hover_color=BORDER, font=ctk.CTkFont(size=12), command=self._annotate_open_external)
        self.annotate_external_btn.pack(fill="x", padx=20)

    def _on_viewer_zoom(self, percent: float | None):
        self.zoom_label.configure(text=f"Zoom {percent:.0f} %" if percent else "")

    def _annotate_accepts(self, path: str, mode: str) -> bool:
        name = os.path.basename(path)
        if mode == "Offen":
            return name not in self.ground_truth
        if mode == "Fehler":
            return self._outcomes.get(name) in ("fail", "wrong")
        return True

    def _annotate_current_path(self) -> str | None:
        if 0 <= self._annotate_index < len(self._annotate_paths):
            return self._annotate_paths[self._annotate_index]
        return None

    def _annotate_rebuild(self, keep: str | None = None):
        """Bildliste für den Filter neu aufbauen; ohne `keep` (oder wenn es herausfällt) beginnt sie beim ersten Bild."""
        mode = self.annotate_filter.get()
        self._annotate_paths = [p for p in self.image_paths if self._annotate_accepts(p, mode)]
        self._annotate_index = self._annotate_paths.index(keep) if keep in self._annotate_paths else 0

    def _enter_annotate(self):
        self._annotate_rebuild(keep=self._annotate_current_path())
        self._annotate_show()
        self.annotate_entry.focus_set()

    def _annotate_refresh(self):
        """Neuer Lauf o. Ä. bei geöffnetem Reiter: Liste und Hinweise aktualisieren, Eingabe und Zoom bleiben."""
        self._annotate_rebuild(keep=self._annotate_current_path())
        self._annotate_show(force=False)

    def _on_annotate_filter(self, _value: str):
        self._annotate_rebuild(keep=self._annotate_current_path())
        self._annotate_show()

    def _open_annotation(self, path: str):
        """Öffnet ein Bild im Ground-Truth-Editor (Filter wird bei Bedarf auf „Alle“ gestellt)."""
        if path not in self.image_paths:
            messagebox.showinfo("Ground Truth", "Dieses Bild gehört nicht zu den aktuellen Testbildern.")
            return
        if not self._annotate_accepts(path, self.annotate_filter.get()):
            self.annotate_filter.set("Alle")
        self._annotate_rebuild(keep=path)
        self._goto_tab("Ground Truth")

    def _annotate_empty_text(self) -> str:
        if not self.image_paths:
            return f"Keine Bilder gefunden in\n{self.image_dir}"
        mode = self.annotate_filter.get()
        if mode == "Offen":
            return "Alle Bilder sind annotiert."
        if mode == "Fehler":
            return "Keine Fehler im angezeigten Lauf." if self.results else "Kein Lauf geladen."
        return "Keine Bilder."

    def _annotate_show(self, force: bool = True):
        """Zeigt das aktuelle Bild; ohne force bleiben Eingabe und Zoom erhalten, solange das Bild dasselbe ist."""
        path = self._annotate_current_path()
        changed = force or path != self._annotate_shown
        self._annotate_shown = path
        total, count = len(self.image_paths), len(self._annotate_paths)
        annotated = len({os.path.basename(p) for p in self.image_paths} & self.ground_truth.keys())
        self.annotate_count_label.configure(text=f"✓ {annotated} von {total} annotiert")
        controls = (self.annotate_save_btn, self.annotate_skip_btn, self.annotate_prev_btn, self.annotate_next_btn,
                    self.annotate_open_btn, self.annotate_external_btn, self.annotate_delete_btn)
        if path is None:
            self.viewer.show(None, self._annotate_empty_text())
            self.annotate_progress_label.configure(text="Keine Bilder")
            self.annotate_progress_bar.set(0)
            self.annotate_name.configure(text="—")
            self.annotate_state.configure(text="")
            self.annotate_hint.configure(text="")
            self._annotate_set_entry("")
            for button in controls:
                button.configure(state="disabled")
            self.annotate_open_btn.configure(state="normal" if self.image_paths else "disabled")
            return

        for button in controls:
            button.configure(state="normal")
        name = os.path.basename(path)
        code = self.ground_truth.get(name, "")
        self.annotate_progress_label.configure(text=f"Bild {self._annotate_index + 1} / {count}")
        self.annotate_progress_bar.set((self._annotate_index + 1) / count)
        self.annotate_name.configure(text=name)
        self.annotate_state.configure(text=f"Annotiert: {code}" if code else "Noch nicht annotiert",
                                      text_color=SUCCESS if code else WARN)
        result = self._result_by_file.get(name)
        if code and result is not None:  # Scanner-Ergebnis erst nach der eigenen Annotation zeigen (kein Vorbefüllen)
            same = result["success"] and result["result_code"] == code
            self.annotate_hint.configure(
                text=f"Scanner ({self._run_label()}): {display_code(result)}  —  {'stimmt überein' if same else 'weicht ab'}",
                text_color=SUCCESS if same else WARN)
        else:
            self.annotate_hint.configure(text="")
        self.annotate_prev_btn.configure(state="normal" if self._annotate_index > 0 else "disabled")
        self.annotate_next_btn.configure(state="normal" if self._annotate_index < count - 1 else "disabled")
        if not changed:
            return
        self._annotate_set_entry(code)
        try:
            with Image.open(path) as image:
                self.viewer.show(image.convert("RGB"))
        except Exception as e:
            self.viewer.show(None, f"Bild konnte nicht geladen werden:\n{e}")
        if self.tabview.get() == "Ground Truth":
            self.annotate_entry.focus_set()

    def _annotate_set_entry(self, text: str):
        self.annotate_entry.delete(0, "end")
        if text:
            self.annotate_entry.insert(0, text)
        self._validate_code()

    def _validate_code(self):
        code = self.annotate_entry.get().strip().upper()
        if not code:
            self.annotate_entry.configure(border_color=BORDER)
            self.annotate_format.configure(text="Format: A, B, P oder W + 3 Ziffern", text_color=TXT_LIGHT)
        elif HORDEN_PATTERN.match(code):
            self.annotate_entry.configure(border_color=SUCCESS)
            self.annotate_format.configure(text="✓ Format gültig", text_color=SUCCESS)
        else:
            self.annotate_entry.configure(border_color=WARN)
            self.annotate_format.configure(text="Ungewöhnliches Format – erwartet A, B, P oder W + 3 Ziffern",
                                           text_color=WARN)

    def _on_code_key(self, _event):
        text = self.annotate_entry.get()
        if text != text.upper():
            position = self.annotate_entry.index("insert")
            self.annotate_entry.delete(0, "end")
            self.annotate_entry.insert(0, text.upper())
            self.annotate_entry.icursor(position)
        self._validate_code()

    def _annotate_step(self, step: int) -> bool:
        """Zum nächsten / vorherigen Bild der Liste; False, wenn das Listenende erreicht ist."""
        target = self._annotate_index + step
        moved = 0 <= target < len(self._annotate_paths)
        if moved:
            self._annotate_index = target
        self._annotate_show()
        return moved

    def _annotate_move(self, step: int):
        if self._annotate_paths:
            self._annotate_step(step)

    def _persist_ground_truth(self, name: str, previous: str | None) -> bool:
        try:
            save_ground_truth(self.ground_truth, self.gt_path)
        except OSError as e:
            if previous is None:
                self.ground_truth.pop(name, None)
            else:
                self.ground_truth[name] = previous
            messagebox.showerror("Ground Truth", f"Die Ground-Truth-Datei konnte nicht gespeichert werden:\n{e}")
            return False
        self._on_gt_changed()
        return True

    def _annotate_save_and_next(self):
        path = self._annotate_current_path()
        if path is None:
            return
        code = self.annotate_entry.get().strip().upper()
        if not code:
            self.annotate_entry.configure(border_color=DANGER)
            self.after(700, self._validate_code)
            self._set_status("Bitte einen Code eingeben oder „Überspringen“ wählen.", WARN)
            return
        if not HORDEN_PATTERN.match(code) and not messagebox.askyesno(
                "Format prüfen", f"„{code}“ entspricht nicht dem Format A, B, P oder W + 3 Ziffern.\nTrotzdem speichern?"):
            return
        name = os.path.basename(path)
        previous = self.ground_truth.get(name)
        self.ground_truth[name] = code
        if not self._persist_ground_truth(name, previous):
            return
        moved = self._annotate_step(1)
        self._set_status(f"Gespeichert: {name} = {code}" + ("" if moved else "\nLetztes Bild der Liste erreicht."), SUCCESS)

    def _annotate_delete(self):
        path = self._annotate_current_path()
        name = os.path.basename(path) if path else ""
        if name not in self.ground_truth:
            return
        previous = self.ground_truth.pop(name)
        if self._persist_ground_truth(name, previous):
            self._set_status(f"Annotation gelöscht: {name}", TXT_MID)
        self._annotate_show()

    def _annotate_next_open(self):
        paths, current = self.image_paths, self._annotate_current_path()
        start = paths.index(current) if current in paths else -1
        for offset in range(1, len(paths) + 1):
            candidate = paths[(start + offset) % len(paths)]
            if os.path.basename(candidate) not in self.ground_truth:
                self._open_annotation(candidate)
                return
        self._set_status("Alle Bilder sind annotiert.", SUCCESS)

    def _annotate_open_external(self):
        path = self._annotate_current_path()
        if path and os.path.isfile(path):
            os.startfile(path)


    # ------------------------------------------------------------------ #
    #  Reiter „Benchmark“                                                  #
    # ------------------------------------------------------------------ #
    def _build_benchmark(self):
        tab = self.tab_benchmark
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(3, weight=1)

        info = ctk.CTkFrame(tab, fg_color=BG_SIDE, border_width=1, border_color=BORDER, corner_radius=8)
        info.grid(row=0, column=0, sticky="ew", padx=6, pady=(10, 8))
        info.grid_columnconfigure(1, weight=1)
        self.bench_info: dict[str, ctk.CTkLabel] = {}
        rows = (("images", "Testbilder"), ("gt", "Ground Truth"), ("model", "YOLO-Modell"), ("pipeline", "Pipeline"))
        for row, (key, title) in enumerate(rows):
            top, bottom = (12 if row == 0 else 2), (12 if row == len(rows) - 1 else 2)
            ctk.CTkLabel(info, text=f"{title}:", font=ctk.CTkFont(size=12, weight="bold"), text_color=TXT_MID,
                         anchor="w", width=110).grid(row=row, column=0, padx=(16, 4), pady=(top, bottom), sticky="nw")
            label = ctk.CTkLabel(info, text="", font=ctk.CTkFont(size=12), text_color=TXT_DARK, anchor="w",
                                 justify="left", wraplength=560)
            label.grid(row=row, column=1, padx=(4, 16), pady=(top, bottom), sticky="w")
            self.bench_info[key] = label
        self._bench_info_wrap = 0
        info.bind("<Configure>", self._on_bench_info_resized)

        controls = ctk.CTkFrame(tab, fg_color="transparent")
        controls.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 8))
        controls.grid_columnconfigure(2, weight=1)
        self.bench_start_btn = ctk.CTkButton(
            controls, text="▶  Benchmark starten", width=190, height=36, fg_color=ACCENT, hover_color=ACCENT_HOVER,
            text_color_disabled=TXT_MID, font=ctk.CTkFont(size=13, weight="bold"), command=self._start_benchmark)
        self.bench_start_btn.grid(row=0, column=0)
        self.bench_stop_btn = ctk.CTkButton(
            controls, text="■  Abbrechen", width=120, height=36, fg_color=BORDER, hover_color=DANGER_HOVER,
            text_color_disabled=TXT_LIGHT, font=ctk.CTkFont(size=13, weight="bold"), state="disabled",
            command=self._stop_benchmark)
        self.bench_stop_btn.grid(row=0, column=1, padx=8)
        self.bench_show_btn = ctk.CTkButton(
            controls, text="Ergebnisse anzeigen  →", width=150, height=36, fg_color="transparent", text_color=ACCENT,
            hover_color=BORDER, font=ctk.CTkFont(size=12, weight="bold"), command=lambda: self._goto_tab("Ergebnisse"))
        self.bench_show_btn.grid(row=0, column=3, sticky="e")

        scope = ctk.CTkFrame(controls, fg_color="transparent")
        scope.grid(row=1, column=0, columnspan=4, sticky="w", pady=(8, 0))
        ctk.CTkLabel(scope, text="Umfang:", font=ctk.CTkFont(size=12, weight="bold"), text_color=TXT_DARK
                     ).pack(side="left", padx=(4, 8))
        self.bench_scope = ctk.CTkSegmentedButton(
            scope, values=["Alle Bilder", "Nur Fehler wiederholen"],
            command=lambda _v: self._refresh_benchmark_info(), selected_color=ACCENT, selected_hover_color=ACCENT_HOVER)
        self.bench_scope.set("Alle Bilder")
        self.bench_scope.pack(side="left")
        self.bench_scope_label = ctk.CTkLabel(scope, text="", font=ctk.CTkFont(size=11), text_color=TXT_MID)
        self.bench_scope_label.pack(side="left", padx=12)

        progress = ctk.CTkFrame(tab, fg_color=BG_SIDE, border_width=1, border_color=BORDER, corner_radius=8)
        progress.grid(row=2, column=0, sticky="ew", padx=6, pady=(0, 8))
        self.bench_progress_label = ctk.CTkLabel(progress, text="Bereit zum Start", font=ctk.CTkFont(size=14, weight="bold"),
                                                 text_color=TXT_DARK)
        self.bench_progress_label.pack(anchor="w", padx=16, pady=(12, 0))
        self.bench_progress_bar = ctk.CTkProgressBar(progress, height=12, fg_color=BORDER, progress_color=ACCENT)
        self.bench_progress_bar.set(0)
        self.bench_progress_bar.pack(fill="x", padx=16, pady=(8, 4))
        self.bench_progress_detail = ctk.CTkLabel(progress, text="", font=ctk.CTkFont(size=11), text_color=TXT_MID)
        self.bench_progress_detail.pack(anchor="w", padx=16, pady=(0, 12))

        log_card = ctk.CTkFrame(tab, fg_color=BG_CARD, border_width=1, border_color=BORDER, corner_radius=8)
        log_card.grid(row=3, column=0, sticky="nsew", padx=6, pady=(0, 6))
        log_card.grid_columnconfigure(0, weight=1)
        log_card.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(log_card, text="LIVE-PROTOKOLL", font=ctk.CTkFont(size=12, weight="bold"), text_color=TXT_LIGHT
                     ).grid(row=0, column=0, padx=16, pady=(10, 2), sticky="w")
        self.bench_log_text = ctk.CTkTextbox(
            log_card, fg_color=BG_CARD, text_color=TXT_DARK, font=ctk.CTkFont(family="Consolas", size=12),
            border_width=0, corner_radius=8, wrap="none", state="disabled")
        self.bench_log_text.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 6))
        for tag, color in (("fail", DANGER), ("warn", WARN), ("muted", TXT_LIGHT), ("heading", ACCENT)):
            self.bench_log_text.tag_config(tag, foreground=color)

    def _on_bench_info_resized(self, event):
        """Infozeilen brechen passend zur Kartenbreite um (Spalte links: 110 px + Abstände)."""
        wrap = max(200, int(event.width / self._scaling) - 150)
        if wrap != self._bench_info_wrap:
            self._bench_info_wrap = wrap
            for label in self.bench_info.values():
                label.configure(wraplength=wrap)

    def _failure_paths(self) -> list[str]:
        """Testbilder, die im angezeigten Lauf nicht oder falsch gelesen wurden."""
        return [p for p in self.image_paths if self._outcomes.get(os.path.basename(p)) in ("fail", "wrong")]

    def _refresh_benchmark_info(self):
        n = len(self.image_paths)
        annotated = len({os.path.basename(p) for p in self.image_paths} & self.ground_truth.keys())
        model_path, is_2class = yolo_detector.find_model_path(APP_DIR)
        try:
            model_path = os.path.relpath(model_path, APP_DIR)
        except ValueError:
            pass
        self.bench_info["images"].configure(text=f"{_fmt_int(n)} Bilder in {self.image_dir}")
        self.bench_info["gt"].configure(text=f"{annotated} von {n} Bildern annotiert  ({os.path.basename(self.gt_path)})")
        self.bench_info["model"].configure(text=f"{model_path}  ({'2-Klassen' if is_2class else '1-Klasse / Basismodell'})")
        self.bench_info["pipeline"].configure(
            text="YOLO-Detektion → DataMatrix-Decode → OCR-Fusion → Gegenprobe auf dem Gesamtbild (wie im DataMatrixReader)")
        if self.bench_scope.get() == "Alle Bilder":
            text = f"{n} Bilder werden ausgewertet"
        else:
            failures = len(self._failure_paths())
            text = f"{failures} Bilder werden ausgewertet" if failures else "Der angezeigte Lauf hat keine Fehler."
        self.bench_scope_label.configure(text=text)
        self._update_run_controls()

    def _update_run_controls(self):
        running = self.run_active
        stoppable = running and not self._stop_event.is_set()
        self.bench_start_btn.configure(state="disabled" if running else "normal", fg_color=BORDER if running else ACCENT,
                                       text="⏳  Benchmark läuft …" if running else "▶  Benchmark starten")
        self.bench_stop_btn.configure(state="normal" if stoppable else "disabled", fg_color=DANGER if stoppable else BORDER)
        self.bench_scope.configure(state="disabled" if running else "normal")

    def _append_log(self, text: str, tag: str | None = None):
        if tag is None:
            tag = {"✗": "fail", "⚠": "warn"}.get(text.lstrip()[:1])
        box = self.bench_log_text
        box.configure(state="normal")
        box.insert("end", text + "\n", tag) if tag else box.insert("end", text + "\n")
        box.see("end")
        box.configure(state="disabled")

    def _start_benchmark(self):
        if self.run_active:
            return
        if not self.image_paths:
            messagebox.showinfo("Benchmark", f"Keine Bilder gefunden in\n{self.image_dir}")
            return
        partial = self.bench_scope.get() != "Alle Bilder"
        paths = self._failure_paths() if partial else list(self.image_paths)
        if not paths:
            messagebox.showinfo("Benchmark", "Der angezeigte Lauf hat keine Fehler – es gibt nichts zu wiederholen.")
            return
        self.run_active = True
        self._bench_partial = partial
        self._stop_event.clear()
        self.bench_log_text.configure(state="normal")
        self.bench_log_text.delete("1.0", "end")
        self.bench_log_text.configure(state="disabled")
        self.bench_progress_bar.set(0)
        self.bench_progress_label.configure(text="Starte …", text_color=TXT_DARK)
        self.bench_progress_detail.configure(text="")
        self.run_bar.set(0)
        self.run_status_label.configure(text="Benchmark startet …")
        self.run_status_label.grid()
        self.run_bar.grid()
        self._update_run_controls()
        threading.Thread(target=self._benchmark_worker, args=(paths, dict(self.ground_truth), partial), daemon=True).start()

    def _stop_benchmark(self):
        self._stop_event.set()
        self._update_run_controls()
        self.bench_progress_detail.configure(text="Abbruch angefordert – das aktuelle Bild wird noch zu Ende gelesen …")
        self.run_status_label.configure(text="Wird abgebrochen …")

    def _benchmark_worker(self, paths: list[str], gt_map: dict, partial: bool):
        """Läuft im Hintergrund; meldet sich ausschließlich über self._events."""
        post = self._events.put
        ok = 0

        def on_progress(idx, total, results, elapsed):
            nonlocal ok
            ok += bool(results[-1]["success"])
            post(("bench_progress", (idx, total, ok, elapsed)))

        try:
            if self._model is None:
                post(("bench_status", "YOLO-Modell wird geladen …"))
                post(("bench_log", ("Lade YOLO-Modell …", "muted")))
                self._model = yolo_detector.load_model(APP_DIR)[0]
                post(("bench_log", ("YOLO-Modell geladen.", "muted")))
            results, summary = run_benchmark(
                self._model, paths, gt_map, log=lambda line: post(("bench_log", (line, None))),
                on_progress=on_progress, should_stop=self._stop_event.is_set,
                on_start=lambda idx, total, path: post(("bench_current", (idx, total, os.path.basename(path)))))
            report_path = None
            if not partial and not summary["cancelled"]:
                try:
                    report_path = save_report(results, summary, self.image_dir, self.gt_path)
                    post(("bench_log", (f"\nBericht gespeichert: {report_path}", "muted")))
                except OSError as e:
                    post(("bench_log", (f"\nBericht konnte nicht gespeichert werden: {e}", "fail")))
            for line in format_summary_lines(summary):
                post(("bench_log", (line, "heading")))
            post(("bench_done", {"results": results, "summary": summary, "report_path": report_path}))
        except Exception as e:
            logger.exception("Benchmark fehlgeschlagen")
            post(("bench_error", f"{type(e).__name__}: {e}"))

    def _on_bench_status(self, text: str):
        self.bench_progress_label.configure(text=text)
        self.run_status_label.configure(text=text)

    def _on_bench_log(self, payload: tuple):
        self._append_log(*payload)

    def _on_bench_current(self, payload: tuple):
        idx, total, name = payload
        self.bench_progress_label.configure(text=f"Bild {idx + 1} / {total}:  {name}")

    def _on_bench_progress(self, payload: tuple):
        idx, total, ok, elapsed = payload
        done = idx + 1
        self.bench_progress_bar.set(done / total)
        self.run_bar.set(done / total)
        if not self._stop_event.is_set():
            self.run_status_label.configure(text=f"Benchmark läuft: {done} / {total}")
        self.bench_progress_detail.configure(
            text=f"Gelesen: {ok} / {done}   ·   vergangen {_fmt_duration(elapsed)}   ·   "
                 f"Rest ca. {_fmt_duration(elapsed / done * (total - done))}")

    def _finish_run_ui(self):
        self.run_active = False
        self.run_status_label.grid_remove()
        self.run_bar.grid_remove()
        self._update_run_controls()

    def _on_bench_done(self, payload: dict):
        self._finish_run_ui()
        results, summary, report_path = payload["results"], payload["summary"], payload["report_path"]
        if not results:
            self.bench_progress_label.configure(text="Abgebrochen – es wurden keine Bilder ausgewertet.", text_color=WARN)
            return
        if report_path:
            kind = "live"
        elif summary["cancelled"]:
            kind = "cancelled"
        else:
            kind = "partial" if self._bench_partial else "unsaved"
        source = {"kind": kind, "path": report_path, "timestamp": datetime.now(), "version": BENCHMARK_VERSION}
        self._set_displayed_run(results, summary, source)
        if report_path:
            self._refresh_reports()
        if summary["cancelled"]:
            self.bench_progress_label.configure(text=f"Abgebrochen nach {len(results)} Bildern", text_color=WARN)
            self.bench_progress_detail.configure(text="Der Teil-Lauf wurde nicht als Bericht gespeichert.")
        else:
            self.bench_progress_bar.set(1.0)
            self.bench_progress_label.configure(
                text=f"Fertig – {_fmt_pct(summary['success_rate_pct'])} Erfolgsrate", text_color=SUCCESS)
            self.bench_progress_detail.configure(
                text=f"{summary['total_images']} Bilder in {_fmt_duration(summary['total_time_s'])}"
                     + ("  ·  Teil-Lauf, nicht als Bericht gespeichert" if self._bench_partial else ""))
        self._set_status(f"Lauf beendet: {summary['success_count']}/{summary['total_images']} gelesen, "
                         f"{summary['false_read_count']} falsch", SUCCESS if not summary["false_read_count"] else WARN)

    def _on_bench_error(self, text: str):
        self._finish_run_ui()
        self._append_log(f"\nFEHLER: {text}", "fail")
        self.bench_progress_label.configure(text="Benchmark fehlgeschlagen", text_color=DANGER)
        self.bench_progress_detail.configure(text=text)
        messagebox.showerror("Benchmark", f"Der Benchmark ist fehlgeschlagen:\n{text}")


    # ------------------------------------------------------------------ #
    #  Reiter „Ergebnisse“                                                 #
    # ------------------------------------------------------------------ #
    RESULT_COLUMNS = (
        ("nr", "Nr.", 52, False, "e"),
        ("file", "Datei", 96, False, "w"),
        ("code", "Ergebnis", 96, False, "w"),
        ("expected", "Soll", 76, False, "w"),
        ("method", "Methode", 112, False, "w"),
        ("conf", "Konf.", 64, False, "e"),
        ("duration", "Dauer", 92, False, "e"),
        ("reason", "Fehlerursache", 190, True, "w"),
        ("delta", "Vergleich", 130, False, "w"),
    )
    DESCENDING_FIRST = {"duration", "conf"}

    def _build_results(self):
        tab = self.tab_results
        for column in range(5):
            tab.grid_columnconfigure(column, weight=1, uniform="kpi")
        tab.grid_rowconfigure(3, weight=1)
        self.rs_rate = self._kpi_card(tab, 0, "Erfolgsrate", SUCCESS)
        self.rs_accuracy = self._kpi_card(tab, 1, "Genauigkeit", ACCENT)
        self.rs_wrong = self._kpi_card(tab, 2, "Falsch gelesen", SUCCESS)
        self.rs_duration = self._kpi_card(tab, 3, "Ø Dauer", TXT_DARK)
        self.rs_total = self._kpi_card(tab, 4, "Gesamtzeit", TXT_DARK)

        dist = ctk.CTkFrame(tab, fg_color="transparent")
        dist.grid(row=1, column=0, columnspan=5, sticky="ew", pady=(0, 8))
        dist.grid_columnconfigure((0, 1, 2), weight=1, uniform="dist")
        self.dist_methods = self._dist_card(dist, 0, "Erfolgsmethoden")
        self.dist_failures = self._dist_card(dist, 1, "Fehlerursachen")
        self._build_compare_card(dist, 2)

        toolbar = ctk.CTkFrame(tab, fg_color="transparent")
        toolbar.grid(row=2, column=0, columnspan=5, sticky="ew", padx=6, pady=(0, 6))
        toolbar.grid_columnconfigure(0, weight=1)
        self.result_search = ctk.CTkEntry(toolbar, placeholder_text="Suche nach Datei, Code, Methode, Fehlerursache ...")
        self.result_search.grid(row=0, column=0, padx=(0, 8), sticky="ew")
        self.result_search.bind("<KeyRelease>", self._on_result_search_key)
        self.result_filter = ctk.CTkOptionMenu(toolbar, values=list(RESULT_FILTERS), width=170,
                                               command=lambda _v: self._apply_result_filters(), **MENU_STYLE)
        self.result_filter.grid(row=0, column=1, padx=4)
        ctk.CTkButton(toolbar, text="Zurücksetzen", width=90, fg_color="transparent", text_color=ACCENT,
                      hover_color=BORDER, command=self._reset_result_filters).grid(row=0, column=2, padx=4)
        ctk.CTkButton(toolbar, text="📥 CSV exportieren", width=130, fg_color=ACCENT, hover_color=ACCENT_HOVER,
                      font=ctk.CTkFont(size=12, weight="bold"), command=self._export_csv).grid(row=0, column=3, padx=(4, 0))

        table = ctk.CTkFrame(tab, fg_color=BG_CARD, border_width=1, border_color=BORDER, corner_radius=8)
        table.grid(row=3, column=0, columnspan=5, sticky="nsew", padx=6, pady=(0, 4))
        table.grid_columnconfigure(0, weight=1)
        table.grid_rowconfigure(0, weight=1)
        self.result_tree = ttk.Treeview(table, style="Bench.Treeview", columns=[c[0] for c in self.RESULT_COLUMNS],
                                        show=("tree", "headings"), selectmode="browse")
        self.result_tree.grid(row=0, column=0, sticky="nsew", padx=(6, 0), pady=(6, 0))
        scrollbar = ctk.CTkScrollbar(table, command=self.result_tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns", padx=(0, 4), pady=6)
        x_scrollbar = ctk.CTkScrollbar(table, orientation="horizontal", command=self.result_tree.xview)
        x_scrollbar.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 4))
        self.result_tree.configure(yscrollcommand=scrollbar.set, xscrollcommand=x_scrollbar.set)
        self._setup_tree_columns(self.result_tree, self.RESULT_COLUMNS)
        self._update_result_headings()
        self.result_empty = ctk.CTkLabel(table, text="", fg_color=BG_CARD, text_color=TXT_LIGHT, font=ctk.CTkFont(size=13))
        self.result_tree.bind("<Double-1>", self._on_result_double_click)
        self.result_tree.bind("<Return>", lambda _e: self._open_selected_result())
        self._bind_tree_hover(self.result_tree)

        footer = ctk.CTkFrame(tab, fg_color="transparent")
        footer.grid(row=4, column=0, columnspan=5, sticky="ew", padx=12, pady=(0, 4))
        footer.grid_columnconfigure(1, weight=1)
        self.result_count_label = ctk.CTkLabel(footer, text="", font=ctk.CTkFont(size=11, weight="bold"), text_color=TXT_MID)
        self.result_count_label.grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(footer, text="Doppelklick öffnet die Detail-Ansicht", font=ctk.CTkFont(size=11),
                     text_color=TXT_LIGHT).grid(row=0, column=1, sticky="w", padx=16)
        self.rescore_switch = ctk.CTkSwitch(
            footer, text="Mit aktueller GT bewerten", font=ctk.CTkFont(size=11, weight="bold"),
            text_color=TXT_DARK, progress_color=ACCENT, command=self._on_rescore_toggled)
        self.rescore_switch.grid(row=0, column=2, sticky="e")

    def _setup_tree_columns(self, tree: ttk.Treeview, columns: tuple):
        """Spaltenbreiten sind Pixel: an die Windows-Skalierung anpassen."""
        scale = self._scaling
        tree.column("#0", width=int(58 * scale), minwidth=int(58 * scale), stretch=False, anchor="center")
        for column, _title, width, stretch, anchor in columns:
            tree.column(column, width=int(width * scale), minwidth=int(40 * scale), stretch=stretch, anchor=anchor)
        tree.tag_configure("fail", foreground=DANGER)
        tree.tag_configure("wrong", foreground=WARN)
        tree.tag_configure("regression", background=ROW_REGRESSION)
        tree.tag_configure("improved", background=ROW_IMPROVED)
        tree.tag_configure("hover", background=HOVER_ROW)

    def _dist_card(self, parent, column: int, title: str) -> ctk.CTkFrame:
        card = ctk.CTkFrame(parent, fg_color=BG_SIDE, border_width=1, border_color=BORDER, corner_radius=8)
        card.grid(row=0, column=column, padx=6, sticky="nsew")
        ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=12, weight="bold"), text_color=TXT_MID
                     ).pack(anchor="w", padx=14, pady=(10, 4))
        body = ctk.CTkFrame(card, fg_color="transparent", height=4 * 34)
        body.pack(fill="x", padx=14, pady=(0, 10))
        body.pack_propagate(False)
        return body

    def _fill_distribution(self, body: ctk.CTkFrame, counts: dict, total: int, colors, empty_text: str):
        for child in body.winfo_children():
            child.destroy()
        if not counts or not total:
            ctk.CTkLabel(body, text=empty_text, font=ctk.CTkFont(size=11), text_color=TXT_LIGHT).pack(anchor="w")
            return
        items = sorted(counts.items(), key=lambda item: -item[1])
        for name, count in items[:4]:
            row = ctk.CTkFrame(body, fg_color="transparent")
            row.pack(fill="x", pady=(0, 5))
            row.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(row, text=name, anchor="w", height=16, font=ctk.CTkFont(size=11), text_color=TXT_DARK
                         ).grid(row=0, column=0, sticky="w")
            ctk.CTkLabel(row, text=f"{count}× ({_fmt_pct(count / total * 100)})", anchor="e", height=16,
                         font=ctk.CTkFont(size=10), text_color=TXT_MID).grid(row=0, column=1, sticky="e")
            bar = ctk.CTkProgressBar(row, height=8, fg_color=BORDER,
                                     progress_color=colors(name) if callable(colors) else colors)
            bar.set(count / total)
            bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(1, 0))

    def _build_compare_card(self, parent, column: int):
        card = ctk.CTkFrame(parent, fg_color=BG_SIDE, border_width=1, border_color=BORDER, corner_radius=8)
        card.grid(row=0, column=column, padx=6, sticky="nsew")
        head = ctk.CTkFrame(card, fg_color="transparent")
        head.pack(fill="x", padx=14, pady=(10, 4))
        ctk.CTkLabel(head, text="Vergleich", font=ctk.CTkFont(size=12, weight="bold"), text_color=TXT_MID).pack(side="left")
        self.compare_menu = ctk.CTkOptionMenu(head, values=["Kein Vergleich"], height=24, width=120, dynamic_resizing=False,
                                              font=ctk.CTkFont(size=11), command=self._on_compare_selected, **MENU_STYLE)
        self.compare_menu.pack(side="right", fill="x", expand=True, padx=(10, 0))
        self.compare_lines = []
        for _ in range(3):
            label = ctk.CTkLabel(card, text="", font=ctk.CTkFont(size=12), text_color=TXT_MID, anchor="w", justify="left")
            label.pack(anchor="w", padx=14, pady=1)
            self.compare_lines.append(label)
        self._compare_wrap = 0
        card.bind("<Configure>", self._on_compare_card_resized)
        self.compare_changes_btn = ctk.CTkButton(
            card, text="Nur Änderungen anzeigen", height=22, fg_color="transparent", text_color=ACCENT, hover_color=BORDER,
            font=ctk.CTkFont(size=11, weight="bold"), anchor="w", command=self._show_changes_only)
        self.compare_changes_btn.pack(anchor="w", padx=8, pady=(0, 6))

    def _on_compare_card_resized(self, event):
        """Textzeilen der Vergleichskarte brechen um, statt am Kartenrand abgeschnitten zu werden."""
        wrap = max(140, int(event.width / self._scaling) - 32)
        if wrap != self._compare_wrap:
            self._compare_wrap = wrap
            for label in self.compare_lines:
                label.configure(wraplength=wrap)

    def _refresh_results(self):
        self._dirty.discard("results")
        self._update_results_kpis()
        s = self.summary
        total = s.get("total_images", 0)
        self._fill_distribution(self.dist_methods, s.get("method_distribution", {}), total,
                                lambda name: METHOD_COLORS.get(name, TXT_LIGHT), "Noch kein Lauf geladen.")
        self._fill_distribution(self.dist_failures, s.get("failure_categories", {}), total, DANGER,
                                "Keine Fehler." if s else "Noch kein Lauf geladen.")
        self._update_compare_menu()
        self._apply_result_filters()

    def _update_results_kpis(self):
        s = self.summary
        cards = (self.rs_rate, self.rs_accuracy, self.rs_wrong, self.rs_duration, self.rs_total)
        if not s:
            for value, sub in cards:
                value.configure(text="—", text_color=TXT_DARK)
                sub.configure(text="Noch kein Lauf geladen" if value is self.rs_rate[0] else "", text_color=TXT_MID)
            return
        durations = sorted(r["duration_ms"] for r in self.results)
        self.rs_rate[0].configure(text=_fmt_pct(s["success_rate_pct"]), text_color=self._rate_color(s["success_rate_pct"]))
        self.rs_rate[1].configure(text=f"{s['success_count']} von {s['total_images']} gelesen")
        self.rs_accuracy[0].configure(text=_fmt_pct(s["accuracy_pct"]), text_color=self._rate_color(s["accuracy_pct"]))
        self.rs_accuracy[1].configure(text=f"{s['accuracy_count']} von {s['accuracy_total_gt']} korrekt")
        wrong = s["false_read_count"]
        self.rs_wrong[0].configure(text=_fmt_int(wrong), text_color=DANGER if wrong else SUCCESS)
        self.rs_wrong[1].configure(text=f"{s['no_read_count']} nicht gelesen")
        self.rs_duration[0].configure(text=f"{_fmt_int(s['avg_duration_ms'])} ms")
        self.rs_duration[1].configure(
            text=f"Median {_fmt_int(durations[len(durations) // 2])}  ·  max {_fmt_int(durations[-1])}" if durations else "")
        self.rs_total[0].configure(text=_fmt_duration(s["total_time_s"]))
        self.rs_total[1].configure(text=f"{s['total_images']} Bilder")

    # --- Vergleich ---
    def _update_compare_menu(self):
        own = self.run_source.get("path")
        self._compare_labels = {}
        for r in self.reports:
            if r["path"] == own:
                continue
            label = f"{r['timestamp']:%d.%m.%y %H:%M}  ·  v{r['version']}"
            if label in self._compare_labels:  # zwei Berichte in derselben Minute
                label = f"{r['timestamp']:%d.%m.%y %H:%M:%S}  ·  v{r['version']}"
            self._compare_labels[label] = r["path"]
        self.compare_menu.configure(values=["Kein Vergleich", *self._compare_labels])
        current = next((label for label, path in self._compare_labels.items() if path == self.compare_path), "Kein Vergleich")
        self.compare_menu.set(current)
        self._update_compare_info()

    def _update_compare_info(self):
        report = self._compare_report()
        if report is None or not self.results:
            texts = ["Kein Vergleichslauf ausgewählt.", "", ""]
            for label, text in zip(self.compare_lines, texts):
                label.configure(text=text, text_color=TXT_LIGHT)
            self.compare_changes_btn.pack_forget()
            return
        names = {r["filename"] for r in self.results}
        common = [r for r in self._compare_by_file.values() if r["filename"] in names]
        before = summarize(common, len(common), 0)
        now = summarize([r for r in self.results if r["filename"] in self._compare_by_file], len(common), 0)
        d_rate = now["success_rate_pct"] - before["success_rate_pct"]
        d_acc = now["accuracy_pct"] - before["accuracy_pct"]
        d_ms = now["avg_duration_ms"] - before["avg_duration_ms"]
        counts = Counter(self.diff.values())
        worse, better = d_rate < 0 or d_acc < 0, d_rate > 0 or d_acc > 0
        self.compare_lines[0].configure(
            text=f"Erfolgsrate\u00a0{_fmt_delta(d_rate, '\u00a0%-Pkt.')}  ·  Genauigkeit\u00a0{_fmt_delta(d_acc, '\u00a0%-Pkt.')}",
            text_color=DANGER if worse else (SUCCESS if better else TXT_MID))
        regressions = counts["regression"]
        nb = "\u00a0"  # geschützte Leerzeichen: Umbruch nur zwischen den Gruppen
        self.compare_lines[1].configure(
            text=f"▲{nb}{counts['improved']}{nb}besser    ▼{nb}{regressions}{nb}Regression{'' if regressions == 1 else 'en'}    "
                 f"↔{nb}{counts['changed']}{nb}geändert",
            text_color=DANGER if regressions else TXT_DARK)
        common_note = "" if len(common) == len(self.results) else f", {len(common)} gemeinsame Bilder"
        self.compare_lines[2].configure(
            text=f"Ø Dauer {_fmt_delta(d_ms, ' ms', 0)}   (vorher {_fmt_int(before['avg_duration_ms'])} ms{common_note})",
            text_color=TXT_MID)
        if counts["regression"] or counts["improved"] or counts["changed"]:
            self.compare_changes_btn.pack(anchor="w", padx=8, pady=(0, 6))
        else:
            self.compare_changes_btn.pack_forget()

    def _on_compare_selected(self, label: str):
        self.compare_path = self._compare_labels.get(label)
        self._apply_compare()
        self._dirty |= {"results", "detail"}
        self._refresh_results()

    def _show_changes_only(self):
        self.result_filter.set("Änderungen zum Vergleich")
        self._apply_result_filters()

    def _on_rescore_toggled(self):
        self._recompute_run()
        self._apply_compare()
        self._dirty |= {"overview", "results", "detail"}
        self._update_overview_kpis()
        self._refresh_visible(entering=False)
        self._set_status("Bewertung mit der aktuellen Ground Truth (der gespeicherte Bericht bleibt unverändert)."
                         if self.rescore_enabled else "Bewertung wie im Bericht gespeichert.", TXT_MID)

    # --- Filter, Sortierung, Tabelle ---
    def _matches_filter(self, entry: dict, mode: str) -> bool:
        kind = outcome(entry)
        if mode == "Nicht gelesen":
            return kind == "fail"
        if mode == "Falsch gelesen":
            return kind == "wrong"
        if mode == "Korrekt / gelesen":
            return kind == "ok"
        if mode == "Ohne Ground Truth":
            return not entry["expected"]
        if mode == "Änderungen zum Vergleich":
            return self.diff.get(entry["filename"], "new") != "new"
        return True

    @staticmethod
    def _search_text(entry: dict) -> str:
        return " ".join((entry["filename"], display_code(entry), entry["expected"] or "", entry["method"],
                         entry["fail_reason"] or "", status_text(entry))).lower()

    def _sort_key(self, column: str):
        diff_rank = {"regression": 0, "changed": 1, "new": 2, "": 3, "improved": 4}
        keys = {
            "#0": lambda r: {"fail": 0, "wrong": 1, "ok": 2}[outcome(r)],
            "nr": lambda r: r["index"],
            "file": lambda r: _natural_key(r["filename"]),
            "code": display_code,
            "expected": lambda r: r["expected"] or "",
            "method": lambda r: r["method"],
            "conf": lambda r: r["confidence"],
            "duration": lambda r: r["duration_ms"],
            "reason": lambda r: r["fail_reason"] or "",
            "delta": lambda r: diff_rank[self.diff.get(r["filename"], "")],
        }
        return keys[column]

    def _sort_results_by(self, column: str):
        if self._result_sort == column:
            self._result_sort_desc = not self._result_sort_desc
        else:
            self._result_sort, self._result_sort_desc = column, column in self.DESCENDING_FIRST
        self._update_result_headings()
        self._apply_result_filters()

    def _update_result_headings(self):
        def title(column: str, text: str) -> str:
            return f"{text} {'▼' if self._result_sort_desc else '▲'}" if column == self._result_sort else text
        tree = self.result_tree
        tree.heading("#0", text=title("#0", "Status"), command=lambda: self._sort_results_by("#0"))
        for column, text, _width, _stretch, anchor in self.RESULT_COLUMNS:
            tree.heading(column, text=title(column, text), anchor=anchor, command=lambda c=column: self._sort_results_by(c))

    def _on_result_search_key(self, _event=None):
        if self._search_job is not None:
            self.after_cancel(self._search_job)
        self._search_job = self.after(200, self._apply_result_filters)

    def _reset_result_filters(self):
        self.result_search.delete(0, "end")
        self.result_filter.set(RESULT_FILTERS[0])
        self._apply_result_filters()

    def _apply_result_filters(self):
        self._search_job = None
        query = self.result_search.get().strip().lower()
        mode = self.result_filter.get()
        rows = [r for r in self.results
                if self._matches_filter(r, mode) and (not query or query in self._search_text(r))]
        if self._result_sort is not None:
            rows.sort(key=self._sort_key(self._result_sort), reverse=self._result_sort_desc)
        self._result_rows = rows
        self._render_results()

    def _render_results(self):
        tree = self.result_tree
        selected = tree.focus()
        tree.delete(*tree.get_children())
        self._hover_iid[str(tree)] = ""
        for r in self._result_rows:
            kind, delta = outcome(r), self.diff.get(r["filename"], "")
            tags = [t for t in (kind if kind != "ok" else "", delta if delta in ("regression", "improved") else "") if t]
            tree.insert("", "end", iid=r["filename"], image=self._badges[kind], tags=tags,
                        values=(r["index"], r["filename"], display_code(r), r["expected"] or "—", r["method"],
                                f"{r['confidence'] * 100:.0f} %", f"{_fmt_int(r['duration_ms'])} ms",
                                r["fail_reason"] or "", DIFF_TEXT.get(delta, "")))
        if selected and tree.exists(selected):
            tree.selection_set(selected)
            tree.focus(selected)
            tree.see(selected)
        if self._result_rows:
            self.result_empty.place_forget()
        else:
            self.result_empty.configure(
                text="Keine Einträge für diese Filter." if self.results
                else "Noch kein Lauf geladen.\nStarte einen Benchmark oder wähle links einen gespeicherten Bericht.")
            self.result_empty.place(relx=0.5, rely=0.4, anchor="center")
        rescored = "  ·  bewertet mit aktueller Ground Truth" if self.rescore_enabled and self.results else ""
        self.result_count_label.configure(
            text=f"{len(self._result_rows)} von {len(self.results)} Einträgen{rescored}"
                 + (f"  ·  {self._run_label()}" if self.results else ""))

    def _bind_tree_hover(self, tree: ttk.Treeview):
        self._hover_iid[str(tree)] = ""
        tree.bind("<Motion>", lambda e: self._set_row_hover(tree, "" if (iid := tree.identify_row(e.y)) in tree.selection() else iid))
        tree.bind("<Leave>", lambda _e: self._set_row_hover(tree, ""))

    def _set_row_hover(self, tree: ttk.Treeview, iid: str):
        key = str(tree)
        old = self._hover_iid.get(key, "")
        if iid == old:
            return
        if old and tree.exists(old):
            tree.item(old, tags=[t for t in tree.item(old, "tags") or () if t != "hover"])
        tags = tree.item(iid, "tags") or () if iid else ()
        tinted = iid and any(t in ("regression", "improved") for t in tags)
        if iid and not tinted:
            tree.item(iid, tags=[*tags, "hover"])
        self._hover_iid[key] = iid if iid and not tinted else ""

    def _on_result_double_click(self, event):
        if self.result_tree.identify_region(event.x, event.y) in ("tree", "cell"):
            self._open_selected_result()

    def _open_selected_result(self):
        iid = self.result_tree.focus()
        if iid:
            self._show_detail_for(iid)

    def _export_csv(self):
        if not self._result_rows:
            messagebox.showinfo("Export", "Keine Einträge zum Exportieren vorhanden.")
            return
        stamp = self.run_source.get("timestamp") or datetime.now()
        path = filedialog.asksaveasfilename(
            title="Ergebnisse exportieren", defaultextension=".csv", filetypes=[("CSV-Datei", "*.csv")],
            initialfile=f"benchmark_{stamp:%Y-%m-%d_%H-%M-%S}.csv")
        if not path:
            return
        try:
            export_results_csv(self._result_rows, path)
        except OSError as e:
            messagebox.showerror("Export", f"Der Export ist fehlgeschlagen:\n{e}")
            return
        self._set_status(f"{len(self._result_rows)} Zeilen exportiert:\n{path}", SUCCESS)


    # ------------------------------------------------------------------ #
    #  Reiter „Detail“                                                     #
    # ------------------------------------------------------------------ #
    DETAIL_COLUMNS = (("file", "Datei", 100, False, "w"), ("code", "Ergebnis", 70, True, "w"))
    DETAIL_FIELDS = (
        ("code", "Erkannter Code"), ("expected", "Soll (Ground Truth)"),
        ("method", "Methode"), ("conf", "Konfidenz"),
        ("duration", "Dauer"), ("dmtx", "DataMatrix (DMTX)"),
        ("ocr", "OCR (Klarschrift)"), ("partial", "OCR teilweise"),
        ("reason", "Fehlerursache"), ("previous", "Vorheriger Lauf"),
    )

    def _build_detail(self):
        tab = self.tab_detail
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(tab, width=310, fg_color=BG_CARD, border_width=1, border_color=BORDER, corner_radius=8)
        left.grid(row=0, column=0, sticky="ns", padx=(6, 8), pady=(10, 6))
        left.grid_propagate(False)
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(2, weight=1)
        self.detail_filter = ctk.CTkOptionMenu(left, values=list(RESULT_FILTERS),
                                               command=lambda _v: self._refresh_detail_list(), **MENU_STYLE)
        self.detail_filter.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=(10, 6))
        self.detail_search = ctk.CTkEntry(left, placeholder_text="Suche ...")
        self.detail_search.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 6))
        self.detail_search.bind("<KeyRelease>", self._on_detail_search_key)
        self.detail_tree = ttk.Treeview(left, style="Bench.Treeview", columns=[c[0] for c in self.DETAIL_COLUMNS],
                                        show=("tree", "headings"), selectmode="browse")
        self.detail_tree.grid(row=2, column=0, sticky="nsew", padx=(6, 0), pady=(0, 6))
        scrollbar = ctk.CTkScrollbar(left, command=self.detail_tree.yview)
        scrollbar.grid(row=2, column=1, sticky="ns", padx=(0, 4), pady=(0, 6))
        self.detail_tree.configure(yscrollcommand=scrollbar.set)
        self._setup_tree_columns(self.detail_tree, self.DETAIL_COLUMNS)
        self.detail_tree.heading("#0", text="Status")
        for column, text, _width, _stretch, anchor in self.DETAIL_COLUMNS:
            self.detail_tree.heading(column, text=text, anchor=anchor)
        self.detail_tree.bind("<<TreeviewSelect>>", self._on_detail_tree_select)
        self._bind_tree_hover(self.detail_tree)
        self.detail_count_label = ctk.CTkLabel(left, text="", font=ctk.CTkFont(size=11, weight="bold"), text_color=TXT_MID)
        self.detail_count_label.grid(row=3, column=0, columnspan=2, sticky="w", padx=12, pady=(0, 8))
        self.detail_list_empty = ctk.CTkLabel(left, text="", fg_color=BG_CARD, text_color=TXT_LIGHT,
                                              font=ctk.CTkFont(size=12), wraplength=240)

        right = ctk.CTkFrame(tab, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 6), pady=(10, 6))
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)
        bar = ctk.CTkFrame(right, fg_color=BG_SIDE, border_width=1, border_color=BORDER, corner_radius=8)
        bar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        arrow = {"width": 36, "height": 30, "fg_color": BG_CARD, "text_color": TXT_DARK, "hover_color": BORDER,
                 "border_width": 1, "border_color": BORDER, "font": ctk.CTkFont(size=20, weight="bold")}
        self.detail_prev_btn = ctk.CTkButton(bar, text="‹", command=lambda: self._detail_step(-1), **arrow)
        self.detail_prev_btn.pack(side="left", padx=(10, 4), pady=8)
        self.detail_next_btn = ctk.CTkButton(bar, text="›", command=lambda: self._detail_step(1), **arrow)
        self.detail_next_btn.pack(side="left", padx=(0, 8), pady=8)
        self.detail_pos_label = ctk.CTkLabel(bar, text="", font=ctk.CTkFont(size=12, weight="bold"), text_color=TXT_DARK)
        self.detail_pos_label.pack(side="left", padx=4)
        text_button = {"height": 28, "fg_color": "transparent", "text_color": ACCENT, "hover_color": BORDER,
                       "font": ctk.CTkFont(size=12, weight="bold")}
        ctk.CTkButton(bar, text="Im Bildbetrachter öffnen", command=self._detail_open_external, **text_button
                      ).pack(side="right", padx=(0, 10))
        ctk.CTkButton(bar, text="Zur Annotation", command=self._detail_open_annotation, **text_button
                      ).pack(side="right", padx=4)

        self.detail_scroll = ctk.CTkScrollableFrame(right, fg_color="transparent")
        self.detail_scroll.grid(row=1, column=0, sticky="nsew")
        self.detail_placeholder = ctk.CTkLabel(right, text="", font=ctk.CTkFont(size=13), text_color=TXT_LIGHT)
        self._build_detail_content(self.detail_scroll)

    def _build_detail_content(self, parent):
        card_style = {"fg_color": BG_SIDE, "border_width": 1, "border_color": BORDER, "corner_radius": 8}

        head = ctk.CTkFrame(parent, **card_style)
        head.pack(fill="x", pady=(0, 8))
        title_row = ctk.CTkFrame(head, fg_color="transparent")
        title_row.pack(fill="x", padx=16, pady=(12, 6))
        self.detail_title = ctk.CTkLabel(title_row, text="—", font=ctk.CTkFont(size=20, weight="bold"), text_color=TXT_DARK)
        self.detail_title.pack(side="left")
        self.detail_pill = ctk.CTkLabel(title_row, text="", width=130, height=24, corner_radius=6, fg_color=TXT_LIGHT,
                                        text_color="#FFFFFF", font=ctk.CTkFont(size=11, weight="bold"))
        self.detail_pill.pack(side="right")
        grid = ctk.CTkFrame(head, fg_color="transparent")
        grid.pack(fill="x", padx=16, pady=(0, 12))
        self._detail_grid = grid
        self._detail_layout: tuple | None = None
        self._detail_pairs: list[tuple] = []
        self.detail_fields: dict[str, ctk.CTkLabel] = {}
        for key, title in self.DETAIL_FIELDS:
            name = ctk.CTkLabel(grid, text=f"{title}:", font=ctk.CTkFont(size=11, weight="bold"), text_color=TXT_MID,
                                anchor="nw", width=125)
            value = ctk.CTkLabel(grid, text="—", font=ctk.CTkFont(size=12), text_color=TXT_DARK, anchor="nw",
                                 justify="left", wraplength=190)
            self._detail_pairs.append((name, value))
            self.detail_fields[key] = value
        head.bind("<Configure>", self._layout_detail_fields)

        images = ctk.CTkFrame(parent, fg_color="transparent")
        images.pack(fill="x", pady=(0, 8))
        images.grid_columnconfigure(0, weight=3, uniform="images")
        images.grid_columnconfigure(1, weight=2, uniform="images")
        yolo = ctk.CTkFrame(images, **card_style)
        yolo.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        yolo.bind("<Configure>", lambda e: self.detail_yolo_info.configure(wraplength=max(120, int(e.width / self._scaling) - 32)))
        ctk.CTkLabel(yolo, text="YOLO-Detektionen", font=ctk.CTkFont(size=13, weight="bold"), text_color=TXT_DARK
                     ).grid(row=0, column=0, sticky="w", padx=14, pady=(10, 0))
        self.detail_yolo_info = ctk.CTkLabel(yolo, text="", font=ctk.CTkFont(size=11), text_color=TXT_MID, anchor="w",
                                             justify="left", wraplength=360)
        self.detail_yolo_info.grid(row=1, column=0, sticky="w", padx=14)
        self.detail_yolo_img = ctk.CTkLabel(yolo, text="", cursor="hand2")
        self.detail_yolo_img.grid(row=2, column=0, padx=10, pady=(6, 12))
        self.detail_yolo_img.bind("<Button-1>", lambda _e: self._detail_open_external())

        crops = ctk.CTkFrame(images, **card_style)
        crops.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        crops.bind("<Configure>", lambda e: self.detail_crop_notice.configure(wraplength=max(120, int(e.width / self._scaling) - 32)))
        crops.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(crops, text="Scanner-Crops", font=ctk.CTkFont(size=13, weight="bold"), text_color=TXT_DARK
                     ).grid(row=0, column=0, sticky="w", padx=14, pady=(10, 0))
        self.detail_crop_notice = ctk.CTkLabel(
            crops, text="Keine YOLO-Detektion über der Schwelle – der Scan fällt auf das Gesamtbild zurück.",
            font=ctk.CTkFont(size=11), text_color=TXT_MID, anchor="w", justify="left", wraplength=250)
        self.detail_crop_notice.grid(row=1, column=0, sticky="w", padx=14, pady=(4, 12))
        self.detail_crops: dict[str, tuple] = {}
        for row, (key, color) in enumerate((("dmx", SUCCESS), ("txt", ACCENT)), start=2):
            block = ctk.CTkFrame(crops, fg_color="transparent")
            block.grid(row=row, column=0, sticky="ew", padx=10, pady=(4, 8))
            title = ctk.CTkLabel(block, text="", font=ctk.CTkFont(size=11, weight="bold"), text_color=color)
            title.pack(anchor="w", padx=4)
            image = ctk.CTkLabel(block, text="")
            image.pack(pady=(2, 0))
            self.detail_crops[key] = (block, title, image)
            block.grid_remove()

        pipe = ctk.CTkFrame(parent, **card_style)
        pipe.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(pipe, text="Pipeline-Ablauf", font=ctk.CTkFont(size=13, weight="bold"), text_color=TXT_DARK
                     ).pack(anchor="w", padx=14, pady=(10, 6))
        steps = ctk.CTkFrame(pipe, fg_color="transparent")
        steps.pack(fill="x", padx=14, pady=(0, 12))
        steps.grid_columnconfigure(0, weight=1)
        self.detail_steps = []
        for row in range(5):
            frame = ctk.CTkFrame(steps, fg_color=BG_CARD, border_width=1, border_color=BORDER, corner_radius=6)
            frame.grid(row=row, column=0, sticky="ew", pady=2)
            icon = ctk.CTkLabel(frame, text="", width=26, font=ctk.CTkFont(size=14, weight="bold"))
            icon.pack(side="left", padx=(10, 2), pady=6)
            name = ctk.CTkLabel(frame, text="", font=ctk.CTkFont(size=12), text_color=TXT_MID, anchor="w")
            name.pack(side="left", padx=4)
            result = ctk.CTkLabel(frame, text="", font=ctk.CTkFont(size=12, weight="bold"), anchor="e")
            result.pack(side="right", padx=14)
            self.detail_steps.append((frame, icon, name, result))
        self.detail_images_refs: list = []

    # --- Liste ---
    def _refresh_detail_list(self):
        self._dirty.discard("detail")
        mode = self.detail_filter.get()
        query = self.detail_search.get().strip().lower()
        rows = [r for r in self.results
                if self._matches_filter(r, mode) and (not query or query in self._search_text(r))]
        keep = self._detail_pending or (self._detail_current["filename"] if self._detail_current else "")
        self._detail_pending = None
        tree = self.detail_tree
        tree.delete(*tree.get_children())
        self._hover_iid[str(tree)] = ""
        self._detail_rows = {}
        for r in rows:
            kind, delta = outcome(r), self.diff.get(r["filename"], "")
            tags = [t for t in (kind if kind != "ok" else "", delta if delta in ("regression", "improved") else "") if t]
            tree.insert("", "end", iid=r["filename"], image=self._badges[kind], tags=tags,
                        values=(r["filename"], display_code(r)))
            self._detail_rows[r["filename"]] = r
        self.detail_count_label.configure(text=f"{len(rows)} von {len(self.results)} Bildern")
        if rows:
            self.detail_list_empty.place_forget()
            self.detail_placeholder.grid_forget()
            self.detail_scroll.grid(row=1, column=0, sticky="nsew")
            self._detail_select(keep if keep in self._detail_rows else rows[0]["filename"])
            return
        self._detail_current = None
        self.detail_scroll.grid_forget()
        self.detail_pos_label.configure(text="")
        self.detail_placeholder.configure(text="Noch kein Lauf geladen.\nStarte einen Benchmark oder wähle links einen gespeicherten Bericht."
                                          if not self.results else "Keine Einträge für diesen Filter.")
        self.detail_placeholder.grid(row=1, column=0)
        self.detail_list_empty.configure(text="Keine Einträge." if self.results else "")
        self.detail_list_empty.place(relx=0.5, rely=0.35, anchor="center")

    def _on_detail_search_key(self, _event=None):
        if self._search_job is not None:
            self.after_cancel(self._search_job)
        self._search_job = self.after(200, self._refresh_detail_list)

    def _detail_select(self, iid: str):
        tree = self.detail_tree
        tree.selection_set(iid)
        tree.focus(iid)
        tree.see(iid)
        self._load_detail(self._detail_rows[iid])

    def _on_detail_tree_select(self, _event=None):
        selection = self.detail_tree.selection()
        entry = self._detail_rows.get(selection[0]) if selection else None
        if entry is not None and entry is not self._detail_current:
            self._load_detail(entry)

    def _detail_step(self, step: int):
        ids = list(self._detail_rows)
        if self._detail_current is None or not ids:
            return
        position = ids.index(self._detail_current["filename"]) if self._detail_current["filename"] in ids else 0
        if 0 <= position + step < len(ids):
            self._detail_select(ids[position + step])

    def _show_detail_for(self, filename: str):
        """Öffnet die Detail-Ansicht für ein Bild; ein Filter, der das Bild ausblendet, wird zurückgesetzt."""
        entry = self._result_by_file.get(filename)
        if entry is None:
            return
        query = self.detail_search.get().strip().lower()
        if not (self._matches_filter(entry, self.detail_filter.get()) and (not query or query in self._search_text(entry))):
            self.detail_filter.set(RESULT_FILTERS[0])
            self.detail_search.delete(0, "end")
        self._detail_pending = filename
        self._dirty.add("detail")
        self._goto_tab("Detail")

    def _detail_open_annotation(self):
        if self._detail_current is not None:
            self._open_annotation(self._detail_current["path"])

    def _detail_open_external(self):
        entry = self._detail_current
        if entry is not None and os.path.isfile(entry["path"]):
            os.startfile(entry["path"])

    # --- Anzeige eines Bildes ---
    def _detail_set(self, key: str, text: str, color: str = TXT_DARK):
        self.detail_fields[key].configure(text=text, text_color=color)

    def _load_detail(self, entry: dict):
        self._detail_current = entry
        self._fill_detail_fields(entry)
        self._detail_request += 1
        request = self._detail_request
        if self._detail_job is not None:
            self.after_cancel(self._detail_job)
        self._detail_job = self.after(60, lambda: self._start_detail_worker(request, entry))

    @staticmethod
    def _detection_label(det: dict) -> str:
        return f"{'DMX' if det['cls'] == 0 else 'TXT'} ({det['conf']:.0%})"

    def _fill_detail_fields(self, entry: dict):
        kind = outcome(entry)
        pill = {"ok": ("KORREKT" if entry["expected"] else "GELESEN", SUCCESS), "wrong": ("FALSCH GELESEN", WARN),
                "fail": ("NICHT GELESEN", DANGER)}[kind]
        self.detail_title.configure(text=entry["filename"])
        self.detail_pill.configure(text=pill[0], fg_color=pill[1])
        self._detail_set("code", entry["result_code"] if entry["success"] else "—", OUTCOME_COLORS[kind])
        self._detail_set("expected", entry["expected"] or "Nicht annotiert", TXT_DARK if entry["expected"] else TXT_LIGHT)
        self._detail_set("method", entry["method"], ACCENT)
        self._detail_set("conf", f"{entry['confidence']:.1%}".replace(".", ","))
        self._detail_set("duration", f"{_fmt_int(entry['duration_ms'])} ms")
        self._detail_set("dmtx", entry["dmtx_result"] or "Nicht erkannt", TXT_DARK if entry["dmtx_result"] else TXT_LIGHT)
        ocr = entry["ocr_result"] or ("Übersprungen (DMX dekodiert)" if entry["dmtx_result"] else "Nicht erkannt")
        self._detail_set("ocr", ocr, TXT_DARK if entry["ocr_result"] else TXT_LIGHT)
        self._detail_set("partial", entry["ocr_partial"] or "—", TXT_DARK if entry["ocr_partial"] else TXT_LIGHT)
        reasons = [entry["fail_reason"]] if entry["fail_reason"] else []
        if not entry["success"] and entry["result_code"]:
            reasons.append(entry["result_code"])
        if entry.get("error"):
            reasons.append(entry["error"])
        self._detail_set("reason", "\n".join(reasons) or "—", DANGER if reasons else TXT_LIGHT)

        previous = self._compare_by_file.get(entry["filename"])
        if not self._compare_by_file:
            self._detail_set("previous", "Kein Vergleichslauf", TXT_LIGHT)
        elif previous is None:
            self._detail_set("previous", "Nicht im Vergleichslauf", TXT_LIGHT)
        else:
            delta = self.diff.get(entry["filename"], "")
            color = {"regression": DANGER, "improved": SUCCESS, "changed": WARN}.get(delta, TXT_DARK)
            self._detail_set("previous", f"{display_code(previous)}  ({status_text(previous)})"
                             + (f"\n{DIFF_TEXT[delta]}" if delta else ""), color)

        dets = entry["detections"]
        summary = f"{len(dets)} Detektion{'en' if len(dets) != 1 else ''}"
        self.detail_yolo_info.configure(
            text=summary + (": " + ", ".join(self._detection_label(d) for d in dets) if dets else " gefunden"),
            text_color=TXT_MID)
        steps = pipeline_steps(entry)
        for index, (frame, icon, name, result) in enumerate(self.detail_steps):
            if index < len(steps):
                step, outcome_text, level = steps[index]
                frame.grid()
                icon.configure(text=LEVEL_ICONS[level], text_color=LEVEL_COLORS[level])
                name.configure(text=step)
                result.configure(text=outcome_text, text_color=LEVEL_COLORS[level] if level != "muted" else TXT_MID)
            else:
                frame.grid_remove()

        ids = list(self._detail_rows)
        if entry["filename"] in ids:
            position = ids.index(entry["filename"])
            self.detail_pos_label.configure(text=f"{position + 1} / {len(ids)}")
            self.detail_prev_btn.configure(state="normal" if position > 0 else "disabled")
            self.detail_next_btn.configure(state="normal" if position < len(ids) - 1 else "disabled")

    def _layout_detail_fields(self, event):
        """Zwei Spalten Titel/Wert bei breitem Bereich, sonst eine Spalte (bei 1100 px Fensterbreite wurden Werte abgeschnitten)."""
        width = event.width / self._scaling
        columns = 2 if width >= 640 else 1
        wrap = max(80, int((width - 32 - columns * (125 + 12)) / columns))
        if (columns, wrap) == self._detail_layout:
            return
        self._detail_layout = (columns, wrap)
        grid = self._detail_grid
        for column in range(4):
            active = column % 2 == 1 and column < columns * 2
            grid.grid_columnconfigure(column, weight=1 if active else 0, uniform="fields" if active else "")
        for index, (name, value) in enumerate(self._detail_pairs):
            row, col = divmod(index, columns)
            name.grid(row=row, column=col * 2, sticky="nw", pady=2)
            value.grid(row=row, column=col * 2 + 1, sticky="nw", pady=2, padx=(0, 12))
            value.configure(wraplength=wrap)

    def _start_detail_worker(self, request: int, entry: dict):
        self._detail_job = None
        self.detail_scroll.update_idletasks()  # Breite erst nach dem Layout messen (Reiter wurde gerade geöffnet)
        scale = self._scaling
        inner = max(int(320 * scale), self.detail_scroll.winfo_width() - int(50 * scale))
        margin = int(28 * scale)
        boxes = {"yolo": (int(inner * 0.6) - margin, int(360 * scale)),
                 "dmx": (int(inner * 0.4) - margin, int(220 * scale)),
                 "txt": (int(inner * 0.4) - margin, int(110 * scale))}
        threading.Thread(target=self._detail_worker, args=(request, entry, boxes), daemon=True).start()

    def _detail_worker(self, request: int, entry: dict, boxes: dict):
        """Hintergrund: Bild laden, Boxen einzeichnen und die Scanner-Crops ausschneiden."""
        try:
            image = read_image(entry["path"])
            if image is None:
                raise ValueError(f"Bild nicht lesbar: {entry['path']}")
            data = {"yolo": fit_image(cv2_to_pil(draw_boxes_on_image(image, entry["detections"])), *boxes["yolo"])}
            dmx_det, txt_det = scanner.select_label_detections(entry["detections"])
            for key, det, padding in (("dmx", dmx_det, 40), ("txt", txt_det, 30)):
                if det is not None:
                    crop = scanner.deskew_crop(image, det["box"], padding=padding)
                    data[key] = (fit_image(cv2_to_pil(crop), *boxes[key]), det)
            payload = (request, data, None)
        except Exception as e:
            logger.exception(f"Detail-Ansicht für {entry['filename']} fehlgeschlagen")
            payload = (request, None, f"{type(e).__name__}: {e}")
        self._events.put(("detail_images", payload))

    def _on_detail_images(self, payload: tuple):
        request, data, error = payload
        if request != self._detail_request:
            return
        scale = self._scaling

        def to_ctk(image: Image.Image) -> ctk.CTkImage:
            return ctk.CTkImage(light_image=image, dark_image=image, size=(image.width / scale, image.height / scale))

        if data is None:
            self.detail_yolo_info.configure(text=f"Bild konnte nicht verarbeitet werden: {error}", text_color=DANGER)
            self.detail_yolo_img.grid_remove()
            for block, _title, _image in self.detail_crops.values():
                block.grid_remove()
            self.detail_crop_notice.grid()
            return
        refs = []
        yolo = to_ctk(data["yolo"])
        refs.append(yolo)
        self.detail_yolo_img.configure(image=yolo)
        self.detail_yolo_img.grid()
        shown = False
        for key, label in (("dmx", "DataMatrix-Crop"), ("txt", "Klarschrift-Crop")):
            block, title, image_label = self.detail_crops[key]
            if key not in data:
                block.grid_remove()
                continue
            crop, det = data[key]
            shown = True
            source = "abgeleitet" if det.get("derived") else f"{det['conf']:.0%}"
            title.configure(text=f"{label} ({source})")
            crop_image = to_ctk(crop)
            refs.append(crop_image)
            image_label.configure(image=crop_image)
            block.grid()
        if shown:
            self.detail_crop_notice.grid_remove()
        else:
            self.detail_crop_notice.grid()
        self.detail_images_refs = refs  # CTkImage-Referenzen halten, sonst verschwinden die Bilder


# ═══════════════════════════════════════════════════════════════════════════════
#  Start
# ═══════════════════════════════════════════════════════════════════════════════

def _use_utf8_output():
    """Umlaute und Symbole auch bei umgeleiteter Ausgabe (dort gilt unter Windows sonst cp1252)."""
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def run_headless(image_dir: str, gt_path: str, fail_on_regression: bool = False, model_path: str | None = None) -> int:
    """Benchmark ohne GUI: schreibt denselben Bericht wie die GUI und vergleicht mit dem vorherigen Lauf → Exit-Code."""
    image_paths = load_image_list(image_dir)
    if not image_paths:
        print(f"Keine Bilder gefunden in {image_dir}")
        return 2
    names = {os.path.basename(p) for p in image_paths}
    baseline = next((r for r in list_reports(image_dir) if report_fits(r, names)), None)  # vor dem neuen Bericht
    if model_path:
        from ultralytics import YOLO
        print(f"YOLO-Modell: {model_path}")
        model = YOLO(model_path)
    else:
        model, _ = yolo_detector.load_model(APP_DIR)
    try:
        results, summary = run_benchmark(model, image_paths, load_ground_truth(gt_path))
    except KeyboardInterrupt:
        print("\nAbgebrochen – es wurde kein Bericht gespeichert.")
        return 130
    print(f"\nBericht gespeichert: {save_report(results, summary, image_dir, gt_path)}")
    for line in format_summary_lines(summary):
        print(line)

    regressions = 0
    if baseline is not None:
        diff = compare_runs(results, baseline["results"])
        regressions = sum(1 for kind in diff.values() if kind == "regression")
        for line in format_compare_lines(diff, results, baseline["results"], report_title(baseline)):
            print(line)
    if fail_on_regression and (regressions or summary["false_read_count"]):
        print(f"\nFEHLER: {regressions} Regression(en), {summary['false_read_count']} falsch gelesen.")
        return 1
    return 0


def main():
    _use_utf8_output()
    parser = argparse.ArgumentParser(description="DataDetector Benchmark & Ground-Truth-Tool")
    parser.add_argument("--images", default=IMAGE_DIR, help="Bildordner (Standard: training_data)")
    parser.add_argument("--gt", default=GROUND_TRUTH_PATH, help="Ground-Truth-Datei (Standard: ground_truth.json)")
    parser.add_argument("--headless", action="store_true",
                        help="Benchmark ohne GUI ausführen, Bericht speichern und mit dem vorherigen Lauf vergleichen")
    parser.add_argument("--fail-on-regression", action="store_true",
                        help="mit --headless: Exit-Code 1 bei Regressionen oder falsch gelesenen Codes")
    parser.add_argument("--model", default=None,
                        help="mit --headless: anderes YOLO-Modell (.pt) statt des Produktionsmodells testen")
    args = parser.parse_args()
    images_dir = os.path.abspath(args.images)
    gt_file = os.path.abspath(args.gt)
    if args.headless:
        sys.exit(run_headless(images_dir, gt_file, args.fail_on_regression, args.model))
    app = BenchmarkApp(images_dir, gt_file)
    app.mainloop()
    if app.run_active:
        os._exit(0)  # laufende Pipeline-Threads (Torch) nicht abwarten


if __name__ == "__main__":
    main()
