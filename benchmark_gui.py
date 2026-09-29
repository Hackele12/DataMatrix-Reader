"""
benchmark_gui.py — Interaktives Benchmark & Ground-Truth-Annotationstool
für DataDetector.

Features:
  1. Übersicht aller Testbilder aus training_data/ (oder --images <Ordner> --gt <datei.json>)
  2. Ground-Truth-Editor: Bild-für-Bild den tatsächlichen Code annotieren
  3. Benchmark starten mit Fortschrittsbalken (ohne GUI: --headless)
  4. Detaillierte Ergebnis-Analyse (Erfolgsrate, Genauigkeit, Fehlerverteilung)
  5. Per-Image-Detailansicht: YOLO-Boxen, Crops, Scan-Ergebnis, Pipeline-Schritte

Der Benchmark nutzt dieselbe Pipeline wie die Produktion: YOLO-Detektion → scanner.scan_2class().
"""

import argparse
import glob
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime

# --- KMP Fix ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import tkinter as tk
import customtkinter as ctk
from PIL import Image

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

# --- Farben (Dark Theme) ---
BG_DARK = "#0f0f13"
BG_CARD = "#1a1a24"
BG_CARD_HOVER = "#22223a"
BG_INPUT = "#262637"
ACCENT_PRIMARY = "#6c5ce7"
ACCENT_SECONDARY = "#a29bfe"
ACCENT_SUCCESS = "#00b894"
ACCENT_DANGER = "#e17055"
ACCENT_WARNING = "#fdcb6e"
ACCENT_INFO = "#74b9ff"
TEXT_PRIMARY = "#f0f0f8"
TEXT_SECONDARY = "#9e9ebd"
TEXT_MUTED = "#6c6c8a"
BORDER_COLOR = "#2d2d44"
PROGRESS_BG = "#1e1e30"


# ═══════════════════════════════════════════════════════════════════════════════
#  Ground-Truth & Testbilder
# ═══════════════════════════════════════════════════════════════════════════════

def load_ground_truth(path: str) -> dict:
    """Lädt die Ground-Truth-Datei (Dateiname → Code)."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Fehler beim Laden der Ground Truth: {e}")
    return {}


def save_ground_truth(gt_map: dict, path: str):
    """Speichert die Ground-Truth-Datei."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(gt_map, f, indent=2, ensure_ascii=False)
    logger.info(f"Ground Truth gespeichert: {len(gt_map)} Einträge.")


def load_image_list(image_dir: str) -> list[str]:
    """Alle Bilder (jpg, jpeg, png, bmp) des Ordners, sortiert."""
    files = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        files.extend(glob.glob(os.path.join(image_dir, pattern)))
    return sorted(files)


# ═══════════════════════════════════════════════════════════════════════════════
#  Benchmark-Kern (GUI und --headless)
# ═══════════════════════════════════════════════════════════════════════════════

def _failure_reason(detections: list[dict], scan_res: dict) -> str:
    if not detections:
        return "YOLO_NO_DETECTION"
    if not scan_res.get("dmtx_result") and not scan_res.get("ocr_result"):
        return "DMTX_AND_OCR_FAILED"
    if scan_res.get("ocr_result") and not scan_res.get("dmtx_result"):
        return "RECONSTRUCTION_REJECTED"
    return "UNKNOWN"


def benchmark_image(model, path: str, gt_map: dict, index: int) -> dict | None:
    """Ein Bild durch YOLO + scanner.scan_2class(); None, wenn das Bild nicht lesbar ist."""
    image = cv2.imread(path)
    if image is None:
        return None
    fname = os.path.basename(path)

    t0 = time.time()
    yolo_result = model.predict(image, conf=yolo_detector.PREDICT_CONF, verbose=False)[0]
    detections = yolo_detector.extract_detections(yolo_result)
    scan_res = scanner.scan_2class(image, detections)
    duration_ms = int((time.time() - t0) * 1000)

    is_success = scan_res.get("success", False)
    result_code = scan_res.get("result", "")
    expected = gt_map.get(fname, None)
    return {
        "index": index,
        "filename": fname,
        "path": path,
        "success": is_success,
        "result_code": result_code,
        "expected": expected,
        "is_match": bool(expected) and is_success and result_code == expected,
        "method": scan_res.get("method", "Unbekannt"),
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
    failure_cats = {}
    for r in results:
        if r["success"]:
            method_dist[r["method"]] = method_dist.get(r["method"], 0) + 1
        else:
            failure_cats[r["fail_reason"]] = failure_cats.get(r["fail_reason"], 0) + 1

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
        "failure_categories": failure_cats,
    }


def format_result_line(entry: dict, total: int) -> str:
    status = "✅" if entry["success"] else "❌"
    match_str = f" ({'✓' if entry['is_match'] else '✗'})" if entry["expected"] else ""
    return (
        f"  {status} [{entry['index']:>3}/{total}] {entry['filename']:<14} "
        f"→ {entry['result_code'] if entry['success'] else 'FAIL':<8} "
        f"({entry['method']}, {entry['duration_ms']}ms){match_str}"
    )


def format_summary_lines(summary: dict) -> list[str]:
    s = summary
    return [
        f"\n{'═' * 60}",
        "  BENCHMARK ABGESCHLOSSEN",
        f"{'═' * 60}",
        f"  Bilder:         {s['total_images']}",
        f"  Erfolgsrate:    {s['success_count']}/{s['total_images']} ({s['success_rate_pct']:.1f}%)",
        f"  Genauigkeit GT: {s['accuracy_count']}/{s['accuracy_total_gt']} ({s['accuracy_pct']:.1f}%)",
        f"  Falsch gelesen: {s['false_read_count']}",
        f"  Nicht gelesen:  {s['no_read_count']}",
        f"  Ø Dauer/Bild:   {s['avg_duration_ms']:.0f} ms",
        f"  Gesamtzeit:     {s['total_time_s']:.1f}s",
        f"{'═' * 60}",
    ]


def run_benchmark(model, image_paths: list[str], gt_map: dict, log=print, on_progress=None) -> tuple[list, dict]:
    """
    Führt den Benchmark über alle Bilder aus.

    Args:
        log: Ausgabe für Protokollzeilen.
        on_progress: Optionaler Callback (bild_index, gesamt, ergebnisse, laufzeit_s) nach jedem Bild.
    """
    total = len(image_paths)
    results = []
    log(f"\n🚀 Benchmark gestartet: {total} Bilder\n{'─' * 60}")
    start_time = time.time()

    for idx, path in enumerate(image_paths):
        entry = benchmark_image(model, path, gt_map, idx + 1)
        if entry is None:
            log(f"  ⚠️  {os.path.basename(path)}: Bild konnte nicht geladen werden")
            continue
        results.append(entry)
        log(format_result_line(entry, total))
        if on_progress:
            on_progress(idx, total, results, time.time() - start_time)

    return results, summarize(results, total, time.time() - start_time)


def save_report(results: list[dict], summary: dict, image_dir: str, gt_path: str) -> str:
    """Schreibt den Bericht nach benchmark_reports/ (mit Datum, Zeit & Version) und als benchmark_baseline.json."""
    now_dt = datetime.now()
    report = {
        "version": BENCHMARK_VERSION,
        "timestamp": now_dt.isoformat(),
        "image_dir": image_dir,
        "ground_truth_file": gt_path,
        "summary": summary,
        "details": [{k: v for k, v in r.items() if k != "scan_res"} for r in results],
    }
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, f"benchmark_{now_dt.strftime('%Y-%m-%d_%H-%M-%S')}_v{BENCHMARK_VERSION}.json")
    for path in (report_path, BASELINE_PATH):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    return report_path


# ═══════════════════════════════════════════════════════════════════════════════
#  Bild-Hilfsfunktionen
# ═══════════════════════════════════════════════════════════════════════════════


def cv2_to_pil(cv_img: np.ndarray) -> Image.Image:
    """Konvertiert ein OpenCV-Bild (BGR) in ein PIL-Bild (RGB)."""
    if len(cv_img.shape) == 2:
        return Image.fromarray(cv_img)
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


def fit_image(pil_img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    """Skaliert ein PIL-Bild proportional in den gegebenen Rahmen."""
    w, h = pil_img.size
    ratio = min(max_w / w, max_h / h)
    new_w = max(1, int(w * ratio))
    new_h = max(1, int(h * ratio))
    return pil_img.resize((new_w, new_h), Image.LANCZOS)


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

        # Label-Hintergrund
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return annotated


# ═══════════════════════════════════════════════════════════════════════════════
#  Hauptapplikation
# ═══════════════════════════════════════════════════════════════════════════════

class BenchmarkApp(ctk.CTk):
    """Hauptfenster der Benchmark-Applikation."""

    def __init__(self, image_dir: str = IMAGE_DIR, gt_path: str = GROUND_TRUTH_PATH):
        super().__init__()

        # --- Fenster-Setup ---
        self.title("DataDetector — Benchmark & Annotation Tool")
        self.geometry("1400x900")
        self.minsize(1200, 800)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.configure(fg_color=BG_DARK)

        # --- Daten ---
        self.image_dir = image_dir
        self.gt_path = gt_path
        self.image_paths = load_image_list(image_dir)
        self.ground_truth = load_ground_truth(gt_path)
        self.benchmark_results = []
        self.benchmark_summary = {}
        self.benchmark_running = False

        # --- Navigation Stack ---
        self.frames: dict[str, ctk.CTkFrame] = {}
        self.container = ctk.CTkFrame(self, fg_color=BG_DARK)
        self.container.pack(fill="both", expand=True)

        self._build_sidebar()
        self._build_pages()
        self._show_page("overview")

    # ─────────────────────────────────────────────────────────────────────────
    #  Sidebar
    # ─────────────────────────────────────────────────────────────────────────

    def _build_sidebar(self):
        """Erstellt die linke Navigationsleiste."""
        self.sidebar = ctk.CTkFrame(
            self.container, width=220, fg_color=BG_CARD,
            corner_radius=0,
        )
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        # Logo / Titel
        title_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        title_frame.pack(fill="x", pady=(25, 5), padx=15)

        ctk.CTkLabel(
            title_frame, text="⬡", font=("Segoe UI", 32),
            text_color=ACCENT_PRIMARY,
        ).pack(side="left", padx=(0, 8))

        title_text = ctk.CTkFrame(title_frame, fg_color="transparent")
        title_text.pack(side="left")
        ctk.CTkLabel(
            title_text, text="DataDetector", font=("Segoe UI Semibold", 16),
            text_color=TEXT_PRIMARY,
        ).pack(anchor="w")
        ctk.CTkLabel(
            title_text, text="Benchmark Suite", font=("Segoe UI", 11),
            text_color=TEXT_MUTED,
        ).pack(anchor="w")

        # Separator
        ctk.CTkFrame(
            self.sidebar, height=1, fg_color=BORDER_COLOR
        ).pack(fill="x", padx=15, pady=15)

        # Nav Buttons
        self.nav_buttons = {}
        nav_items = [
            ("overview", "📊", "Übersicht"),
            ("annotate", "✏️", "Ground Truth"),
            ("benchmark", "🚀", "Benchmark"),
            ("results", "📈", "Ergebnisse"),
            ("detail", "🔍", "Detail-Ansicht"),
        ]

        for page_id, icon, label in nav_items:
            btn = ctk.CTkButton(
                self.sidebar, text=f"  {icon}  {label}",
                font=("Segoe UI", 14), height=42,
                fg_color="transparent", text_color=TEXT_SECONDARY,
                hover_color=BG_CARD_HOVER, anchor="w",
                corner_radius=10,
                command=lambda pid=page_id: self._show_page(pid),
            )
            btn.pack(fill="x", padx=10, pady=2)
            self.nav_buttons[page_id] = btn

        # Footer Info
        spacer = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        spacer.pack(fill="both", expand=True)

        self.status_label = ctk.CTkLabel(
            self.sidebar, text="", font=("Segoe UI", 11),
            text_color=TEXT_MUTED, wraplength=190,
        )
        self.status_label.pack(padx=15, pady=(0, 20))
        self._update_status_label()

    def _update_status_label(self):
        """Aktualisiert die Statusanzeige in der Sidebar."""
        n_images = len(self.image_paths)
        n_gt = sum(1 for f in self.image_paths
                   if os.path.basename(f) in self.ground_truth)
        self.status_label.configure(
            text=f"📁 {n_images} Bilder\n✅ {n_gt}/{n_images} annotiert"
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  Seiten-Verwaltung
    # ─────────────────────────────────────────────────────────────────────────

    def _build_pages(self):
        """Erstellt alle Seiten."""
        self.main_area = ctk.CTkFrame(self.container, fg_color=BG_DARK)
        self.main_area.pack(side="right", fill="both", expand=True)

        self._build_overview_page()
        self._build_annotate_page()
        self._build_benchmark_page()
        self._build_results_page()
        self._build_detail_page()

    def _show_page(self, page_id: str):
        """Zeigt eine bestimmte Seite an."""
        # Sidebar Buttons aktualisieren
        for pid, btn in self.nav_buttons.items():
            if pid == page_id:
                btn.configure(fg_color=ACCENT_PRIMARY, text_color=TEXT_PRIMARY)
            else:
                btn.configure(fg_color="transparent", text_color=TEXT_SECONDARY)

        # Frames
        for fid, frame in self.frames.items():
            frame.pack_forget()

        if page_id in self.frames:
            self.frames[page_id].pack(fill="both", expand=True)

        # Callbacks bei Seitenwechsel
        if page_id == "results" and self.benchmark_results:
            self._populate_results()
        if page_id == "detail" and self.benchmark_results:
            self._populate_detail_list()

    # ─────────────────────────────────────────────────────────────────────────
    #  Seite 1: Übersicht
    # ─────────────────────────────────────────────────────────────────────────

    def _build_overview_page(self):
        """Erstellt die Übersichtsseite."""
        frame = ctk.CTkFrame(self.main_area, fg_color=BG_DARK)
        self.frames["overview"] = frame

        # Header
        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.pack(fill="x", padx=30, pady=(25, 15))

        ctk.CTkLabel(
            header, text="Benchmark Übersicht",
            font=("Segoe UI Semibold", 26), text_color=TEXT_PRIMARY,
        ).pack(anchor="w")
        ctk.CTkLabel(
            header, text="Testbilder und Ground-Truth-Status auf einen Blick",
            font=("Segoe UI", 13), text_color=TEXT_MUTED,
        ).pack(anchor="w", pady=(3, 0))

        # Stats Cards
        cards_frame = ctk.CTkFrame(frame, fg_color="transparent")
        cards_frame.pack(fill="x", padx=30, pady=10)

        n_images = len(self.image_paths)
        n_gt = sum(1 for f in self.image_paths
                   if os.path.basename(f) in self.ground_truth)
        pct = (n_gt / n_images * 100) if n_images > 0 else 0

        stats = [
            ("📁", "Testbilder", str(n_images), ACCENT_INFO),
            ("✅", "Annotiert", f"{n_gt}/{n_images}", ACCENT_SUCCESS),
            ("📊", "Abdeckung", f"{pct:.0f}%", ACCENT_PRIMARY),
            ("⏱️", "Letzter Run", self._get_last_run_info(), ACCENT_WARNING),
        ]

        for i, (icon, label, value, color) in enumerate(stats):
            card = ctk.CTkFrame(
                cards_frame, fg_color=BG_CARD, corner_radius=16,
                height=110,
            )
            card.pack(side="left", fill="x", expand=True,
                      padx=(0 if i == 0 else 8, 0))
            card.pack_propagate(False)

            inner = ctk.CTkFrame(card, fg_color="transparent")
            inner.place(relx=0.5, rely=0.5, anchor="center")

            ctk.CTkLabel(
                inner, text=f"{icon} {label}",
                font=("Segoe UI", 12), text_color=TEXT_MUTED,
            ).pack()
            ctk.CTkLabel(
                inner, text=value,
                font=("Segoe UI Bold", 28), text_color=color,
            ).pack(pady=(4, 0))

        # Bild-Grid (Thumbnails)
        grid_label = ctk.CTkLabel(
            frame, text=f"Bildvorschau  ({os.path.basename(self.image_dir)}/)",
            font=("Segoe UI Semibold", 16), text_color=TEXT_PRIMARY,
        )
        grid_label.pack(anchor="w", padx=30, pady=(20, 8))

        self.overview_scroll = ctk.CTkScrollableFrame(
            frame, fg_color=BG_CARD, corner_radius=12,
        )
        self.overview_scroll.pack(fill="both", expand=True, padx=30, pady=(0, 20))

        # Thumbnails in einem Grid laden (in Thread)
        threading.Thread(target=self._load_overview_thumbnails, daemon=True).start()

    def _get_last_run_info(self) -> str:
        """Gibt Info zum letzten Benchmark-Run zurück."""
        if os.path.exists(BASELINE_PATH):
            try:
                with open(BASELINE_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                ts = data.get("timestamp", "")
                if ts:
                    dt = datetime.fromisoformat(ts)
                    return dt.strftime("%d.%m.%Y")
            except Exception:
                pass
        return "—"

    def _load_overview_thumbnails(self):
        """Lädt Thumbnails in einem Hintergrund-Thread."""
        cols = 8
        thumb_size = 120

        for idx, path in enumerate(self.image_paths):
            try:
                img = Image.open(path)
                img = fit_image(img, thumb_size, thumb_size)
                fname = os.path.basename(path)
                has_gt = fname in self.ground_truth

                # Thumbnail-Frame
                row = idx // cols
                col = idx % cols

                ctk_img = ctk.CTkImage(light_image=img, dark_image=img,
                                       size=(img.width, img.height))

                def _create_thumb(r, c, ct_image, filename, annotated):
                    thumb_frame = ctk.CTkFrame(
                        self.overview_scroll, fg_color=BG_INPUT,
                        corner_radius=8, width=thumb_size + 10,
                        height=thumb_size + 35,
                    )
                    thumb_frame.grid(row=r, column=c, padx=4, pady=4, sticky="nsew")
                    thumb_frame.grid_propagate(False)

                    lbl = ctk.CTkLabel(thumb_frame, image=ct_image, text="")
                    lbl._ct_img_ref = ct_image
                    lbl.pack(pady=(5, 0))

                    status_color = ACCENT_SUCCESS if annotated else TEXT_MUTED
                    status_icon = "✓" if annotated else "·"
                    ctk.CTkLabel(
                        thumb_frame, text=f"{status_icon} {filename}",
                        font=("Segoe UI", 9), text_color=status_color,
                    ).pack(pady=(2, 0))

                self.after(0, _create_thumb, row, col, ctk_img, fname, has_gt)

            except Exception as e:
                logger.warning(f"Thumbnail-Fehler für {path}: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    #  Seite 2: Ground-Truth Annotation
    # ─────────────────────────────────────────────────────────────────────────

    def _build_annotate_page(self):
        """Erstellt die Annotationsseite."""
        frame = ctk.CTkFrame(self.main_area, fg_color=BG_DARK)
        self.frames["annotate"] = frame

        self.annotate_index = 0

        # Header
        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.pack(fill="x", padx=30, pady=(25, 15))

        ctk.CTkLabel(
            header, text="Ground Truth Editor",
            font=("Segoe UI Semibold", 26), text_color=TEXT_PRIMARY,
        ).pack(anchor="w")
        ctk.CTkLabel(
            header, text="Tippe den Code ein, der auf dem Etikett steht (4 Zeichen)",
            font=("Segoe UI", 13), text_color=TEXT_MUTED,
        ).pack(anchor="w", pady=(3, 0))

        # Fortschritt
        progress_frame = ctk.CTkFrame(frame, fg_color=BG_CARD, corner_radius=12, height=50)
        progress_frame.pack(fill="x", padx=30, pady=(0, 10))
        progress_frame.pack_propagate(False)

        progress_inner = ctk.CTkFrame(progress_frame, fg_color="transparent")
        progress_inner.place(relx=0.5, rely=0.5, anchor="center")

        self.annotate_progress_label = ctk.CTkLabel(
            progress_inner, text="Bild 1 / 172",
            font=("Segoe UI Semibold", 14), text_color=TEXT_PRIMARY,
        )
        self.annotate_progress_label.pack(side="left", padx=15)

        self.annotate_progress_bar = ctk.CTkProgressBar(
            progress_inner, width=400, height=12,
            fg_color=PROGRESS_BG, progress_color=ACCENT_PRIMARY,
            corner_radius=6,
        )
        self.annotate_progress_bar.pack(side="left", padx=10)
        self.annotate_progress_bar.set(0)

        self.annotate_annotated_count = ctk.CTkLabel(
            progress_inner, text="",
            font=("Segoe UI", 12), text_color=ACCENT_SUCCESS,
        )
        self.annotate_annotated_count.pack(side="left", padx=15)

        # Content Area
        content = ctk.CTkFrame(frame, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=30, pady=10)

        # Bild-Anzeige (links)
        img_frame = ctk.CTkFrame(content, fg_color=BG_CARD, corner_radius=16)
        img_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.annotate_image_label = ctk.CTkLabel(
            img_frame, text="Bild wird geladen...",
            font=("Segoe UI", 14), text_color=TEXT_MUTED,
        )
        self.annotate_image_label.pack(fill="both", expand=True, padx=10, pady=10)

        # Eingabe-Panel (rechts)
        input_panel = ctk.CTkFrame(content, fg_color=BG_CARD, corner_radius=16, width=320)
        input_panel.pack(side="right", fill="y", padx=(10, 0))
        input_panel.pack_propagate(False)

        input_inner = ctk.CTkFrame(input_panel, fg_color="transparent")
        input_inner.pack(fill="both", expand=True, padx=20, pady=20)

        # Dateiname
        self.annotate_filename_label = ctk.CTkLabel(
            input_inner, text="",
            font=("Segoe UI Semibold", 16), text_color=TEXT_PRIMARY,
        )
        self.annotate_filename_label.pack(pady=(10, 5))

        # Aktueller GT
        self.annotate_current_gt = ctk.CTkLabel(
            input_inner, text="Aktuelle Annotation: —",
            font=("Segoe UI", 12), text_color=TEXT_MUTED,
        )
        self.annotate_current_gt.pack(pady=(0, 20))

        # Eingabefeld
        ctk.CTkLabel(
            input_inner, text="Code eingeben:",
            font=("Segoe UI Semibold", 13), text_color=TEXT_SECONDARY,
        ).pack(anchor="w")

        self.annotate_entry = ctk.CTkEntry(
            input_inner, height=55, font=("Consolas", 28),
            fg_color=BG_INPUT, border_color=ACCENT_PRIMARY,
            text_color=TEXT_PRIMARY, placeholder_text="z.B. W032",
            justify="center", corner_radius=12,
        )
        self.annotate_entry.pack(fill="x", pady=(8, 15))
        self.annotate_entry.bind("<Return>", lambda e: self._annotate_save_and_next())

        # Speichern-Button
        self.annotate_save_btn = ctk.CTkButton(
            input_inner, text="💾  Speichern & Weiter",
            font=("Segoe UI Semibold", 14), height=48,
            fg_color=ACCENT_SUCCESS, hover_color="#00a381",
            text_color="#ffffff", corner_radius=12,
            command=self._annotate_save_and_next,
        )
        self.annotate_save_btn.pack(fill="x", pady=(0, 10))

        # Überspringen-Button
        ctk.CTkButton(
            input_inner, text="⏭️  Überspringen",
            font=("Segoe UI", 13), height=40,
            fg_color=BG_INPUT, hover_color=BG_CARD_HOVER,
            text_color=TEXT_SECONDARY, corner_radius=10,
            command=self._annotate_next,
        ).pack(fill="x", pady=(0, 10))

        # Löschen-Button
        ctk.CTkButton(
            input_inner, text="🗑️  Annotation löschen",
            font=("Segoe UI", 13), height=40,
            fg_color=BG_INPUT, hover_color="#3d1f1f",
            text_color=ACCENT_DANGER, corner_radius=10,
            command=self._annotate_delete,
        ).pack(fill="x", pady=(0, 20))

        # Navigation
        nav_frame = ctk.CTkFrame(input_inner, fg_color="transparent")
        nav_frame.pack(fill="x")

        self.annotate_prev_btn = ctk.CTkButton(
            nav_frame, text="◀  Zurück",
            font=("Segoe UI", 13), height=40, width=130,
            fg_color=BG_INPUT, hover_color=BG_CARD_HOVER,
            text_color=TEXT_SECONDARY, corner_radius=10,
            command=self._annotate_prev,
        )
        self.annotate_prev_btn.pack(side="left")

        self.annotate_next_btn = ctk.CTkButton(
            nav_frame, text="Weiter  ▶",
            font=("Segoe UI", 13), height=40, width=130,
            fg_color=BG_INPUT, hover_color=BG_CARD_HOVER,
            text_color=TEXT_SECONDARY, corner_radius=10,
            command=self._annotate_next,
        )
        self.annotate_next_btn.pack(side="right")

        # Bild laden
        self.after(200, self._annotate_show_current)

    def _annotate_show_current(self):
        """Zeigt das aktuelle Bild im Annotations-Editor."""
        if not self.image_paths:
            return

        idx = self.annotate_index
        path = self.image_paths[idx]
        fname = os.path.basename(path)
        n = len(self.image_paths)

        # UI Update
        self.annotate_filename_label.configure(text=fname)
        progress = (idx + 1) / n
        self.annotate_progress_bar.set(progress)
        self.annotate_progress_label.configure(text=f"Bild {idx + 1} / {n}")

        n_annotated = sum(1 for f in self.image_paths
                         if os.path.basename(f) in self.ground_truth)
        self.annotate_annotated_count.configure(
            text=f"✅ {n_annotated} annotiert"
        )

        # Aktueller GT
        current_gt = self.ground_truth.get(fname, None)
        if current_gt:
            self.annotate_current_gt.configure(
                text=f"Aktuelle Annotation: {current_gt}",
                text_color=ACCENT_SUCCESS,
            )
        else:
            self.annotate_current_gt.configure(
                text="Noch nicht annotiert",
                text_color=ACCENT_WARNING,
            )

        # Eingabefeld
        self.annotate_entry.delete(0, "end")
        if current_gt:
            self.annotate_entry.insert(0, current_gt)
        self.annotate_entry.focus_set()

        # Bild laden
        try:
            img = Image.open(path)
            # Verfügbare Größe abschätzen
            img = fit_image(img, 750, 550)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img,
                                    size=(img.width, img.height))
            self.annotate_image_label.configure(image=ctk_img, text="")
            self.annotate_image_label._ctk_img_ref = ctk_img
        except Exception as e:
            self.annotate_image_label.configure(
                text=f"Fehler: {e}", image=None
            )

        # Buttons
        self.annotate_prev_btn.configure(
            state="normal" if idx > 0 else "disabled"
        )

    def _annotate_save_and_next(self):
        """Speichert die Annotation und wechselt zum nächsten Bild."""
        code = self.annotate_entry.get().strip().upper()
        if code:
            fname = os.path.basename(self.image_paths[self.annotate_index])
            self.ground_truth[fname] = code
            save_ground_truth(self.ground_truth, self.gt_path)
            self._update_status_label()

        self._annotate_next()

    def _annotate_delete(self):
        """Löscht die aktuelle Annotation."""
        fname = os.path.basename(self.image_paths[self.annotate_index])
        if fname in self.ground_truth:
            del self.ground_truth[fname]
            save_ground_truth(self.ground_truth, self.gt_path)
            self._update_status_label()
        self._annotate_show_current()

    def _annotate_next(self):
        """Wechselt zum nächsten Bild."""
        if self.annotate_index < len(self.image_paths) - 1:
            self.annotate_index += 1
            self._annotate_show_current()

    def _annotate_prev(self):
        """Wechselt zum vorherigen Bild."""
        if self.annotate_index > 0:
            self.annotate_index -= 1
            self._annotate_show_current()

    # ─────────────────────────────────────────────────────────────────────────
    #  Seite 3: Benchmark
    # ─────────────────────────────────────────────────────────────────────────

    def _build_benchmark_page(self):
        """Erstellt die Benchmark-Seite."""
        frame = ctk.CTkFrame(self.main_area, fg_color=BG_DARK)
        self.frames["benchmark"] = frame

        # Header
        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.pack(fill="x", padx=30, pady=(25, 15))

        ctk.CTkLabel(
            header, text="Benchmark starten",
            font=("Segoe UI Semibold", 26), text_color=TEXT_PRIMARY,
        ).pack(anchor="w")
        ctk.CTkLabel(
            header,
            text="Alle Bilder durch die Scanner-Pipeline verarbeiten und auswerten",
            font=("Segoe UI", 13), text_color=TEXT_MUTED,
        ).pack(anchor="w", pady=(3, 0))

        # Info-Box
        info_card = ctk.CTkFrame(frame, fg_color=BG_CARD, corner_radius=16)
        info_card.pack(fill="x", padx=30, pady=10)

        info_inner = ctk.CTkFrame(info_card, fg_color="transparent")
        info_inner.pack(padx=25, pady=20)

        n = len(self.image_paths)
        n_gt = sum(1 for f in self.image_paths
                   if os.path.basename(f) in self.ground_truth)

        info_items = [
            f"📁  {n} Bilder werden evaluiert",
            f"✅  {n_gt} Ground-Truth-Einträge verfügbar",
            f"🤖  Pipeline: YOLO-Detektion → DataMatrix + OCR → Triple-Validation",
            f"📍  Quelle: {self.image_dir}",
        ]
        for item in info_items:
            ctk.CTkLabel(
                info_inner, text=item,
                font=("Segoe UI", 13), text_color=TEXT_SECONDARY,
            ).pack(anchor="w", pady=2)

        # Start-Button
        self.benchmark_start_btn = ctk.CTkButton(
            frame, text="🚀  Benchmark starten",
            font=("Segoe UI Semibold", 18), height=60,
            fg_color=ACCENT_PRIMARY, hover_color="#5a4bd6",
            text_color="#ffffff", corner_radius=14,
            command=self._start_benchmark,
        )
        self.benchmark_start_btn.pack(padx=30, pady=20, fill="x")

        # Fortschritts-Card
        progress_card = ctk.CTkFrame(frame, fg_color=BG_CARD, corner_radius=16)
        progress_card.pack(fill="x", padx=30, pady=(0, 10))

        progress_inner = ctk.CTkFrame(progress_card, fg_color="transparent")
        progress_inner.pack(fill="x", padx=25, pady=20)

        self.bench_progress_label = ctk.CTkLabel(
            progress_inner, text="Bereit zum Start",
            font=("Segoe UI Semibold", 15), text_color=TEXT_PRIMARY,
        )
        self.bench_progress_label.pack(anchor="w")

        self.bench_progress_bar = ctk.CTkProgressBar(
            progress_inner, height=18,
            fg_color=PROGRESS_BG, progress_color=ACCENT_PRIMARY,
            corner_radius=9,
        )
        self.bench_progress_bar.pack(fill="x", pady=(10, 5))
        self.bench_progress_bar.set(0)

        self.bench_progress_detail = ctk.CTkLabel(
            progress_inner, text="",
            font=("Segoe UI", 12), text_color=TEXT_MUTED,
        )
        self.bench_progress_detail.pack(anchor="w")

        # Live-Log
        self.bench_log_frame = ctk.CTkFrame(frame, fg_color=BG_CARD, corner_radius=12)
        self.bench_log_frame.pack(fill="both", expand=True, padx=30, pady=(0, 20))

        log_header = ctk.CTkFrame(self.bench_log_frame, fg_color="transparent")
        log_header.pack(fill="x", padx=15, pady=(10, 5))
        ctk.CTkLabel(
            log_header, text="Live-Protokoll",
            font=("Segoe UI Semibold", 13), text_color=TEXT_SECONDARY,
        ).pack(anchor="w")

        self.bench_log_text = ctk.CTkTextbox(
            self.bench_log_frame, fg_color=BG_INPUT,
            text_color=TEXT_PRIMARY, font=("Consolas", 11),
            corner_radius=8, state="disabled",
        )
        self.bench_log_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def _start_benchmark(self):
        """Startet den Benchmark in einem Hintergrund-Thread."""
        if self.benchmark_running:
            return

        self.benchmark_running = True
        self.benchmark_results = []
        self.benchmark_summary = {}
        self.benchmark_start_btn.configure(
            state="disabled", text="⏳  Benchmark läuft...",
            fg_color=TEXT_MUTED,
        )

        # Log leeren
        self.bench_log_text.configure(state="normal")
        self.bench_log_text.delete("1.0", "end")
        self.bench_log_text.configure(state="disabled")

        threading.Thread(target=self._run_benchmark_thread, daemon=True).start()

    def _bench_log(self, msg: str):
        """Fügt eine Zeile zum Live-Log hinzu (threadsicher)."""
        def _update():
            self.bench_log_text.configure(state="normal")
            self.bench_log_text.insert("end", msg + "\n")
            self.bench_log_text.see("end")
            self.bench_log_text.configure(state="disabled")
        self.after(0, _update)

    def _run_benchmark_thread(self):
        """Führt den Benchmark im Hintergrund aus."""
        self._bench_log("🔄 Lade YOLO-Modell...")
        self.after(0, lambda: self.bench_progress_label.configure(
            text="YOLO-Modell wird geladen..."
        ))

        try:
            model, _ = yolo_detector.load_model(APP_DIR)
        except Exception as e:
            logger.error(f"YOLO-Modell konnte nicht geladen werden: {e}")
            self._bench_log("❌ YOLO-Modell konnte nicht geladen werden!")
            self.after(0, self._benchmark_finished_error)
            return

        self._bench_log("✅ YOLO-Modell geladen.")

        def _on_progress(idx, total, results, elapsed):
            success_count = sum(1 for r in results if r["success"])
            eta = (elapsed / (idx + 1)) * (total - idx - 1)

            def _update():
                self.bench_progress_bar.set((idx + 1) / total)
                self.bench_progress_label.configure(text=f"Verarbeite Bild {idx + 1} / {total}")
                self.bench_progress_detail.configure(
                    text=f"Erfolgsrate: {success_count}/{idx + 1} • ETA: {int(eta)}s"
                )
            self.after(0, _update)

        results, summary = run_benchmark(model, self.image_paths, self.ground_truth,
                                         log=self._bench_log, on_progress=_on_progress)
        self.benchmark_results = results
        self.benchmark_summary = summary

        try:
            report_path = save_report(results, summary, self.image_dir, self.gt_path)
            self._bench_log(f"\n📄 Report gespeichert: {report_path}")
        except Exception as e:
            self._bench_log(f"\n⚠️ Report-Fehler: {e}")

        for line in format_summary_lines(summary):
            self._bench_log(line)

        def _done():
            self.bench_progress_bar.set(1.0)
            self.bench_progress_label.configure(
                text=f"✅ Benchmark abgeschlossen — {summary['success_rate_pct']:.1f}% Erfolgsrate"
            )
            self.bench_progress_detail.configure(
                text=f"{summary['total_images']} Bilder in {summary['total_time_s']:.1f}s verarbeitet"
            )
            self.benchmark_start_btn.configure(
                state="normal", text="🔄  Erneut starten",
                fg_color=ACCENT_PRIMARY,
            )
            self.benchmark_running = False
        self.after(0, _done)

    def _benchmark_finished_error(self):
        """Wird aufgerufen wenn der Benchmark fehlschlägt."""
        self.benchmark_start_btn.configure(
            state="normal", text="🚀  Benchmark starten",
            fg_color=ACCENT_PRIMARY,
        )
        self.benchmark_running = False

    # ─────────────────────────────────────────────────────────────────────────
    #  Seite 4: Ergebnisse
    # ─────────────────────────────────────────────────────────────────────────

    def _build_results_page(self):
        """Erstellt die Ergebnisseite."""
        frame = ctk.CTkFrame(self.main_area, fg_color=BG_DARK)
        self.frames["results"] = frame

        # Header
        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.pack(fill="x", padx=30, pady=(25, 15))

        ctk.CTkLabel(
            header, text="Ergebnis-Analyse",
            font=("Segoe UI Semibold", 26), text_color=TEXT_PRIMARY,
        ).pack(anchor="w")
        self.results_subtitle = ctk.CTkLabel(
            header, text="Starte zuerst einen Benchmark",
            font=("Segoe UI", 13), text_color=TEXT_MUTED,
        )
        self.results_subtitle.pack(anchor="w", pady=(3, 0))

        # Stats Cards Row
        self.results_cards_frame = ctk.CTkFrame(frame, fg_color="transparent")
        self.results_cards_frame.pack(fill="x", padx=30, pady=10)

        # Scrollable Content
        self.results_scroll = ctk.CTkScrollableFrame(
            frame, fg_color=BG_DARK,
        )
        self.results_scroll.pack(fill="both", expand=True, padx=30, pady=(0, 20))

    def _populate_results(self):
        """Füllt die Ergebnisseite mit Benchmark-Daten."""
        if not self.benchmark_summary:
            return

        s = self.benchmark_summary

        self.results_subtitle.configure(
            text=f"Letzter Run: {s['total_images']} Bilder, "
                 f"{s['success_rate_pct']}% Erfolgsrate"
        )

        # Cards leeren
        for widget in self.results_cards_frame.winfo_children():
            widget.destroy()
        for widget in self.results_scroll.winfo_children():
            widget.destroy()

        # Stat Cards
        stats = [
            ("🎯", "Erfolgsrate", f"{s['success_rate_pct']}%",
             ACCENT_SUCCESS if s['success_rate_pct'] >= 80 else ACCENT_DANGER),
            ("✅", "Genauigkeit GT", f"{s['accuracy_pct']}%",
             ACCENT_SUCCESS if s['accuracy_pct'] >= 80 else ACCENT_WARNING),
            ("⚠️", "Falsch gelesen", f"{s.get('false_read_count', '–')}",
             ACCENT_DANGER if s.get('false_read_count') else ACCENT_SUCCESS),
            ("⏱️", "Ø Dauer", f"{s['avg_duration_ms']:.0f}ms", ACCENT_INFO),
            ("📊", "Erfolg/Gesamt",
             f"{s['success_count']}/{s['total_images']}", ACCENT_PRIMARY),
        ]

        for i, (icon, label, value, color) in enumerate(stats):
            card = ctk.CTkFrame(
                self.results_cards_frame, fg_color=BG_CARD,
                corner_radius=16, height=100,
            )
            card.pack(side="left", fill="x", expand=True,
                      padx=(0 if i == 0 else 8, 0))
            card.pack_propagate(False)

            inner = ctk.CTkFrame(card, fg_color="transparent")
            inner.place(relx=0.5, rely=0.5, anchor="center")

            ctk.CTkLabel(
                inner, text=f"{icon} {label}",
                font=("Segoe UI", 12), text_color=TEXT_MUTED,
            ).pack()
            ctk.CTkLabel(
                inner, text=value,
                font=("Segoe UI Bold", 26), text_color=color,
            ).pack(pady=(3, 0))

        # Methoden-Verteilung
        if s.get("method_distribution"):
            section = ctk.CTkFrame(self.results_scroll, fg_color=BG_CARD, corner_radius=16)
            section.pack(fill="x", pady=(10, 5))

            ctk.CTkLabel(
                section, text="📊  Methoden-Verteilung",
                font=("Segoe UI Semibold", 16), text_color=TEXT_PRIMARY,
            ).pack(anchor="w", padx=20, pady=(15, 10))

            for method, count in sorted(
                s["method_distribution"].items(), key=lambda x: -x[1]
            ):
                pct = count / s["total_images"] * 100
                row = ctk.CTkFrame(section, fg_color="transparent")
                row.pack(fill="x", padx=20, pady=2)

                ctk.CTkLabel(
                    row, text=method,
                    font=("Segoe UI", 13), text_color=TEXT_SECONDARY,
                    width=200, anchor="w",
                ).pack(side="left")

                bar_bg = ctk.CTkFrame(
                    row, fg_color=PROGRESS_BG, height=16, corner_radius=8,
                )
                bar_bg.pack(side="left", fill="x", expand=True, padx=10)

                bar_fill_width = max(4, int(pct * 3))
                bar_fill = ctk.CTkFrame(
                    bar_bg, fg_color=ACCENT_PRIMARY, height=16,
                    width=bar_fill_width, corner_radius=8,
                )
                bar_fill.place(x=0, y=0, relheight=1)

                ctk.CTkLabel(
                    row, text=f"{count}× ({pct:.1f}%)",
                    font=("Segoe UI", 12), text_color=TEXT_MUTED,
                    width=100, anchor="e",
                ).pack(side="right")

            # Padding bottom
            ctk.CTkFrame(section, fg_color="transparent", height=15).pack()

        # Fehlerursachen
        if s.get("failure_categories"):
            section = ctk.CTkFrame(self.results_scroll, fg_color=BG_CARD, corner_radius=16)
            section.pack(fill="x", pady=5)

            ctk.CTkLabel(
                section, text="❌  Fehlerursachen",
                font=("Segoe UI Semibold", 16), text_color=TEXT_PRIMARY,
            ).pack(anchor="w", padx=20, pady=(15, 10))

            for reason, count in sorted(
                s["failure_categories"].items(), key=lambda x: -x[1]
            ):
                pct = count / s["total_images"] * 100
                row = ctk.CTkFrame(section, fg_color="transparent")
                row.pack(fill="x", padx=20, pady=2)

                ctk.CTkLabel(
                    row, text=reason,
                    font=("Segoe UI", 13), text_color=ACCENT_DANGER,
                    width=250, anchor="w",
                ).pack(side="left")

                ctk.CTkLabel(
                    row, text=f"{count}× ({pct:.1f}%)",
                    font=("Segoe UI", 12), text_color=TEXT_MUTED,
                ).pack(side="right")

            ctk.CTkFrame(section, fg_color="transparent", height=15).pack()

        # Einzelergebnis-Tabelle
        table_section = ctk.CTkFrame(self.results_scroll, fg_color=BG_CARD, corner_radius=16)
        table_section.pack(fill="x", pady=5)

        ctk.CTkLabel(
            table_section, text="📋  Einzelergebnisse",
            font=("Segoe UI Semibold", 16), text_color=TEXT_PRIMARY,
        ).pack(anchor="w", padx=20, pady=(15, 10))

        # Tabellen-Header
        header_row = ctk.CTkFrame(table_section, fg_color=BG_INPUT, corner_radius=8)
        header_row.pack(fill="x", padx=15, pady=(0, 5))

        cols = [
            ("#", 40), ("Datei", 120), ("Status", 60), ("Ergebnis", 80),
            ("Soll", 60), ("Match", 50), ("Methode", 130), ("Dauer", 60),
        ]
        for col_name, w in cols:
            ctk.CTkLabel(
                header_row, text=col_name, width=w,
                font=("Segoe UI Semibold", 11), text_color=TEXT_MUTED,
                anchor="w",
            ).pack(side="left", padx=5, pady=5)

        # Zeilen
        for r in self.benchmark_results:
            row_frame = ctk.CTkFrame(table_section, fg_color="transparent", height=30)
            row_frame.pack(fill="x", padx=15, pady=1)

            status = "✅" if r["success"] else "❌"
            match_str = "✓" if r["is_match"] else ("✗" if r["expected"] else "—")
            match_color = ACCENT_SUCCESS if r["is_match"] else (
                ACCENT_DANGER if r["expected"] else TEXT_MUTED
            )

            values = [
                (str(r["index"]), 40, TEXT_MUTED),
                (r["filename"], 120, TEXT_PRIMARY),
                (status, 60, None),
                (r["result_code"] if r["success"] else "FAIL", 80,
                 ACCENT_SUCCESS if r["success"] else ACCENT_DANGER),
                (r["expected"] or "—", 60, TEXT_MUTED),
                (match_str, 50, match_color),
                (r["method"], 130, TEXT_SECONDARY),
                (f"{r['duration_ms']}ms", 60, TEXT_MUTED),
            ]

            for val, w, color in values:
                ctk.CTkLabel(
                    row_frame, text=val, width=w,
                    font=("Segoe UI", 11),
                    text_color=color or TEXT_PRIMARY,
                    anchor="w",
                ).pack(side="left", padx=5)

        ctk.CTkFrame(table_section, fg_color="transparent", height=15).pack()

    # ─────────────────────────────────────────────────────────────────────────
    #  Seite 5: Detail-Ansicht
    # ─────────────────────────────────────────────────────────────────────────

    def _build_detail_page(self):
        """Erstellt die Detail-Ansicht."""
        frame = ctk.CTkFrame(self.main_area, fg_color=BG_DARK)
        self.frames["detail"] = frame

        # Header
        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.pack(fill="x", padx=30, pady=(25, 10))

        ctk.CTkLabel(
            header, text="Pipeline Detail-Ansicht",
            font=("Segoe UI Semibold", 26), text_color=TEXT_PRIMARY,
        ).pack(anchor="w")
        ctk.CTkLabel(
            header, text="Visualisiere wie die KI das Bild verarbeitet hat",
            font=("Segoe UI", 13), text_color=TEXT_MUTED,
        ).pack(anchor="w", pady=(3, 0))

        # Bildauswahl-Leiste
        select_frame = ctk.CTkFrame(frame, fg_color=BG_CARD, corner_radius=12, height=50)
        select_frame.pack(fill="x", padx=30, pady=(0, 10))
        select_frame.pack_propagate(False)

        select_inner = ctk.CTkFrame(select_frame, fg_color="transparent")
        select_inner.pack(fill="x", padx=15, pady=8)

        ctk.CTkLabel(
            select_inner, text="Bild auswählen:",
            font=("Segoe UI Semibold", 13), text_color=TEXT_SECONDARY,
        ).pack(side="left", padx=(0, 10))

        self.detail_combo_var = tk.StringVar(value="—")
        self.detail_combo = ctk.CTkComboBox(
            select_inner, variable=self.detail_combo_var,
            values=["Starte zuerst einen Benchmark"],
            font=("Segoe UI", 12), width=250,
            fg_color=BG_INPUT, border_color=BORDER_COLOR,
            button_color=ACCENT_PRIMARY, dropdown_fg_color=BG_CARD,
            dropdown_text_color=TEXT_PRIMARY,
            command=self._on_detail_select,
        )
        self.detail_combo.pack(side="left", padx=5)

        # Filter
        self.detail_filter_var = tk.StringVar(value="Alle")
        filter_options = ["Alle", "Nur Fehler", "Nur Erfolge"]
        for opt in filter_options:
            ctk.CTkRadioButton(
                select_inner, text=opt,
                variable=self.detail_filter_var, value=opt,
                font=("Segoe UI", 12), text_color=TEXT_SECONDARY,
                fg_color=ACCENT_PRIMARY, hover_color=ACCENT_SECONDARY,
                command=self._populate_detail_list,
            ).pack(side="left", padx=10)

        # Navigation
        self.detail_prev_btn = ctk.CTkButton(
            select_inner, text="◀", width=36, height=36,
            fg_color=BG_INPUT, hover_color=BG_CARD_HOVER,
            text_color=TEXT_SECONDARY, corner_radius=8,
            command=self._detail_prev,
        )
        self.detail_prev_btn.pack(side="right", padx=2)

        self.detail_next_btn = ctk.CTkButton(
            select_inner, text="▶", width=36, height=36,
            fg_color=BG_INPUT, hover_color=BG_CARD_HOVER,
            text_color=TEXT_SECONDARY, corner_radius=8,
            command=self._detail_next,
        )
        self.detail_next_btn.pack(side="right", padx=2)

        # Haupt-Content (scrollbar)
        self.detail_scroll = ctk.CTkScrollableFrame(
            frame, fg_color=BG_DARK,
        )
        self.detail_scroll.pack(fill="both", expand=True, padx=30, pady=(0, 20))

        self.detail_current_idx = 0
        self.detail_filtered_results = []

    def _populate_detail_list(self):
        """Füllt die Auswahlliste."""
        if not self.benchmark_results:
            return

        filter_val = self.detail_filter_var.get()
        filtered = []
        for r in self.benchmark_results:
            if filter_val == "Nur Fehler" and r["success"]:
                continue
            if filter_val == "Nur Erfolge" and not r["success"]:
                continue
            filtered.append(r)

        self.detail_filtered_results = filtered

        labels = []
        for r in filtered:
            status = "✅" if r["success"] else "❌"
            labels.append(f"{status} {r['filename']} → {r['result_code'] or 'FAIL'}")

        if labels:
            self.detail_combo.configure(values=labels)
            self.detail_combo_var.set(labels[0])
            self.detail_current_idx = 0
            self._show_detail(0)
        else:
            self.detail_combo.configure(values=["Keine Ergebnisse"])
            self.detail_combo_var.set("Keine Ergebnisse")

    def _on_detail_select(self, selection):
        """Callback wenn ein Bild in der Combo ausgewählt wird."""
        labels = self.detail_combo.cget("values")
        if selection in labels:
            idx = labels.index(selection)
            self.detail_current_idx = idx
            self._show_detail(idx)

    def _detail_prev(self):
        if self.detail_current_idx > 0:
            self.detail_current_idx -= 1
            labels = self.detail_combo.cget("values")
            if self.detail_current_idx < len(labels):
                self.detail_combo_var.set(labels[self.detail_current_idx])
            self._show_detail(self.detail_current_idx)

    def _detail_next(self):
        if self.detail_current_idx < len(self.detail_filtered_results) - 1:
            self.detail_current_idx += 1
            labels = self.detail_combo.cget("values")
            if self.detail_current_idx < len(labels):
                self.detail_combo_var.set(labels[self.detail_current_idx])
            self._show_detail(self.detail_current_idx)

    def _show_detail(self, idx: int):
        """Zeigt die Detail-Ansicht für ein bestimmtes Bild."""
        if idx >= len(self.detail_filtered_results):
            return

        result = self.detail_filtered_results[idx]

        # Content leeren
        for widget in self.detail_scroll.winfo_children():
            widget.destroy()

        path = result["path"]
        fname = result["filename"]

        # ===== Ergebnis-Card =====
        result_card = ctk.CTkFrame(
            self.detail_scroll, fg_color=BG_CARD, corner_radius=16,
        )
        result_card.pack(fill="x", pady=(0, 10))

        result_inner = ctk.CTkFrame(result_card, fg_color="transparent")
        result_inner.pack(fill="x", padx=25, pady=20)

        # Titel-Zeile
        title_row = ctk.CTkFrame(result_inner, fg_color="transparent")
        title_row.pack(fill="x", pady=(0, 15))

        status_emoji = "✅" if result["success"] else "❌"
        status_text = "ERFOLGREICH" if result["success"] else "FEHLGESCHLAGEN"
        status_color = ACCENT_SUCCESS if result["success"] else ACCENT_DANGER

        ctk.CTkLabel(
            title_row, text=f"{status_emoji}  {fname}",
            font=("Segoe UI Semibold", 20), text_color=TEXT_PRIMARY,
        ).pack(side="left")

        ctk.CTkLabel(
            title_row, text=status_text,
            font=("Segoe UI Bold", 14), text_color=status_color,
        ).pack(side="right")

        # Info Grid
        info_items = [
            ("Erkannter Code", result["result_code"] or "—",
             ACCENT_SUCCESS if result["success"] else ACCENT_DANGER),
            ("Erwarteter Code (GT)", result["expected"] or "Nicht annotiert",
             TEXT_PRIMARY if result["expected"] else TEXT_MUTED),
            ("Methode", result["method"], ACCENT_INFO),
            ("Konfidenz", f"{result['confidence']:.1%}", TEXT_PRIMARY),
            ("Dauer", f"{result['duration_ms']} ms", TEXT_PRIMARY),
            ("DataMatrix (DMTX)", result["dmtx_result"] or "Nicht erkannt", TEXT_SECONDARY),
            ("OCR (Klarschrift)", result["ocr_result"] or ("Übersprungen (DMX dekodiert)" if result["dmtx_result"] else "Nicht erkannt"), TEXT_SECONDARY),
            ("OCR Partial", result["ocr_partial"] or "—", TEXT_MUTED),
        ]

        if result["fail_reason"]:
            info_items.append(
                ("Fehlerursache", result["fail_reason"], ACCENT_DANGER)
            )

        for label, value, color in info_items:
            row = ctk.CTkFrame(result_inner, fg_color="transparent")
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(
                row, text=f"{label}:", width=200, anchor="w",
                font=("Segoe UI", 13), text_color=TEXT_MUTED,
            ).pack(side="left")
            ctk.CTkLabel(
                row, text=str(value), anchor="w",
                font=("Segoe UI Semibold", 13), text_color=color,
            ).pack(side="left", padx=10)

        # ===== Visualisierungs-Cards =====
        try:
            image = cv2.imread(path)
            if image is None:
                raise ValueError("Bild konnte nicht geladen werden")

            # Bilder-Grid
            images_frame = ctk.CTkFrame(
                self.detail_scroll, fg_color="transparent",
            )
            images_frame.pack(fill="x", pady=5)

            # 1. Originalbild mit YOLO-Boxen
            box_card = ctk.CTkFrame(images_frame, fg_color=BG_CARD, corner_radius=14)
            box_card.pack(side="left", fill="both", expand=True, padx=(0, 5))

            ctk.CTkLabel(
                box_card, text="🔲  YOLO-Detektionen",
                font=("Segoe UI Semibold", 14), text_color=TEXT_PRIMARY,
            ).pack(anchor="w", padx=15, pady=(12, 5))

            detections = result.get("detections", [])
            n_dets = len(detections)
            det_info = f"{n_dets} Detektion{'en' if n_dets != 1 else ''} gefunden"
            if detections:
                det_details = ", ".join(
                    f"{'DMX' if d['cls'] == 0 else 'TXT'} ({d['conf']:.0%})"
                    for d in detections
                )
                det_info += f": {det_details}"
            ctk.CTkLabel(
                box_card, text=det_info,
                font=("Segoe UI", 11), text_color=TEXT_MUTED,
            ).pack(anchor="w", padx=15)

            annotated_img = draw_boxes_on_image(image, detections)
            pil_annotated = cv2_to_pil(annotated_img)
            pil_annotated = fit_image(pil_annotated, 500, 380)
            ctk_annotated = ctk.CTkImage(light_image=pil_annotated, dark_image=pil_annotated,
                                          size=(pil_annotated.width, pil_annotated.height))

            img_lbl_1 = ctk.CTkLabel(box_card, image=ctk_annotated, text="")
            img_lbl_1._ctk_img_ref = ctk_annotated
            img_lbl_1.pack(padx=10, pady=(5, 12))

            # 2. Crops (Eingaben für 2-Klassen Pipeline: DMX & Text)
            crop_card = ctk.CTkFrame(images_frame, fg_color=BG_CARD, corner_radius=14)
            crop_card.pack(side="right", fill="both", expand=True, padx=(5, 0))

            ctk.CTkLabel(
                crop_card, text="✂️  Scanner-Crops (Eingaben für 2-Klassen Pipeline)",
                font=("Segoe UI Semibold", 14), text_color=TEXT_PRIMARY,
            ).pack(anchor="w", padx=15, pady=(12, 5))

            # Dieselbe Auswahl (inkl. abgeleiteter Boxen) wie in scanner.scan_2class()
            best_dmx, best_txt = scanner.select_label_detections(detections)

            crop_container = ctk.CTkFrame(crop_card, fg_color="transparent")
            crop_container.pack(fill="both", expand=True, padx=10, pady=5)

            if best_dmx or best_txt:
                if best_dmx:
                    dmx_sub = ctk.CTkFrame(crop_container, fg_color=BG_INPUT, corner_radius=8)
                    dmx_sub.pack(side="left", fill="both", expand=True, padx=4, pady=4)
                    c_str = f"{best_dmx['conf']:.0%}" if "conf" in best_dmx else "0%"
                    lbl_txt_dmx = f"DataMatrix Crop ({'Abgeleitet' if best_dmx.get('derived') else c_str})"
                    ctk.CTkLabel(dmx_sub, text=lbl_txt_dmx, font=("Segoe UI Semibold", 11), text_color=ACCENT_SUCCESS).pack(pady=(4, 2))
                    dmx_crop = scanner.deskew_crop(image, best_dmx["box"], padding=40)
                    pil_dmx = fit_image(cv2_to_pil(dmx_crop), 230, 320)
                    ctk_dmx = ctk.CTkImage(light_image=pil_dmx, dark_image=pil_dmx, size=(pil_dmx.width, pil_dmx.height))
                    lbl_d = ctk.CTkLabel(dmx_sub, image=ctk_dmx, text="")
                    lbl_d._ctk_img_ref = ctk_dmx
                    lbl_d.pack(pady=4)

                if best_txt:
                    txt_sub = ctk.CTkFrame(crop_container, fg_color=BG_INPUT, corner_radius=8)
                    txt_sub.pack(side="right", fill="both", expand=True, padx=4, pady=4)
                    c_str = f"{best_txt['conf']:.0%}" if "conf" in best_txt else "0%"
                    lbl_txt_ocr = f"Klarschrift Crop ({'Abgeleitet' if best_txt.get('derived') else c_str})"
                    ctk.CTkLabel(txt_sub, text=lbl_txt_ocr, font=("Segoe UI Semibold", 11), text_color=ACCENT_INFO).pack(pady=(4, 2))
                    txt_crop = scanner.deskew_crop(image, best_txt["box"], padding=30)
                    pil_txt = fit_image(cv2_to_pil(txt_crop), 230, 320)
                    ctk_txt = ctk.CTkImage(light_image=pil_txt, dark_image=pil_txt, size=(pil_txt.width, pil_txt.height))
                    lbl_t = ctk.CTkLabel(txt_sub, image=ctk_txt, text="")
                    lbl_t._ctk_img_ref = ctk_txt
                    lbl_t.pack(pady=4)
            else:
                ctk.CTkLabel(
                    crop_card, text="Keine YOLO Detektionen über Schwelle — Fallback Vollbild scan()",
                    font=("Segoe UI", 12), text_color=TEXT_MUTED,
                ).pack(padx=15, pady=50)

            # 3. Pipeline-Schritte Visualisierung
            pipeline_card = ctk.CTkFrame(
                self.detail_scroll, fg_color=BG_CARD, corner_radius=14,
            )
            pipeline_card.pack(fill="x", pady=5)

            ctk.CTkLabel(
                pipeline_card, text="🔄  Pipeline-Ablauf",
                font=("Segoe UI Semibold", 14), text_color=TEXT_PRIMARY,
            ).pack(anchor="w", padx=15, pady=(12, 10))

            # Pipeline Steps
            steps = self._build_pipeline_steps(result)
            for step_icon, step_name, step_result, step_color in steps:
                step_row = ctk.CTkFrame(pipeline_card, fg_color=BG_INPUT, corner_radius=10)
                step_row.pack(fill="x", padx=15, pady=3)

                ctk.CTkLabel(
                    step_row, text=f"  {step_icon}  {step_name}",
                    font=("Segoe UI", 13), text_color=TEXT_SECONDARY,
                    anchor="w",
                ).pack(side="left", padx=10, pady=8)

                ctk.CTkLabel(
                    step_row, text=step_result,
                    font=("Segoe UI Semibold", 13), text_color=step_color,
                    anchor="e",
                ).pack(side="right", padx=15, pady=8)

            ctk.CTkFrame(pipeline_card, fg_color="transparent", height=12).pack()

        except Exception as e:
            ctk.CTkLabel(
                self.detail_scroll, text=f"Fehler: {e}",
                font=("Segoe UI", 13), text_color=ACCENT_DANGER,
            ).pack(padx=20, pady=20)

    def _build_pipeline_steps(self, result: dict) -> list[tuple]:
        """Erstellt die Pipeline-Schritt-Darstellung."""
        steps = []

        # Step 1: YOLO
        n_dets = len(result.get("detections", []))
        if n_dets > 0:
            steps.append((
                "🔲", "YOLO-Detektion (Etikett finden)",
                f"{n_dets} Box{'en' if n_dets > 1 else ''} gefunden",
                ACCENT_SUCCESS,
            ))
        else:
            steps.append((
                "🔲", "YOLO-Detektion",
                "Keine Box erkannt", ACCENT_DANGER,
            ))

        # Step 2: DataMatrix
        dmtx = result.get("dmtx_result")
        if dmtx:
            steps.append((
                "📊", "DataMatrix-Dekodierung (zxing/pylibdmtx)",
                f"Erkannt: {dmtx}", ACCENT_SUCCESS,
            ))
        else:
            steps.append((
                "📊", "DataMatrix-Dekodierung",
                "Nicht dekodierbar", ACCENT_DANGER,
            ))

        # Step 3: OCR
        ocr = result.get("ocr_result")
        partial = result.get("ocr_partial")
        if ocr:
            steps.append((
                "🔤", "OCR / Klarschrift-Erkennung",
                f"Erkannt: {ocr}", ACCENT_SUCCESS,
            ))
        elif partial:
            steps.append((
                "🔤", "OCR / Klarschrift-Erkennung",
                f"Teilweise: {partial}", ACCENT_WARNING,
            ))
        elif dmtx:
            steps.append((
                "🔤", "OCR / Klarschrift-Erkennung",
                "Übersprungen (DataMatrix dekodiert)", TEXT_MUTED,
            ))
        else:
            steps.append((
                "🔤", "OCR / Klarschrift-Erkennung",
                "Nicht erkannt", ACCENT_DANGER,
            ))

        # Step 4: Fusion
        method = result.get("method", "")
        if result["success"]:
            steps.append((
                "🔗", "Triple-Validation / Fusion",
                f"→ {result['result_code']} via {method}",
                ACCENT_SUCCESS,
            ))
        else:
            steps.append((
                "🔗", "Triple-Validation / Fusion",
                f"Gescheitert ({result.get('fail_reason', '?')})",
                ACCENT_DANGER,
            ))

        # Step 5: Match
        if result["expected"]:
            if result["is_match"]:
                steps.append((
                    "✅", "Ground-Truth Vergleich",
                    f"{result['result_code']} = {result['expected']} ✓",
                    ACCENT_SUCCESS,
                ))
            else:
                steps.append((
                    "❌", "Ground-Truth Vergleich",
                    f"{result['result_code'] or 'FAIL'} ≠ {result['expected']}",
                    ACCENT_DANGER,
                ))

        return steps


# ═══════════════════════════════════════════════════════════════════════════════
#  Start
# ═══════════════════════════════════════════════════════════════════════════════

def run_headless(image_dir: str, gt_path: str):
    """Benchmark ohne GUI; schreibt denselben Bericht wie die GUI."""
    model, _ = yolo_detector.load_model(APP_DIR)
    results, summary = run_benchmark(model, load_image_list(image_dir), load_ground_truth(gt_path))
    print(f"\n📄 Report gespeichert: {save_report(results, summary, image_dir, gt_path)}")
    for line in format_summary_lines(summary):
        print(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DataDetector Benchmark & Ground-Truth-Tool")
    parser.add_argument("--images", default=IMAGE_DIR, help="Bildordner (Standard: training_data)")
    parser.add_argument("--gt", default=GROUND_TRUTH_PATH, help="Ground-Truth-Datei (Standard: ground_truth.json)")
    parser.add_argument("--headless", action="store_true", help="Benchmark ohne GUI ausführen und Bericht speichern")
    args = parser.parse_args()
    images_dir = os.path.abspath(args.images)
    gt_file = os.path.abspath(args.gt)
    if args.headless:
        run_headless(images_dir, gt_file)
    else:
        BenchmarkApp(images_dir, gt_file).mainloop()
