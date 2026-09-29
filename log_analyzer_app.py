"""
log_analyzer_app.py — Analyse- und Diagnose-App für die Scan-Logs des DataMatrixReaders.

Alle Dateizugriffe laufen in einem Hintergrund-Thread; beim Live-Update werden nur neu angehängte Zeilen
gelesen. Die Oberfläche arbeitet auf kompakten ScanRecord-Objekten und bekommt Kennzahlen fertig berechnet,
vollständige Datensätze und Bilder werden erst bei Auswahl nachgeladen.
"""

import bisect
import csv
import itertools
import json
import os
import queue
import re
import shutil
import sys
import threading
import time
import tkinter.font as tkfont
import traceback
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from tkinter import filedialog, messagebox, ttk

import customtkinter as ctk
import matplotlib
from PIL import Image, ImageDraw, ImageFont, ImageTk

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Wedge  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

# --- UI Styling (Light Theme) ---
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

ACCENT   = "#2563EB"  # Blue
SUCCESS  = "#16A34A"  # Green
WARN     = "#D97706"  # Orange
DANGER   = "#DC2626"  # Red
BG_SIDE  = "#F1F5F9"  # Light gray sidebar
BG_CARD  = "#FFFFFF"  # White card background
BG_MAIN  = "#E2E8F0"  # Gray main background
BORDER   = "#CBD5E1"  # Border gray
TXT_DARK = "#1E293B"  # Text dark slate
TXT_MID  = "#475569"  # Text medium gray
TXT_LIGHT= "#94A3B8"  # Text light gray
HOVER_ROW = "#F1F5F9"  # Subtle row hover (Fluent Design)
SELECT_ROW= "#DBEAFE"  # Selected row highlight
HOVER_CARD= "#F8FAFC"  # KPI card hover

GRADE_COLORS = {"A": SUCCESS, "B": ACCENT, "C": WARN, "D": DANGER}
METHOD_COLORS = {"Verifiziert": SUCCESS, "Modulabgleich": "#0891B2", "OCR": ACCENT, "Rekonstruiert": WARN,
                 "Fehler": DANGER}

APP_VERSION = "5.0"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(APP_DIR, "config.json")
DEFAULT_LOG_DIR = r"U:\Temp\DataMatrixReader.logFiles"
ALL_SOURCES = "__ALL__"
TIME_RANGES = ("Heute", "7 Tage", "30 Tage", "Alle")
GRADE_FILTERS = ("Alle Grades", "Grade A", "Grade B", "Grade C", "Grade D")
STATUS_FILTERS = ("Alle Status", "Erfolg", "Fehlgeschlagen", "Verifiziert", "Modulabgleich", "OCR", "Rekonstruiert",
                  "Fehler")
POLL_MS = 2000
PAGE_SIZE = 500
SLOW_SCAN_MS = 12000        # wie das Flag SLOW_SCAN des ScanLoggers
CHART_INTERVAL_S = 2.0      # Diagramme höchstens alle 2 s neu zeichnen
DISK_INTERVAL_S = 30.0
DETAIL_IMAGE_SIZE = (340, 190)

# Tabellenspalten: (id, Überschrift, Breite, dehnbar)
COLUMNS = (
    ("time", "Zeit", 118, False),
    ("camera", "Kamera", 90, False),
    ("code", "Ergebnis", 74, False),
    ("method", "Methode", 100, False),
    ("conf", "Konf.", 58, False),
    ("duration", "Dauer", 80, False),
    ("flags", "Flags", 200, True),
    ("scan_id", "Scan-ID", 190, False),
)
SORT_KEYS = {
    "#0": lambda r: r.grade,
    "time": lambda r: r.epoch,
    "camera": lambda r: r.source,
    "code": lambda r: r.code,
    "method": lambda r: r.method,
    "conf": lambda r: r.confidence,
    "duration": lambda r: r.total_ms,
    "flags": lambda r: len(r.flags),
    "scan_id": lambda r: r.scan_id,
}
DESCENDING_FIRST = {"time", "conf", "duration", "flags"}
DETAIL_FIELDS = ("Scan-ID", "Zeitstempel", "Kamera", "Endergebnis", "Methode", "Konfidenz", "DMTX", "OCR",
                 "YOLO Conf", "Dauer", "Trigger", "Kamera-Modell", "Belichtung/Gain", "Version", "Flags")


def _load_config() -> dict:
    try:
        with open(CONFIG_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _natural_key(text: str) -> list:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _fmt_int(value: float) -> str:
    return f"{int(value):,}".replace(",", ".")


def _fmt_pct(value: float) -> str:
    return f"{value:.1f}".replace(".", ",") + " %"


def _fmt_ago(seconds: float) -> str:
    if seconds < 60:
        return f"vor {max(0, int(seconds))} s"
    if seconds < 3600:
        return f"vor {int(seconds // 60)} min"
    if seconds < 86400:
        return f"vor {int(seconds // 3600)} h"
    return f"vor {int(seconds // 86400)} Tagen"


def _range_start(range_key: str, now: float) -> float | None:
    if range_key == "Heute":
        return datetime.fromtimestamp(now).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    if range_key == "7 Tage":
        return now - 7 * 86400
    if range_key == "30 Tage":
        return now - 30 * 86400
    return None


def _disk_usage(path: str) -> tuple[float, float] | None:
    try:
        usage = shutil.disk_usage(path)
    except OSError:
        return None
    return usage.used / usage.total * 100, usage.free / 2**30


# --------------------------------------------------------------------------- #
#  Daten                                                                       #
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class ScanRecord:
    """Kompakter Tabellen-Datensatz; der vollständige JSON-Datensatz steht ab `offset` in `path`."""
    key: str
    source: str        # Kamera-Ordner im Log-Verzeichnis ("" = Hauptverzeichnis)
    path: str
    offset: int
    epoch: float
    scan_id: str
    code: str
    success: bool
    method: str
    confidence: float
    grade: str
    flags: tuple
    total_ms: int
    has_image: bool
    search: str


@dataclass
class Trend:
    labels: list
    ok: list
    fail: list
    unit: str


@dataclass
class Stats:
    total: int = 0
    success: int = 0
    avg_ms: float = 0.0
    median_ms: int = 0
    p95_ms: int = 0
    images: int = 0
    grades: Counter = field(default_factory=Counter)
    methods: Counter = field(default_factory=Counter)
    flags: Counter = field(default_factory=Counter)
    trend: Trend | None = None
    last: ScanRecord | None = None


@dataclass
class View:
    """Ausschnitt für Kamera + Zeitraum mit fertig berechneten Kennzahlen (wird im Hintergrund erstellt)."""
    params: tuple
    records: list      # neueste zuerst
    stats: Stats
    sources: list
    labels: dict
    source_stats: dict  # Quelle → (Scans, Erfolge) im Zeitraum
    total_loaded: int
    log_dir: str
    load_s: float


def _rewrite_jsonl(path: str, lines: list[bytes], read_until: int):
    """Schreibt die Datei neu; Zeilen, die der Reader währenddessen anhängt, bleiben erhalten."""
    tmp = path + ".tmp"
    for _attempt in range(20):
        with open(path, "rb") as f:
            f.seek(read_until)
            tail = f.read()
        with open(tmp, "wb") as f:
            f.writelines(lines)
            f.write(tail)
        try:
            if os.path.getsize(path) == read_until + len(tail):
                os.replace(tmp, path)
                return
        except PermissionError:
            pass  # Reader schreibt gerade
        time.sleep(0.05)
    os.remove(tmp)
    raise OSError(f"{path} ist dauerhaft gesperrt")


class LogStore:
    """JSONL-Logs eines Log-Verzeichnisses (ein Unterordner pro Kamera); refresh() liest nur neue Zeilen."""

    def __init__(self, log_dir: str, camera_names: dict[str, str]):
        self.log_dir = log_dir
        self.records: list[ScanRecord] = []   # neueste zuerst; wird bei Änderungen ersetzt, nie verändert
        self.sources: list[str] = []
        self._camera_names = camera_names
        self._read_until: dict[str, int] = {}
        self._by_file: dict[str, list[ScanRecord]] = {}
        self._keys = itertools.count()
        self._dirty = False

    def label(self, source: str) -> str:
        if source == ALL_SOURCES:
            return "Alle Kameras"
        if source == "":
            return "Hauptverzeichnis" if len(self.sources) > 1 else "Standard-Kamera"
        name = self._camera_names.get(source.lower())
        if name:
            return name
        match = re.match(r"^(?:cam|camera)[_\s]*(\d+)$", source, re.IGNORECASE)
        return f"Kamera {match.group(1)}" if match else source.capitalize()

    def _discover(self) -> dict[str, str]:
        """JSONL-Dateien → Quelle; durchsucht nur Hauptordner und Kameraordner, nie die Bildordner."""
        files: dict[str, str] = {}
        sources = set()
        try:
            entries = list(os.scandir(self.log_dir))
        except OSError:
            entries = []
        for entry in entries:
            try:
                if entry.is_file():
                    if entry.name.endswith(".jsonl"):
                        files[entry.path] = ""
                        sources.add("")
                elif entry.is_dir() and entry.name.lower() != "images":
                    found = False
                    for sub in os.scandir(entry.path):
                        if sub.name.endswith(".jsonl") and sub.is_file():
                            files[sub.path] = entry.name
                            found = True
                    if found or entry.name.lower().startswith("cam"):
                        sources.add(entry.name)
            except OSError:
                continue
        self.sources = sorted(sources, key=lambda s: (s == "", _natural_key(s)))
        return files

    def refresh(self, progress=None) -> bool:
        """Liest neue und geänderte Dateien ein → True, wenn sich Datensätze oder Quellen geändert haben."""
        old_sources = self.sources
        files = self._discover()
        changed = self._dirty or self.sources != old_sources
        self._dirty = False
        for path in list(self._read_until):
            if path not in files:
                del self._read_until[path]
                self._by_file.pop(path, None)
                changed = True

        work = []
        for path, source in files.items():
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            done = self._read_until.get(path)
            if done is None or size < done:  # neu oder gekürzt (z.B. nach dem Löschen) → komplett lesen
                changed = changed or done is not None
                self._by_file[path] = []
                done = 0
            if size > done:
                work.append((path, source, done, size))

        total_bytes = sum(size - start for _, _, start, size in work) or 1
        read_bytes = 0
        for path, source, start, size in work:
            def report(done_in_file, base=read_bytes):
                if progress is not None:
                    progress(min(1.0, (base + done_in_file) / total_bytes))
            new_records = self._read(path, source, start, report)
            read_bytes += size - start
            if new_records:
                self._by_file[path].extend(new_records)
                changed = True

        if changed:
            merged = [record for records in self._by_file.values() for record in records]
            merged.sort(key=lambda r: r.epoch, reverse=True)
            self.records = merged
        return changed

    def _read(self, path: str, source: str, start: int, report) -> list[ScanRecord]:
        records = []
        offset = start
        label = self.label(source)
        next_report = start + 4 * 2**20
        try:
            with open(path, "rb") as f:
                f.seek(start)
                for raw in f:
                    if not raw.endswith(b"\n"):
                        break  # Zeile wird gerade geschrieben → beim nächsten Durchlauf lesen
                    record = self._parse(raw, source, label, path, offset)
                    if record is not None:
                        records.append(record)
                    offset += len(raw)
                    if offset >= next_report:
                        report(offset - start)
                        next_report = offset + 4 * 2**20
        except OSError:
            pass
        self._read_until[path] = offset
        return records

    def _parse(self, raw: bytes, source: str, label: str, path: str, offset: int) -> ScanRecord | None:
        try:
            data = json.loads(raw)
        except ValueError:
            return None
        if not isinstance(data, dict) or data.get("type") == "SESSION_END":
            return None
        result = data.get("result") or {}
        quality = data.get("quality") or {}
        timing = data.get("timing") or {}
        try:
            epoch = datetime.fromisoformat(data.get("ts") or "").timestamp()
        except (TypeError, ValueError):
            epoch = 0.0
        try:
            total_ms = int(timing.get("total_ms") or data.get("duration_ms") or 0)
            confidence = float(result.get("confidence") or 0.0)
        except (TypeError, ValueError):
            total_ms, confidence = 0, 0.0
        code = str(result.get("code") or "")
        scan_id = str(data.get("scan_id") or "")
        method = sys.intern(str(result.get("method") or "Fehler"))
        flags = tuple(sys.intern(str(flag)) for flag in quality.get("flags") or ())
        return ScanRecord(
            key=str(next(self._keys)), source=source, path=path, offset=offset, epoch=epoch,
            scan_id=scan_id, code=code, success=bool(result.get("success")), method=method,
            confidence=confidence, grade=sys.intern(str(quality.get("grade") or "D")), flags=flags,
            total_ms=total_ms, has_image=bool((data.get("image") or {}).get("path")),
            search=f"{code} {scan_id} {method} {label} {' '.join(flags)}".lower(),
        )

    @staticmethod
    def read_full(record: ScanRecord) -> dict | None:
        """Vollständiger JSON-Datensatz (None, wenn die Datei inzwischen umgeschrieben wurde)."""
        try:
            with open(record.path, "rb") as f:
                f.seek(record.offset)
                data = json.loads(f.readline())
        except (OSError, ValueError):
            return None
        if isinstance(data, dict) and str(data.get("scan_id") or "") == record.scan_id:
            return data
        return None

    def load_full_many(self, records: list[ScanRecord]) -> dict[tuple[str, int], dict]:
        """Vollständige Datensätze vieler Records, pro Datei in einem Durchlauf gelesen."""
        wanted: dict[str, set[int]] = {}
        for record in records:
            wanted.setdefault(record.path, set()).add(record.offset)
        result = {}
        for path, offsets in wanted.items():
            try:
                with open(path, "rb") as f:
                    offset = 0
                    for raw in f:
                        if offset in offsets:
                            try:
                                result[(path, offset)] = json.loads(raw)
                            except ValueError:
                                pass
                        offset += len(raw)
            except OSError:
                continue
        return result

    def delete(self, records: list[ScanRecord]) -> tuple[int, int]:
        """Entfernt Datensätze samt gespeicherten Bildern aus den JSONL-Dateien → (Datensätze, Bilder)."""
        targets: dict[str, dict[int, str]] = {}
        for record in records:
            targets.setdefault(record.path, {})[record.offset] = record.scan_id
        removed = images = 0
        for path, by_offset in targets.items():
            keep = []
            offset = 0
            with open(path, "rb") as f:
                for raw in f:
                    scan_id = by_offset.get(offset)
                    offset += len(raw)
                    if scan_id is not None and raw.endswith(b"\n"):
                        try:
                            data = json.loads(raw)
                        except ValueError:
                            data = None
                        if isinstance(data, dict) and str(data.get("scan_id") or "") == scan_id:
                            removed += 1
                            images += self._delete_image(path, data)
                            continue
                    keep.append(raw)
            _rewrite_jsonl(path, keep, offset)
            self._read_until.pop(path, None)  # beim nächsten refresh() komplett neu einlesen
            self._by_file.pop(path, None)
            self._dirty = True
        return removed, images

    def _delete_image(self, jsonl_path: str, data: dict) -> int:
        relative = (data.get("image") or {}).get("path")
        if not relative:
            return 0
        for candidate in (os.path.join(os.path.dirname(jsonl_path), relative), os.path.join(self.log_dir, relative)):
            if os.path.isfile(candidate):
                try:
                    os.remove(candidate)
                    return 1
                except OSError:
                    return 0
        return 0


def _trend(records: list[ScanRecord], range_key: str, start: float | None, now: float) -> Trend | None:
    """Erfolge/Fehler pro Stunde, Tag oder Woche (je nach Zeitspanne), lückenlos bis jetzt."""
    oldest = next((r.epoch for r in reversed(records) if r.epoch > 0), None)
    if oldest is None:
        return None
    first = start if start is not None else oldest
    span = now - first
    if range_key == "Heute" or span <= 2 * 86400:
        step, fmt, unit = 3600, "%H:00", "pro Stunde"
    elif span <= 120 * 86400:
        step, fmt, unit = 86400, "%d.%m.", "pro Tag"
    else:
        step, fmt, unit = 7 * 86400, "KW %V", "pro Woche"
    begin = datetime.fromtimestamp(first)
    begin = begin.replace(minute=0, second=0, microsecond=0) if step == 3600 else \
        begin.replace(hour=0, minute=0, second=0, microsecond=0)
    if step == 7 * 86400:
        begin -= timedelta(days=begin.weekday())
    base = begin.timestamp()
    count = min(int((now - base) // step) + 1, 1000)
    ok, fail = [0] * count, [0] * count
    for record in records:
        index = int((record.epoch - base) // step)
        if 0 <= index < count:
            if record.success:
                ok[index] += 1
            else:
                fail[index] += 1
    labels = [datetime.fromtimestamp(base + i * step).strftime(fmt) for i in range(count)]
    return Trend(labels, ok, fail, unit)


def compute_stats(records: list[ScanRecord], range_key: str, start: float | None, now: float) -> Stats:
    stats = Stats()
    if not records:
        return stats
    total = len(records)
    durations = sorted(r.total_ms for r in records)
    stats.total = total
    stats.success = sum(1 for r in records if r.success)
    stats.avg_ms = sum(durations) / total
    stats.median_ms = durations[total // 2]
    stats.p95_ms = durations[min(total - 1, int(total * 0.95))]
    stats.images = sum(1 for r in records if r.has_image)
    stats.grades = Counter(r.grade for r in records)
    stats.methods = Counter(r.method for r in records)
    stats.flags = Counter(flag for r in records for flag in r.flags if flag != "IMAGE_SAVED")
    stats.last = records[0]
    stats.trend = _trend(records, range_key, start, now)
    return stats


def build_view(store: LogStore, params: tuple, load_s: float) -> View:
    source, range_key = params
    now = time.time()
    start = _range_start(range_key, now)
    records = store.records
    if start is not None:
        records = records[:bisect.bisect_right(records, -start, key=lambda r: -r.epoch)]

    counts, successes = Counter(), Counter()
    for record in records:
        counts[record.source] += 1
        if record.success:
            successes[record.source] += 1
    source_stats = {s: (counts[s], successes[s]) for s in counts}
    source_stats[ALL_SOURCES] = (len(records), sum(successes.values()))

    if source != ALL_SOURCES:
        records = [r for r in records if r.source == source]
    labels = {s: store.label(s) for s in (ALL_SOURCES, *store.sources)}
    return View(params, records, compute_stats(records, range_key, start, now), list(store.sources), labels,
                source_stats, len(store.records), store.log_dir, load_s)


def _open_store() -> LogStore:
    import scan_logger  # lädt OpenCV, daher erst hier im Hintergrund-Thread
    config = _load_config()
    log_dir = scan_logger.resolve_log_directory(config.get("log_dir") or DEFAULT_LOG_DIR)
    names = {str(c["id"]).lower(): str(c["name"]) for c in config.get("cameras") or []
             if isinstance(c, dict) and c.get("id") and c.get("name")}
    return LogStore(log_dir, names)


class LogService:
    """Wird nur im Hintergrund-Thread benutzt: Einlesen, Kennzahlen, Löschen und Export."""

    def __init__(self):
        self.store: LogStore | None = None
        self._params = None
        self._disk_checked = -DISK_INTERVAL_S

    def refresh(self, params: tuple, progress) -> tuple[View | None, tuple | None]:
        started = time.perf_counter()
        if self.store is None:
            self.store = _open_store()
        changed = self.store.refresh(progress)
        disk = None
        if time.monotonic() - self._disk_checked >= DISK_INTERVAL_S:
            self._disk_checked = time.monotonic()
            disk = _disk_usage(self.store.log_dir)
        view = None
        if changed or params != self._params:
            self._params = params
            view = build_view(self.store, params, time.perf_counter() - started)
        return view, disk

    def delete(self, records: list[ScanRecord]) -> tuple[int, int]:
        return self.store.delete(records)

    def export(self, records: list[ScanRecord], path: str) -> tuple[int, str]:
        store = self.store
        full = store.load_full_many(records)
        if path.lower().endswith(".json"):
            data = [full[(r.path, r.offset)] for r in records if (r.path, r.offset) in full]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return len(data), path
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["Scan-ID", "Zeitstempel", "Kamera", "Code", "Erfolg", "Methode", "Konfidenz", "Grade",
                             "Flags", "Dauer_ms", "DMTX", "OCR", "YOLO_Konfidenz", "Kamera-Modell", "Seriennummer",
                             "Trigger", "Port", "Bild"])
            for r in records:
                data = full.get((r.path, r.offset)) or {}
                meta = data.get("meta") or {}
                writer.writerow([
                    r.scan_id, data.get("ts", ""), store.label(r.source), r.code, "JA" if r.success else "NEIN",
                    r.method, f"{r.confidence * 100:.1f}%", r.grade, ", ".join(r.flags), r.total_ms,
                    (data.get("dmtx") or {}).get("text") or "", (data.get("ocr") or {}).get("text") or "",
                    (data.get("detection") or {}).get("yolo_conf", ""), meta.get("camera_model", ""),
                    meta.get("camera_serial", ""), meta.get("trigger", ""), meta.get("port", ""),
                    (data.get("image") or {}).get("path") or "",
                ])
        return len(records), path


# --------------------------------------------------------------------------- #
#  Oberfläche                                                                  #
# --------------------------------------------------------------------------- #
class LogAnalyzerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(f"Scan Log Analyzer  —  DataDetector v{APP_VERSION}")
        self.geometry("1400x850")
        self.minsize(1100, 700)
        self.configure(fg_color=BG_MAIN)

        # Alle Dateizugriffe im Hintergrund-Thread; Ergebnisse kommen über _results in den UI-Thread
        self._tasks: queue.Queue = queue.Queue()
        self._results: queue.Queue = queue.Queue()
        self._service = LogService()
        threading.Thread(target=self._worker_loop, daemon=True).start()

        self._view: View | None = None
        self._source = ALL_SOURCES
        self._range = "Alle"
        self._live = True
        self._refresh_pending = False
        self._refresh_again = False
        self._source_rows: dict[str, tuple] = {}

        # Log-Explorer
        self._filtered: list[ScanRecord] = []
        self._rows: dict[str, ScanRecord] = {}
        self._page_limit = PAGE_SIZE
        self._sort_column: str | None = None
        self._sort_desc = True
        self._all_filtered_selected = False
        self._hover_iid = ""
        self._search_job = None
        self._table_dirty = True

        # Dashboard
        self._charts_dirty = True
        self._chart_job = None
        self._last_chart_draw = 0.0
        self._chart_background = None
        self._chart_actions: dict = {}
        self._hover_artist = None

        # Detailansicht
        self._detail_key: str | None = None
        self._detail_request = 0
        self._detail_image = None
        self._detail_image_path: str | None = None

        self._setup_table_style()
        self._build_ui()

        self._request_refresh()
        self.after(50, self._drain_results)
        self.after(POLL_MS, self._poll)
        self.after(1000, self._tick)

    # ------------------------------------------------------------------ #
    #  Hintergrund-Thread                                                  #
    # ------------------------------------------------------------------ #
    def _worker_loop(self):
        while True:
            kind, payload = self._tasks.get()
            try:
                if kind == "refresh":
                    result = self._service.refresh(payload, lambda p: self._results.put(("progress", p)))
                elif kind == "delete":
                    result = self._service.delete(payload)
                else:
                    result = self._service.export(*payload)
                self._results.put((kind, result))
            except Exception as e:  # Fehler anzeigen statt den Thread zu beenden
                self._results.put(("error", (kind, e)))

    def _drain_results(self):
        try:
            for _ in range(100):
                kind, payload = self._results.get_nowait()
                try:
                    getattr(self, f"_on_{kind}")(payload)
                except Exception:  # ein fehlerhafter Handler darf die Verarbeitung nicht dauerhaft stoppen
                    traceback.print_exc()
        except queue.Empty:
            pass
        self.after(50, self._drain_results)

    def _request_refresh(self):
        if self._refresh_pending:
            self._refresh_again = True
            return
        self._refresh_pending = True
        self._tasks.put(("refresh", (self._source, self._range)))

    def _poll(self):
        if self._live:
            self._request_refresh()
        self.after(POLL_MS, self._poll)

    def _tick(self):
        last = self._view.stats.last if self._view else None
        if last is not None and last.epoch:
            self.kpi_last.configure(text=_fmt_ago(time.time() - last.epoch))
        self.after(1000, self._tick)

    def _on_progress(self, fraction: float):
        if self._view is None:
            self.loading_label.configure(text=f"Lade Logs ...  {fraction * 100:.0f} %")

    def _on_refresh(self, result: tuple):
        self._refresh_pending = False
        view, disk = result
        if disk is not None:
            used, free = disk
            color = DANGER if used > 70 else (WARN if used > 50 else TXT_MID)
            self.disk_label.configure(text=f"Laufwerk: {_fmt_pct(used)} belegt\n({free:.1f} GB frei)", text_color=color)
        if view is not None and view.params == (self._source, self._range):
            self._apply_view(view)
        if self._refresh_again:
            self._refresh_again = False
            self._request_refresh()

    def _on_error(self, payload: tuple):
        kind, error = payload
        if kind == "refresh":
            self._refresh_pending = False
            self.status_label.configure(text=f"Fehler beim Laden:\n{error}", text_color=DANGER)
            self.loading_label.configure(text=f"Logs konnten nicht geladen werden:\n{error}")
        else:
            action = "Löschen" if kind == "delete" else "Export"
            messagebox.showerror(action, f"{action} fehlgeschlagen:\n{error}")

    def _apply_view(self, view: View):
        self._view = view
        self.loading_label.place_forget()
        if self._source != ALL_SOURCES and self._source not in view.sources:
            self._source = ALL_SOURCES  # Kameraordner existiert nicht mehr
            self._request_refresh()
        self._update_sources(view)
        self._update_kpis()
        self.status_label.configure(
            text=f"{_fmt_int(view.total_loaded)} Datensätze geladen\n"
                 f"Aktualisiert {datetime.now():%H:%M:%S} ({view.load_s:.2f} s)",
            text_color=TXT_MID)
        self._table_dirty = self._charts_dirty = True
        self._update_visible_tab()

    # ------------------------------------------------------------------ #
    #  UI Aufbau                                                           #
    # ------------------------------------------------------------------ #
    def _setup_table_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        self._table_font = tkfont.Font(family="Segoe UI", size=10)
        row_height = self._table_font.metrics("linespace") + 12
        style.configure("Log.Treeview", background=BG_CARD, fieldbackground=BG_CARD, foreground=TXT_DARK,
                        rowheight=row_height, font=self._table_font, borderwidth=0, relief="flat")
        style.configure("Log.Treeview.Heading", background=BG_SIDE, foreground=TXT_MID, relief="flat",
                        borderwidth=0, font=("Segoe UI", 10, "bold"), padding=(6, 6))
        style.map("Log.Treeview", background=[("selected", SELECT_ROW)], foreground=[("selected", TXT_DARK)])
        style.map("Log.Treeview.Heading", background=[("active", BORDER)])
        style.layout("Log.Treeview", [("Treeview.treearea", {"sticky": "nswe"})])
        style.layout("Log.Treeview.Item", [("Treeitem.padding", {"sticky": "nswe", "children": [
            ("Treeitem.image", {"side": "left", "sticky": ""}),
            ("Treeitem.text", {"side": "left", "sticky": ""}),
        ]})])
        self._badges = self._make_badges(max(14, row_height - 12))

    @staticmethod
    def _make_badges(height: int) -> dict:
        """Farbige Grade-Plaketten für die Tabelle (4-fach gerendert und verkleinert für glatte Kanten)."""
        scale, width = 4, int(height * 1.7)
        try:
            font = ImageFont.truetype("segoeuib.ttf", int(height * 0.7 * scale))
        except OSError:
            font = ImageFont.load_default()
        badges = {}
        for grade, color in GRADE_COLORS.items():
            image = Image.new("RGBA", (width * scale, height * scale), (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            draw.rounded_rectangle((0, 0, width * scale - 1, height * scale - 1), radius=height * scale // 4, fill=color)
            try:
                draw.text((width * scale / 2, height * scale / 2), grade, fill="white", font=font, anchor="mm")
            except (TypeError, ValueError):
                draw.text((width * scale / 3, 0), grade, fill="white", font=font)
            badges[grade] = ImageTk.PhotoImage(image.resize((width, height), Image.Resampling.LANCZOS))
        return badges

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_sidebar()

        self.tabview = ctk.CTkTabview(self, fg_color=BG_CARD, segmented_button_selected_color=ACCENT,
                                      command=self._update_visible_tab)
        self.tabview.grid(row=0, column=1, padx=16, pady=16, sticky="nsew")
        self.tab_dashboard = self.tabview.add("Dashboard")
        self.tab_explorer = self.tabview.add("Log-Explorer")
        self._build_dashboard()
        self._build_explorer()

        self.loading_label = ctk.CTkLabel(self.tabview, text="Lade Logs ...", fg_color=BG_CARD,
                                          font=ctk.CTkFont(size=15, weight="bold"), text_color=TXT_MID)
        self.loading_label.place(relx=0.5, rely=0.5, anchor="center")

        self.btn_open_sidebar = ctk.CTkButton(
            self, text=">", width=36, height=36, corner_radius=8,
            fg_color=ACCENT, text_color="#FFFFFF", hover_color="#1D4ED8",
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"), command=self._toggle_sidebar
        )
        self._build_detail_panel()

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
        ctk.CTkLabel(header, text="Log Analyzer", text_color=ACCENT,
                     font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold")).grid(row=0, column=0, sticky="w")
        ctk.CTkButton(
            header, text="<", width=32, height=32, corner_radius=6,
            fg_color="transparent", text_color=TXT_DARK, hover_color=BORDER,
            font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"), command=self._toggle_sidebar
        ).grid(row=0, column=1, sticky="e")
        ctk.CTkLabel(self.sidebar, text=f"Statistiken & Diagnose v{APP_VERSION}",
                     font=ctk.CTkFont(size=11), text_color=TXT_LIGHT).grid(row=1, column=0, padx=20, pady=(0, 14), sticky="w")

        ctk.CTkLabel(self.sidebar, text="Zeitraum:", font=ctk.CTkFont(weight="bold", size=13),
                     text_color=TXT_DARK).grid(row=2, column=0, padx=20, pady=(0, 2), sticky="w")
        self.range_selector = ctk.CTkSegmentedButton(
            self.sidebar, values=list(TIME_RANGES), command=self._on_range_changed,
            selected_color=ACCENT, selected_hover_color="#1D4ED8"
        )
        self.range_selector.set(self._range)
        self.range_selector.grid(row=3, column=0, padx=16, pady=(2, 12), sticky="ew")

        ctk.CTkLabel(self.sidebar, text="Kamera / Log-Quelle:", font=ctk.CTkFont(weight="bold", size=13),
                     text_color=TXT_DARK).grid(row=4, column=0, padx=20, pady=(0, 2), sticky="w")
        self.sources_frame = ctk.CTkScrollableFrame(self.sidebar, fg_color="transparent", label_text="Verfügbare Kameras")
        self.sources_frame.grid(row=5, column=0, padx=12, pady=(4, 10), sticky="nsew")

        live_row = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        live_row.grid(row=6, column=0, padx=20, pady=(0, 4), sticky="ew")
        self.live_switch = ctk.CTkSwitch(live_row, text="Live Auto-Update (2s)", text_color=SUCCESS,
                                         font=ctk.CTkFont(size=12, weight="bold"), command=self._toggle_live)
        self.live_switch.select()
        self.live_switch.pack(side="left")
        ctk.CTkButton(live_row, text="⟳", width=30, height=28, fg_color=TXT_MID, hover_color="#64748B",
                      command=self._request_refresh).pack(side="right")

        ctk.CTkButton(
            self.sidebar, text="📂  Log-Ordner öffnen", height=30, fg_color="transparent",
            border_width=1, border_color=BORDER, text_color=TXT_DARK, hover_color=BORDER, command=self._open_log_dir
        ).grid(row=7, column=0, padx=20, pady=(6, 4), sticky="ew")

        self.status_label = ctk.CTkLabel(self.sidebar, text="Lade Logs ...", font=ctk.CTkFont(size=11),
                                         text_color=TXT_MID, justify="left")
        self.status_label.grid(row=8, column=0, padx=20, pady=(4, 0), sticky="w")
        self.disk_label = ctk.CTkLabel(self.sidebar, text="Laufwerk: Prüfe...", font=ctk.CTkFont(size=11),
                                       text_color=TXT_MID, justify="left")
        self.disk_label.grid(row=9, column=0, padx=20, pady=(2, 14), sticky="w")

    def _build_dashboard(self):
        tab = self.tab_dashboard
        for column in range(5):
            tab.grid_columnconfigure(column, weight=1, uniform="kpi")
        tab.grid_rowconfigure(1, weight=1)

        self.kpi_total, self.kpi_total_sub = self._kpi_card(0, "Scans gesamt", TXT_DARK)
        self.kpi_rate, rate_sub = self._kpi_card(1, "Erfolgsrate", SUCCESS, plain_sub=False)
        self.lbl_success = ctk.CTkLabel(rate_sub, text="", text_color=SUCCESS, font=ctk.CTkFont(size=12, weight="bold"))
        self.lbl_success.pack(side="left", padx=(0, 10))
        self.lbl_fail = ctk.CTkLabel(rate_sub, text="", text_color=DANGER, font=ctk.CTkFont(size=12, weight="bold"))
        self.lbl_fail.pack(side="left")
        self.kpi_duration, self.kpi_duration_sub = self._kpi_card(2, "Ø Scandauer", ACCENT)
        self.kpi_last, self.kpi_last_sub = self._kpi_card(3, "Letzter Scan", TXT_DARK)
        self.kpi_images, self.kpi_images_sub = self._kpi_card(4, "Bilder im Speicher", WARN)

        chart_frame = ctk.CTkFrame(tab, fg_color="transparent")
        chart_frame.grid(row=1, column=0, columnspan=5, sticky="nsew", padx=4, pady=(0, 4))
        self._fig = Figure(figsize=(10, 5.4), dpi=100, facecolor=BG_CARD)
        self._fig.subplots_adjust(left=0.08, right=0.95, top=0.91, bottom=0.08, wspace=0.45, hspace=0.62)
        self.ax_grades = self._fig.add_subplot(221)
        self.ax_methods = self._fig.add_subplot(222)
        self.ax_trend = self._fig.add_subplot(223)
        self.ax_trend_rate = self.ax_trend.twinx()
        self.ax_flags = self._fig.add_subplot(224)
        self._chart_canvas = FigureCanvasTkAgg(self._fig, master=chart_frame)
        chart_widget = self._chart_canvas.get_tk_widget()
        chart_widget.configure(highlightthickness=0, background=BG_CARD)
        chart_widget.pack(fill="both", expand=True)
        self._chart_canvas.mpl_connect("draw_event", self._on_chart_drawn)
        self._chart_canvas.mpl_connect("motion_notify_event", self._on_chart_motion)
        self._chart_canvas.mpl_connect("figure_leave_event", lambda _e: self._set_chart_hover(None))
        self._chart_canvas.mpl_connect("button_press_event", self._on_chart_click)

    def _kpi_card(self, column: int, title: str, color: str, plain_sub: bool = True):
        card = ctk.CTkFrame(self.tab_dashboard, fg_color=BG_SIDE, border_width=1, border_color=BORDER, corner_radius=8)
        card.grid(row=0, column=column, padx=6, pady=(10, 12), sticky="ew")
        ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=11, weight="bold"), text_color=TXT_MID).pack(pady=(10, 0))
        value = ctk.CTkLabel(card, text="—", font=ctk.CTkFont(size=22, weight="bold"), text_color=color)
        value.pack()
        if plain_sub:
            sub = ctk.CTkLabel(card, text="", font=ctk.CTkFont(size=10), text_color=TXT_MID)
        else:
            sub = ctk.CTkFrame(card, fg_color="transparent")
        sub.pack(pady=(0, 10))
        self._bind_hover(card, BG_SIDE, HOVER_CARD)
        return value, sub

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

    def _build_explorer(self):
        tab = self.tab_explorer
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)
        button_font = ctk.CTkFont(size=12, weight="bold")

        toolbar = ctk.CTkFrame(tab, fg_color="transparent")
        toolbar.grid(row=0, column=0, padx=10, pady=(8, 6), sticky="ew")
        row1 = ctk.CTkFrame(toolbar, fg_color="transparent")
        row1.pack(fill="x", pady=(0, 6))
        row1.grid_columnconfigure(0, weight=1)
        self.search_entry = ctk.CTkEntry(row1, placeholder_text="Suche nach Code, ID, Kamera, Flags...")
        self.search_entry.grid(row=0, column=0, padx=(0, 8), sticky="ew")
        self.search_entry.bind("<KeyRelease>", self._on_search_key)
        ctk.CTkButton(row1, text="☑ Alle auswählen", width=120, fg_color=BG_SIDE, text_color=TXT_DARK,
                      hover_color=BORDER, border_width=1, border_color=BORDER, font=button_font,
                      command=self._select_all).grid(row=0, column=1, padx=4)
        ctk.CTkButton(row1, text="🗑️ Löschen", width=100, fg_color=DANGER, hover_color="#991B1B",
                      font=button_font, command=self._delete_selected).grid(row=0, column=2, padx=4)
        ctk.CTkButton(row1, text="📥 Exportieren", width=110, fg_color=ACCENT, hover_color="#1D4ED8",
                      font=button_font, command=self._export).grid(row=0, column=3, padx=(4, 0))

        row2 = ctk.CTkFrame(toolbar, fg_color="transparent")
        row2.pack(fill="x")
        ctk.CTkButton(row2, text="Filter zurücksetzen", width=120, fg_color="transparent", text_color=ACCENT,
                      hover_color=BORDER, command=self._reset_filters).pack(side="right", padx=(4, 0))
        ctk.CTkLabel(row2, text="Filter:", font=ctk.CTkFont(size=11, weight="bold"), text_color=TXT_MID).pack(side="left", padx=(0, 6))
        menu_style = {"fg_color": BG_SIDE, "text_color": TXT_DARK, "button_color": BORDER, "button_hover_color": TXT_LIGHT}
        self.filter_grade = ctk.CTkOptionMenu(row2, values=list(GRADE_FILTERS), width=112,
                                              command=self._on_filters_changed, **menu_style)
        self.filter_grade.pack(side="left", padx=4)
        self.filter_status = ctk.CTkOptionMenu(row2, values=list(STATUS_FILTERS), width=132,
                                               command=self._on_filters_changed, **menu_style)
        self.filter_status.pack(side="left", padx=4)
        self.filter_slow = ctk.CTkCheckBox(row2, text="Langsam (>12 s)", command=self._on_filters_changed,
                                           text_color=TXT_DARK, fg_color=ACCENT, hover_color=ACCENT)
        self.filter_slow.pack(side="left", padx=(10, 4))
        self.filter_images = ctk.CTkCheckBox(row2, text="Mit Bild", command=self._on_filters_changed,
                                             text_color=TXT_DARK, fg_color=ACCENT, hover_color=ACCENT)
        self.filter_images.pack(side="left", padx=4)

        table = ctk.CTkFrame(tab, fg_color=BG_CARD, border_width=1, border_color=BORDER, corner_radius=8)
        table.grid(row=1, column=0, padx=10, pady=(0, 6), sticky="nsew")
        table.grid_columnconfigure(0, weight=1)
        table.grid_rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(table, style="Log.Treeview", columns=[c[0] for c in COLUMNS],
                                 show=("tree", "headings"), selectmode="extended")
        self.tree.grid(row=0, column=0, sticky="nsew", padx=(6, 0), pady=(6, 0))
        scrollbar = ctk.CTkScrollbar(table, command=self.tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns", padx=(0, 4), pady=6)
        x_scrollbar = ctk.CTkScrollbar(table, orientation="horizontal", command=self.tree.xview)
        x_scrollbar.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 4))
        self.tree.configure(yscrollcommand=scrollbar.set, xscrollcommand=x_scrollbar.set)
        # Spaltenbreiten sind Pixel: an die Windows-Skalierung anpassen
        scaling = ctk.ScalingTracker.get_widget_scaling(self)
        self.tree.column("#0", width=int(58 * scaling), minwidth=int(58 * scaling), stretch=False, anchor="center")
        for column, _title, width, stretch in COLUMNS:
            self.tree.column(column, width=int(width * scaling), minwidth=int(50 * scaling), stretch=stretch, anchor="w")
        self._update_headings()
        self.tree.tag_configure("fail", foreground=DANGER)
        self.tree.tag_configure("hover", background=HOVER_ROW)
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self.tree.bind("<Motion>", self._on_tree_motion)
        self.tree.bind("<Leave>", lambda _e: self._set_row_hover(""))
        self.tree.bind("<Control-a>", self._select_all)
        self.tree.bind("<Escape>", lambda _e: self.tree.selection_set([]))
        self.tree.bind("<Delete>", self._delete_selected)

        self.empty_label = ctk.CTkLabel(table, text="", fg_color=BG_CARD, text_color=TXT_LIGHT, font=ctk.CTkFont(size=13))
        footer = ctk.CTkFrame(tab, fg_color="transparent")
        footer.grid(row=2, column=0, padx=12, pady=(0, 6), sticky="ew")
        self.count_label = ctk.CTkLabel(footer, text="0 Einträge", font=ctk.CTkFont(size=11, weight="bold"), text_color=TXT_MID)
        self.count_label.pack(side="left")
        self.more_button = ctk.CTkButton(footer, text="", fg_color="transparent", text_color=ACCENT, hover_color=BORDER,
                                         font=ctk.CTkFont(size=12, weight="bold"), command=self._load_more)

    def _build_detail_panel(self):
        self.detail_panel = ctk.CTkFrame(self, width=380, corner_radius=12, fg_color=BG_CARD,
                                         border_width=1, border_color=BORDER)
        self.detail_panel.grid(row=0, column=2, padx=(0, 16), pady=16, sticky="nsew")
        self.detail_panel.grid_propagate(False)
        self.detail_panel.grid_rowconfigure(2, weight=1)
        self.detail_panel.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(self.detail_panel, fg_color="transparent")
        header.grid(row=0, column=0, padx=16, pady=(12, 4), sticky="ew")
        ctk.CTkLabel(header, text="SCAN-DETAILS", font=ctk.CTkFont(size=12, weight="bold"), text_color=TXT_LIGHT).pack(side="left")
        ctk.CTkButton(header, text="✕", width=26, height=24, fg_color="transparent", text_color=TXT_MID,
                      hover_color=BORDER, font=ctk.CTkFont(size=13, weight="bold"),
                      command=self.detail_panel.grid_remove).pack(side="right")

        self.image_container = ctk.CTkFrame(self.detail_panel, height=DETAIL_IMAGE_SIZE[1] + 10, fg_color=BG_MAIN, corner_radius=8)
        self.image_container.grid_propagate(False)
        self.image_label = ctk.CTkLabel(self.image_container, text="", cursor="hand2")
        self.image_label.place(relx=0.5, rely=0.5, anchor="center")
        self.image_label.bind("<Button-1>", self._open_detail_image)

        details = ctk.CTkScrollableFrame(self.detail_panel, fg_color="transparent")
        details.grid(row=2, column=0, padx=8, pady=(6, 12), sticky="nsew")
        self.detail_labels = {}
        for key in DETAIL_FIELDS:
            row = ctk.CTkFrame(details, fg_color="transparent")
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text=f"{key}:", font=ctk.CTkFont(size=11, weight="bold"), text_color=TXT_MID,
                         anchor="nw", width=110).pack(side="left", anchor="nw")
            value = ctk.CTkLabel(row, text="—", font=ctk.CTkFont(size=11), text_color=TXT_DARK,
                                 anchor="nw", justify="left", wraplength=210)
            value.pack(side="left", fill="x", expand=True)
            self.detail_labels[key] = value
        ctk.CTkLabel(details, text="Tipp: Klick aufs Bild öffnet es in voller Größe.", font=ctk.CTkFont(size=10),
                     text_color=TXT_LIGHT).pack(anchor="w", pady=(10, 0))

    # ------------------------------------------------------------------ #
    #  Seitenleiste                                                        #
    # ------------------------------------------------------------------ #
    def _toggle_sidebar(self):
        if self.sidebar.winfo_ismapped():
            self.sidebar.grid_remove()
            self.btn_open_sidebar.place(x=12, y=24)
            self.btn_open_sidebar.lift()
        else:
            self.btn_open_sidebar.place_forget()
            self.sidebar.grid()

    def _toggle_live(self):
        self._live = bool(self.live_switch.get())
        if self._live:
            self.live_switch.configure(text_color=SUCCESS, text="Live Auto-Update (2s)")
            self._request_refresh()
        else:
            self.live_switch.configure(text_color=TXT_MID, text="Live Auto-Update (PAUSIERT)")

    def _on_range_changed(self, value: str):
        self._range = value
        self._request_refresh()

    def _select_source(self, key: str):
        if key == self._source:
            return
        self._source = key
        for source, (row, _name, _info) in self._source_rows.items():
            row.configure(fg_color=BORDER if source == key else "transparent")
        self._request_refresh()

    def _update_sources(self, view: View):
        keys = [ALL_SOURCES, *view.sources]
        if list(self._source_rows) != keys:
            for row, _name, _info in self._source_rows.values():
                row.destroy()
            self._source_rows = {key: self._make_source_row(key) for key in keys}
        for key, (row, name, info) in self._source_rows.items():
            count, ok = view.source_stats.get(key, (0, 0))
            icon = "★" if key == ALL_SOURCES else "📷"
            name.configure(text=f"{icon}  {view.labels.get(key, key)}")
            info.configure(text=f"{_fmt_int(count)} Scans" + (f"  ·  {_fmt_pct(ok / count * 100)} Erfolg" if count else ""))
            row.configure(fg_color=BORDER if key == self._source else "transparent")

    def _make_source_row(self, key: str) -> tuple:
        row = ctk.CTkFrame(self.sources_frame, fg_color="transparent", corner_radius=6)
        row.pack(fill="x", pady=2)
        name = ctk.CTkLabel(row, text="", anchor="w", height=20, font=ctk.CTkFont(size=12, weight="bold"),
                            text_color=ACCENT if key == ALL_SOURCES else TXT_DARK)
        name.pack(fill="x", padx=10, pady=(5, 0))
        info = ctk.CTkLabel(row, text="", anchor="w", height=16, font=ctk.CTkFont(size=10), text_color=TXT_MID)
        info.pack(fill="x", padx=10, pady=(0, 5))
        for widget in (row, name, info):
            widget.bind("<Button-1>", lambda _e, k=key: self._select_source(k))
        self._bind_hover(row, "transparent", BG_MAIN, is_active=lambda k=key: k == self._source)
        return row, name, info

    def _open_log_dir(self):
        if self._view is not None and os.path.isdir(self._view.log_dir):
            os.startfile(self._view.log_dir)

    # ------------------------------------------------------------------ #
    #  Dashboard                                                           #
    # ------------------------------------------------------------------ #
    def _update_visible_tab(self):
        if self.tabview.get() == "Dashboard":
            if self._charts_dirty:
                self._schedule_charts()
        elif self._table_dirty:
            self._apply_filters()

    def _update_kpis(self):
        stats = self._view.stats
        if not stats.total:
            for label in (self.kpi_total, self.kpi_rate, self.kpi_duration, self.kpi_last, self.kpi_images):
                label.configure(text="—")
            for label in (self.kpi_total_sub, self.kpi_duration_sub, self.kpi_last_sub, self.kpi_images_sub,
                          self.lbl_success, self.lbl_fail):
                label.configure(text="")
            return
        labels = self._view.labels
        self.kpi_total.configure(text=_fmt_int(stats.total))
        self.kpi_total_sub.configure(text=f"{labels.get(self._source, '')}  ·  {self._range}")
        self.kpi_rate.configure(text=_fmt_pct(stats.success / stats.total * 100))
        self.lbl_success.configure(text=f"✓ {_fmt_int(stats.success)}")
        self.lbl_fail.configure(text=f"✗ {_fmt_int(stats.total - stats.success)}")
        self.kpi_duration.configure(text=f"{_fmt_int(round(stats.avg_ms))} ms")
        self.kpi_duration_sub.configure(text=f"Median {_fmt_int(stats.median_ms)}  ·  P95 {_fmt_int(stats.p95_ms)}")
        last = stats.last
        self.kpi_last.configure(text=_fmt_ago(time.time() - last.epoch) if last.epoch else "—")
        self.kpi_last_sub.configure(text=f"{last.code or 'Fehlgeschlagen'}  ·  {labels.get(last.source, last.source)}",
                                    text_color=SUCCESS if last.success else DANGER)
        self.kpi_images.configure(text=_fmt_int(stats.images))
        self.kpi_images_sub.configure(text=f"{_fmt_pct(stats.images / stats.total * 100)} der Scans")

    def _schedule_charts(self):
        if self._chart_job is not None:
            return
        wait = max(0.0, CHART_INTERVAL_S - (time.monotonic() - self._last_chart_draw))
        self._chart_job = self.after(int(wait * 1000), self._draw_charts)

    def _draw_charts(self):
        self._chart_job = None
        if self.tabview.get() != "Dashboard":
            return
        self._last_chart_draw = time.monotonic()
        self._charts_dirty = False
        stats = self._view.stats if self._view else Stats()
        self._chart_actions = {}
        self._draw_grades(stats)
        self._draw_methods(stats)
        self._draw_trend(stats)
        self._draw_flags(stats)
        self._chart_canvas.draw_idle()

    @staticmethod
    def _style_axis(ax, grid_axis: str = "y"):
        ax.grid(axis=grid_axis, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.tick_params(colors=TXT_MID, labelsize=9)
        (ax.yaxis if grid_axis == "y" else ax.xaxis).set_major_formatter(FuncFormatter(lambda v, _p: _fmt_int(v)))
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BORDER)

    @staticmethod
    def _no_data(ax, text: str = "Keine Daten"):
        ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, color=TXT_LIGHT, fontsize=10)
        ax.set_axis_off()

    def _new_chart(self, ax, title: str):
        ax.clear()
        ax.set_axis_on()
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=TXT_DARK)

    def _draw_grades(self, stats: Stats):
        ax = self.ax_grades
        self._new_chart(ax, "Qualitäts-Verteilung (Grades)")
        grades = [(grade, count) for grade, count in sorted(stats.grades.items()) if count]
        if not grades:
            self._no_data(ax)
            return
        wedges, _labels, percents = ax.pie(
            [count for _, count in grades], labels=[f"Grade {grade}" for grade, _ in grades],
            colors=[GRADE_COLORS.get(grade, TXT_LIGHT) for grade, _ in grades], startangle=90, counterclock=False,
            autopct=lambda p: f"{p:.0f}%" if p >= 4 else "", pctdistance=0.78,
            wedgeprops={"width": 0.45, "edgecolor": "white", "linewidth": 1.5},
            textprops={"color": TXT_DARK, "fontsize": 9},
        )
        for text in percents:
            text.set_color("white")
            text.set_fontweight("bold")
            text.set_fontsize(8)
        ax.text(0, 0, f"{_fmt_int(stats.total)}\nScans", ha="center", va="center", fontsize=10,
                fontweight="bold", color=TXT_DARK)
        for wedge, (grade, _count) in zip(wedges, grades):
            self._chart_actions[wedge] = ("grade", grade)

    def _draw_methods(self, stats: Stats):
        ax = self.ax_methods
        self._new_chart(ax, "Erfolgsmethoden")
        items = stats.methods.most_common()
        if not items:
            self._no_data(ax)
            return
        names = [name for name, _ in items]
        counts = [count for _, count in items]
        bars = ax.bar(names, counts, color=[METHOD_COLORS.get(name, TXT_LIGHT) for name in names], width=0.55)
        ax.bar_label(bars, labels=[_fmt_int(c) for c in counts], padding=3, fontsize=9, color=TXT_MID)
        ax.margins(y=0.18)
        self._style_axis(ax)
        for bar, name in zip(bars, names):
            self._chart_actions[bar] = ("method", name)

    def _draw_trend(self, stats: Stats):
        ax, rate_ax = self.ax_trend, self.ax_trend_rate
        trend = stats.trend
        self._new_chart(ax, f"Verlauf ({trend.unit})" if trend else "Verlauf")
        rate_ax.clear()
        rate_ax.patch.set_visible(False)
        if trend is None:
            self._no_data(ax)
            rate_ax.set_axis_off()
            return
        rate_ax.set_axis_on()
        x = list(range(len(trend.labels)))
        ok_bars = ax.bar(x, trend.ok, color=SUCCESS, width=0.8, label="Erfolg")
        fail_bars = ax.bar(x, trend.fail, bottom=trend.ok, color=DANGER, width=0.8, label="Fehler")
        rates = [ok / (ok + fail) * 100 if ok + fail else float("nan") for ok, fail in zip(trend.ok, trend.fail)]
        (rate_line,) = rate_ax.plot(x, rates, color=ACCENT, linewidth=1.4, marker="o", markersize=2.5, label="Erfolgsrate")
        # Oberes Viertel frei für die Legende: Balken und Erfolgsrate enden bei ~75 % der Höhe
        ax.set_ylim(0, max(max(o + f for o, f in zip(trend.ok, trend.fail)), 1) * 1.33)
        rate_ax.set_ylim(0, 133)
        rate_ax.set_yticks([0, 50, 100], ["0 %", "50 %", "100 %"])
        rate_ax.yaxis.tick_right()
        rate_ax.tick_params(colors=ACCENT, labelsize=8)
        for side in ("top", "left", "bottom"):
            rate_ax.spines[side].set_visible(False)
        rate_ax.spines["right"].set_color(BORDER)
        step = max(1, -(-len(x) // 6))
        ax.set_xticks(x[::step], trend.labels[::step])
        self._style_axis(ax)
        ax.tick_params(axis="x", labelsize=8)
        ax.legend(handles=[ok_bars, fail_bars, rate_line], loc="upper left", ncols=3, fontsize=8, frameon=False,
                  handlelength=1.2, columnspacing=0.8, borderaxespad=0.2)

    def _draw_flags(self, stats: Stats):
        ax = self.ax_flags
        self._new_chart(ax, "Häufigste Diagnose-Flags")
        items = stats.flags.most_common(7)
        if not items:
            self._no_data(ax, "Keine Auffälligkeiten")
            return
        names = [name for name, _ in reversed(items)]
        counts = [count for _, count in reversed(items)]
        bars = ax.barh(names, counts, color=WARN, height=0.55)
        ax.bar_label(bars, labels=[_fmt_int(c) for c in counts], padding=3, fontsize=8, color=TXT_MID)
        ax.margins(x=0.2)
        self._style_axis(ax, grid_axis="x")
        ax.tick_params(axis="y", labelsize=8)
        for bar, name in zip(bars, names):
            self._chart_actions[bar] = ("flag", name)

    def _on_chart_drawn(self, _event):
        self._chart_background = self._chart_canvas.copy_from_bbox(self._fig.bbox)
        self._hover_artist = None

    def _on_chart_motion(self, event):
        artist = None
        if event.inaxes is not None:
            artist = next((a for a in self._chart_actions if a.contains(event)[0]), None)
        self._set_chart_hover(artist)

    def _set_chart_hover(self, artist):
        """Hover-Hervorhebung per Blitting: nur das Element wird übermalt, das Diagramm nicht neu gezeichnet."""
        if artist is self._hover_artist:
            return
        self._hover_artist = artist
        self._chart_canvas.get_tk_widget().configure(cursor="hand2" if artist is not None else "")
        if self._chart_background is None:
            return
        self._chart_canvas.restore_region(self._chart_background)
        if artist is not None:
            saved = (artist.get_alpha(), artist.get_linewidth(), artist.get_edgecolor())
            radius = artist.r if isinstance(artist, Wedge) else None
            if radius is not None:
                artist.set_radius(radius * 1.06)
            artist.set_alpha(0.85)
            artist.set_linewidth(2)
            artist.set_edgecolor(TXT_DARK)
            artist.axes.draw_artist(artist)
            artist.set_alpha(saved[0])
            artist.set_linewidth(saved[1])
            artist.set_edgecolor(saved[2])
            if radius is not None:
                artist.set_radius(radius)
        self._chart_canvas.blit(self._fig.bbox)

    def _on_chart_click(self, event):
        if event.button != 1 or self._hover_artist is None:
            return
        kind, value = self._chart_actions.get(self._hover_artist, (None, None))
        if kind is None:
            return
        self._reset_filters(apply=False)
        if kind == "grade":
            self.filter_grade.set(f"Grade {value}")
        elif kind == "method" and value in STATUS_FILTERS:
            self.filter_status.set(value)
        else:
            self.search_entry.insert(0, value)
        self._set_chart_hover(None)
        self.tabview.set("Log-Explorer")
        self._on_filters_changed()

    # ------------------------------------------------------------------ #
    #  Log-Explorer                                                        #
    # ------------------------------------------------------------------ #
    def _on_search_key(self, _event=None):
        if self._search_job is not None:
            self.after_cancel(self._search_job)
        self._search_job = self.after(200, self._on_filters_changed)

    def _on_filters_changed(self, *_args):
        self._search_job = None
        self._all_filtered_selected = False
        self._apply_filters(reset_page=True)

    def _reset_filters(self, apply: bool = True):
        self.search_entry.delete(0, "end")
        self.filter_grade.set(GRADE_FILTERS[0])
        self.filter_status.set(STATUS_FILTERS[0])
        self.filter_slow.deselect()
        self.filter_images.deselect()
        if apply:
            self._on_filters_changed()

    def _apply_filters(self, reset_page: bool = False):
        self._table_dirty = False
        if reset_page:
            self._page_limit = PAGE_SIZE
        records = self._view.records if self._view else []
        query = self.search_entry.get().strip().lower()
        if query:
            records = [r for r in records if query in r.search]
        grade = self.filter_grade.get()
        if grade != GRADE_FILTERS[0]:
            grade = grade[-1]
            records = [r for r in records if r.grade == grade]
        status = self.filter_status.get()
        if status == "Erfolg":
            records = [r for r in records if r.success]
        elif status == "Fehlgeschlagen":
            records = [r for r in records if not r.success]
        elif status != STATUS_FILTERS[0]:
            records = [r for r in records if r.method == status]
        if self.filter_slow.get():
            records = [r for r in records if r.total_ms > SLOW_SCAN_MS]
        if self.filter_images.get():
            records = [r for r in records if r.has_image]
        if self._sort_column is not None:
            records = sorted(records, key=SORT_KEYS[self._sort_column], reverse=self._sort_desc)
        self._filtered = records
        self._render_table()

    def _render_table(self):
        tree = self.tree
        selected = set(tree.selection())
        anchor = ""  # oberste sichtbare Zeile bleibt beim Live-Update stehen
        if self._rows:
            keys = list(self._rows)
            top = min(len(keys) - 1, round(tree.yview()[0] * len(keys)))
            anchor = keys[top] if top > 0 else ""
        rows = self._filtered[:self._page_limit]
        labels = self._view.labels if self._view else {}

        self._hover_iid = ""
        tree.delete(*tree.get_children())
        self._rows = {}
        for record in rows:
            self._rows[record.key] = record
            when = datetime.fromtimestamp(record.epoch).strftime("%d.%m. %H:%M:%S") if record.epoch else "—"
            tree.insert("", "end", iid=record.key, image=self._badges.get(record.grade, ""),
                        tags=() if record.success else ("fail",),
                        values=(when, labels.get(record.source, record.source), record.code or "—", record.method,
                                f"{record.confidence * 100:.0f} %", f"{_fmt_int(record.total_ms)} ms",
                                ", ".join(record.flags), record.scan_id))
        keep = [iid for iid in selected if iid in self._rows]
        if keep:
            tree.selection_set(keep)
        if anchor in self._rows:
            tree.yview_moveto(tree.index(anchor) / max(1, len(rows)))

        if rows:
            self.empty_label.place_forget()
        else:
            has_data = bool(self._view and self._view.records)
            self.empty_label.configure(text="Keine Einträge für diese Filter." if has_data else "Keine Scans im gewählten Zeitraum.")
            self.empty_label.place(relx=0.5, rely=0.4, anchor="center")
        remaining = len(self._filtered) - len(rows)
        if remaining > 0:
            self.more_button.configure(
                text=f"▼  Weitere {_fmt_int(min(PAGE_SIZE, remaining))} von {_fmt_int(remaining)} Einträgen anzeigen")
            self.more_button.pack(side="right")
        else:
            self.more_button.pack_forget()
        self._update_count_label()

    def _load_more(self):
        self._page_limit += PAGE_SIZE
        self._render_table()

    def _sort_by(self, column: str):
        if self._sort_column == column:
            self._sort_desc = not self._sort_desc
        else:
            self._sort_column, self._sort_desc = column, column in DESCENDING_FIRST
        self._update_headings()
        self._apply_filters(reset_page=True)

    def _update_headings(self):
        def title(column: str, text: str) -> str:
            if column != self._sort_column:
                return text
            return f"{text} {'▼' if self._sort_desc else '▲'}"
        self.tree.heading("#0", text=title("#0", "Grade"), command=lambda: self._sort_by("#0"))
        for column, text, _width, _stretch in COLUMNS:
            self.tree.heading(column, text=title(column, text), anchor="w", command=lambda c=column: self._sort_by(c))

    def _update_count_label(self):
        text = f"{_fmt_int(len(self._filtered))} Einträge"
        if self._all_filtered_selected:
            text += f"  ·  alle {_fmt_int(len(self._filtered))} ausgewählt"
        elif self.tree.selection():
            text += f"  ·  {_fmt_int(len(self.tree.selection()))} ausgewählt"
        self.count_label.configure(text=text)

    def _on_tree_select(self, _event=None):
        selection = self.tree.selection()
        if self._all_filtered_selected and len(selection) != len(self._rows):
            self._all_filtered_selected = False
        self._update_count_label()
        focus = self.tree.focus()
        key = focus if focus in selection else (selection[0] if selection else None)
        if key in self._rows and key != self._detail_key:
            if key == self._hover_iid:
                self._set_row_hover("")
            self._show_details(self._rows[key])

    def _on_tree_motion(self, event):
        iid = self.tree.identify_row(event.y)
        self._set_row_hover("" if iid in self.tree.selection() else iid)

    def _set_row_hover(self, iid: str):
        if iid == self._hover_iid:
            return
        tree = self.tree
        if self._hover_iid and tree.exists(self._hover_iid):
            tree.item(self._hover_iid, tags=[t for t in tree.item(self._hover_iid, "tags") or () if t != "hover"])
        if iid:
            tree.item(iid, tags=[*(tree.item(iid, "tags") or ()), "hover"])
        self._hover_iid = iid

    def _select_all(self, _event=None):
        if self._rows:
            self._all_filtered_selected = len(self._filtered) > len(self._rows)
            self.tree.selection_set(list(self._rows))
            self._update_count_label()
        return "break"

    def _selected_records(self) -> list[ScanRecord]:
        if self._all_filtered_selected:
            return list(self._filtered)
        return [self._rows[iid] for iid in self.tree.selection() if iid in self._rows]

    def _delete_selected(self, _event=None):
        records = self._selected_records()
        if not records:
            messagebox.showinfo("Löschen", "Bitte wähle mindestens einen Scan in der Tabelle aus "
                                           "(Klick, Strg/Umschalt + Klick oder „Alle auswählen“).")
            return
        if not messagebox.askyesno("Scans löschen", f"Möchtest du wirklich {_fmt_int(len(records))} Scan(s) samt "
                                                    "gespeicherten Bildern dauerhaft aus den Log-Dateien löschen?"):
            return
        self.status_label.configure(text=f"Lösche {_fmt_int(len(records))} Scans ...", text_color=TXT_MID)
        self._tasks.put(("delete", records))

    def _on_delete(self, result: tuple):
        removed, images = result
        self._all_filtered_selected = False
        self._detail_key = None
        self._request_refresh()
        messagebox.showinfo("Gelöscht", f"{_fmt_int(removed)} Scan(s) und {_fmt_int(images)} Bild(er) gelöscht.")

    def _export(self):
        records = self._selected_records() or list(self._filtered)
        if not records:
            messagebox.showinfo("Export", "Keine Einträge zum Exportieren vorhanden.")
            return
        path = filedialog.asksaveasfilename(title="Scan-Logs exportieren", defaultextension=".csv",
                                            filetypes=[("CSV Datei", "*.csv"), ("JSON Datei", "*.json")])
        if not path:
            return
        self.status_label.configure(text=f"Exportiere {_fmt_int(len(records))} Einträge ...", text_color=TXT_MID)
        self._tasks.put(("export", (records, path)))

    def _on_export(self, result: tuple):
        count, path = result
        messagebox.showinfo("Export erfolgreich", f"{_fmt_int(count)} Datensätze exportiert nach:\n{path}")

    # ------------------------------------------------------------------ #
    #  Detailansicht                                                       #
    # ------------------------------------------------------------------ #
    def _set_detail(self, key: str, text: str, color: str = TXT_DARK):
        self.detail_labels[key].configure(text=text, text_color=color)

    def _show_details(self, record: ScanRecord):
        self._detail_key = record.key
        self._detail_request += 1
        if not self.detail_panel.winfo_ismapped():
            self.detail_panel.grid()
        labels = self._view.labels if self._view else {}
        when = datetime.fromtimestamp(record.epoch).strftime("%d.%m.%Y  %H:%M:%S") if record.epoch else "—"
        self._set_detail("Scan-ID", record.scan_id or "—")
        self._set_detail("Zeitstempel", when)
        self._set_detail("Kamera", labels.get(record.source, record.source))
        self._set_detail("Endergebnis", record.code or "Fehlgeschlagen", SUCCESS if record.success else DANGER)
        self._set_detail("Methode", record.method)
        self._set_detail("Konfidenz", _fmt_pct(record.confidence * 100))
        self._set_detail("Dauer", f"{_fmt_int(record.total_ms)} ms")
        self._set_detail("Flags", ", ".join(record.flags) or "Keine")
        for key in ("DMTX", "OCR", "YOLO Conf", "Trigger", "Kamera-Modell", "Belichtung/Gain", "Version"):
            self._set_detail(key, "…", TXT_LIGHT)
        self.image_container.grid_remove()
        scaling = ctk.ScalingTracker.get_widget_scaling(self)
        size = (int(DETAIL_IMAGE_SIZE[0] * scaling), int(DETAIL_IMAGE_SIZE[1] * scaling))
        log_dir = self._view.log_dir if self._view else ""
        threading.Thread(target=self._load_details, args=(self._detail_request, record, log_dir, size),
                         daemon=True).start()

    def _load_details(self, request: int, record: ScanRecord, log_dir: str, size: tuple):
        """Hintergrund: vollständigen Datensatz und verkleinertes Bild laden."""
        data = LogStore.read_full(record)
        image = image_path = None
        relative = ((data or {}).get("image") or {}).get("path")
        if relative:
            for candidate in (os.path.join(os.path.dirname(record.path), relative), os.path.join(log_dir, relative)):
                if os.path.isfile(candidate):
                    image_path = candidate
                    break
        if image_path:
            try:
                with Image.open(image_path) as source:
                    source.draft("RGB", (size[0] * 2, size[1] * 2))  # JPEG direkt verkleinert dekodieren
                    image = source.convert("RGB")
                image.thumbnail(size)
            except (OSError, ValueError):
                image = None
        self._results.put(("detail", (request, data, image, image_path)))

    def _on_detail(self, payload: tuple):
        request, data, image, image_path = payload
        if request != self._detail_request:
            return
        data = data or {}
        dmtx, ocr = data.get("dmtx") or {}, data.get("ocr") or {}
        timing, meta = data.get("timing") or {}, data.get("meta") or {}
        yolo = (data.get("detection") or {}).get("yolo_conf")
        missing = "—" if data else "Datensatz nicht lesbar"
        self._set_detail("DMTX", dmtx.get("text") or ("Nicht lesbar" if data else missing))
        self._set_detail("OCR", ocr.get("text") or ocr.get("partial") or ("Nicht lesbar" if data else missing))
        self._set_detail("YOLO Conf", _fmt_pct(float(yolo) * 100) if yolo else missing)
        if timing:
            self._set_detail("Dauer", f"{_fmt_int(timing.get('total_ms') or 0)} ms\n(YOLO: {timing.get('yolo_ms', 0)} ms, "
                                      f"OCR: {timing.get('ocr_ms', 0)} ms, DMX: {timing.get('dmtx_ms', 0)} ms)")
        trigger, port = meta.get("trigger"), meta.get("port")
        self._set_detail("Trigger", f"{trigger or '—'}  (Port {port})" if port else (trigger or missing))
        model, serial = meta.get("camera_model"), meta.get("camera_serial")
        self._set_detail("Kamera-Modell", f"{model or '—'}\nS/N {serial}" if serial else (model or missing))
        exposure, gain = meta.get("exposure_us"), meta.get("gain")
        self._set_detail("Belichtung/Gain", f"Exp: {float(exposure) / 1000:.1f} ms  |  Gain: {float(gain):.2f}"
                         if exposure and gain is not None else missing)
        version = meta.get("app_version")
        auto_exposure = "  ·  Auto-Belichtung" if meta.get("auto_exposure") else ""
        self._set_detail("Version", f"v{version}{auto_exposure}" if version else missing)

        self._detail_image_path = image_path
        if image is None:
            self.image_container.grid_remove()
            return
        scaling = ctk.ScalingTracker.get_widget_scaling(self)
        self._detail_image = ctk.CTkImage(light_image=image, dark_image=image,
                                          size=(int(image.width / scaling), int(image.height / scaling)))
        self.image_label.configure(image=self._detail_image)
        self.image_container.grid(row=1, column=0, padx=16, pady=4, sticky="ew")

    def _open_detail_image(self, _event=None):
        if self._detail_image_path and os.path.isfile(self._detail_image_path):
            os.startfile(self._detail_image_path)


if __name__ == "__main__":
    app = LogAnalyzerApp()
    app.mainloop()
