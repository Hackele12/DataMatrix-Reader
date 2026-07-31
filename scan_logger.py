"""
scan_logger.py — Strukturiertes Scan-Logging für DataDetector

Schreibt für jeden Scan-Vorgang einen kompakten, maschinenlesbaren JSON-Lines Record
in eine .jsonl-Datei und speichert das zugehörige Kamerabild. Designed für nachgelagerte
Auswertung durch externe Programme.

Features:
- JSON-Lines Format (.jsonl): Ein JSON-Record pro Zeile, streambar und verarbeitbar
- Qualitäts-Grading (A/B/C/D) mit diagnostischen Flags
- Automatische Bildspeicherung für jeden Scan
- Festplattenschutz: Bei >70% Laufwerksauslastung wird NICHTS mehr gespeichert
- Session-Statistiken (im RAM + Zusammenfassung bei Shutdown)
- Log-Rotation: Neue Datei pro Tag, alte Dateien nach Limit gelöscht
"""

import json
import logging
import os
import shutil
import time
import threading
from datetime import datetime, timezone

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Konstanten & Hilfsfunktionen                                                #
# --------------------------------------------------------------------------- #
_DISK_USAGE_LIMIT = 0.70          # 70% Auslastung → Logging stoppen
_DISK_CHECK_INTERVAL_S = 30.0     # Alle 30s erneut prüfen
_MAX_LOG_FILES = 30               # Maximal 30 Tages-Log-Dateien behalten
_IMAGE_JPEG_QUALITY = 85          # JPEG-Qualität für gespeicherte Bilder


def resolve_log_directory(preferred_path: str = r"U:\Temp\DataMatrixReader.logFiles") -> str:
    """
    Sucht dynamisch nach dem Log-Verzeichnis:
    1. Wenn das bevorzugte Laufwerk (z.B. U:) existiert, erstelle/nutze den Pfad.
    2. Wenn U: nicht existiert, suche nach r'\\Temp\\DataMatrixReader.logFiles' auf anderen Laufwerken (D:, E:, F:, C:).
    3. Versuche den Ordner auf einem verfügbaren Laufwerk zu erstellen.
    4. Fallback: Lokaler Ordner './scan_logs'.
    """
    if not preferred_path:
        preferred_path = r"U:\Temp\DataMatrixReader.logFiles"
        
    drive_letter = os.path.splitdrive(preferred_path)[0]
    if drive_letter and os.path.exists(drive_letter):
        try:
            os.makedirs(preferred_path, exist_ok=True)
            return preferred_path
        except Exception:
            pass

    rel_subpath = preferred_path[2:] if preferred_path[1:2] == ":" else preferred_path
    rel_subpath = rel_subpath.lstrip("\\/")

    # Suche auf existierenden Laufwerken
    for letter in ["U", "D", "E", "F", "Z", "X", "C"]:
        alt_drive = f"{letter}:\\"
        if os.path.exists(alt_drive):
            candidate = os.path.join(alt_drive, rel_subpath)
            if os.path.exists(candidate):
                return candidate

    # Erstelle auf erstbestem verfügbaren Laufwerk
    for letter in ["D", "E", "F", "Z", "X", "C"]:
        alt_drive = f"{letter}:\\"
        if os.path.exists(alt_drive):
            candidate = os.path.join(alt_drive, rel_subpath)
            try:
                os.makedirs(candidate, exist_ok=True)
                return candidate
            except Exception:
                continue

    local_dir = os.path.abspath("scan_logs")
    os.makedirs(local_dir, exist_ok=True)
    return local_dir


def _now_iso() -> str:
    """Gibt den aktuellen Zeitstempel im ISO-8601-Format mit Zeitzonen-Offset zurück."""
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _make_scan_id(counter: int) -> str:
    """Erzeugt eine kompakte, eindeutige Scan-ID."""
    return f"SCN-{time.strftime('%Y%m%d-%H%M%S')}-{counter:04d}"


def _compute_grade(result: dict, flags: list[str]) -> str:
    """
    Berechnet ein Qualitäts-Grade basierend auf dem Scan-Ergebnis.

    A = Perfekt (verifiziert, hohe Konfidenz)
    B = Gut (Erfolg, gute Konfidenz)
    C = Grenzwertig (Rekonstruiert oder niedrige Konfidenz)
    D = Problematisch (Fehlgeschlagen)
    """
    if not result.get("success", False):
        return "D"

    method = result.get("method", "")
    confidence = result.get("confidence", 0.0)
    verified = result.get("verified", False)

    if verified and confidence >= 0.95:
        return "A"
    if confidence >= 0.70 and method != "Rekonstruiert":
        return "B"
    if result.get("success"):
        return "C"
    return "D"


def _compute_flags(result: dict, detection_info: dict, timing: dict) -> list[str]:
    """Berechnet diagnostische Flags für den Scan."""
    flags = []

    # YOLO / Detektion
    yolo_conf = detection_info.get("yolo_conf", 0.0)
    if not detection_info.get("label_detected", True):
        flags.append("NO_LABEL")
    elif yolo_conf > 0 and yolo_conf < 0.50:
        flags.append("LOW_YOLO_CONF")

    # Ergebnis
    success = result.get("success", False)
    method = result.get("method", "")
    confidence = result.get("confidence", 0.0)

    if not success:
        # DataMatrix-Status prüfen
        dmtx_result = result.get("dmtx_result")
        ocr_result = result.get("ocr_result")
        if dmtx_result is None:
            flags.append("DMTX_BLOCKED")
        if ocr_result is None:
            flags.append("OCR_FAILED")

    if method == "Rekonstruiert":
        flags.append("RECONSTRUCTED")

    if result.get("ocr_partial_display") and not result.get("ocr_result"):
        flags.append("OCR_PARTIAL")

    if confidence > 0 and confidence < 0.50:
        flags.append("OCR_LOW_CONF")

    # Timing: > 12 Sekunden = SLOW_SCAN
    total_ms = timing.get("total_ms", result.get("duration_ms", 0))
    if total_ms > 12000:
        flags.append("SLOW_SCAN")

    return flags


class ScanLogger:
    """
    Strukturierter Scan-Logger für industriellen 24/7-Betrieb.

    Schreibt jeden Scan als JSON-Lines Record und speichert das Kamerabild.
    Prüft die Festplattenauslastung und stoppt bei >70%.
    """

    def __init__(self, log_dir: str = r"U:\Temp\DataMatrixReader.logFiles"):
        """
        Args:
            log_dir: Basisverzeichnis für Logs und Bilder (wird dynamisch aufgelöst).
        """
        self._log_dir = resolve_log_directory(log_dir)
        self._images_dir = os.path.join(self._log_dir, "images")
        self._lock = threading.Lock()
        self._scan_counter = 0

        # Session-ID
        self._session_id = f"SES-{time.strftime('%Y%m%d-%H%M%S')}"

        # Session-Statistiken
        self._stats = {
            "session_id": self._session_id,
            "start_time": _now_iso(),
            "total_scans": 0,
            "successful": 0,
            "failed": 0,
            "grade_distribution": {"A": 0, "B": 0, "C": 0, "D": 0},
            "total_duration_ms": 0,
            "methods": {},
            "images_saved": 0,
        }

        # Festplattenschutz: Cache
        self._disk_ok = True
        self._last_disk_check = 0.0

        # Verzeichnisse erstellen
        try:
            os.makedirs(self._log_dir, exist_ok=True)
            os.makedirs(self._images_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"ScanLogger: Verzeichnis erstellen fehlgeschlagen: {e}")

        # Alte Log-Dateien in Haupt-Logdatei scans.jsonl zusammenführen & aufräumen
        self._consolidate_and_rotate_logs()

        logger.info(
            f"ScanLogger initialisiert: log_dir='{log_dir}', "
            f"session='{self._session_id}'"
        )

    # ------------------------------------------------------------------ #
    #  Festplattenschutz                                                   #
    # ------------------------------------------------------------------ #
    def _check_disk_usage(self) -> bool:
        """
        Prüft die Festplattenauslastung. Ergebnis wird 30s gecacht.

        Returns:
            True wenn mindestens 2 GB freier Speicherplatz vorhanden ist, False sonst.
        """
        now = time.time()
        if now - self._last_disk_check < _DISK_CHECK_INTERVAL_S:
            return self._disk_ok

        self._last_disk_check = now
        try:
            path = os.path.abspath(self._log_dir)
            usage = shutil.disk_usage(path)
            free_gb = usage.free / (1024**3)

            if free_gb < 2.0:
                if self._disk_ok:
                    logger.warning(
                        f"ScanLogger: FESTPLATTENSCHUTZ AKTIV! "
                        f"Weniger als 2.0 GB Speicherplatz frei: {free_gb:.1f} GB. "
                        f"Speicherung pausiert bis Platz frei wird."
                    )
                self._disk_ok = False
            else:
                if not self._disk_ok:
                    logger.info(f"ScanLogger: Festplattenschutz aufgehoben ({free_gb:.1f} GB frei).")
                self._disk_ok = True
        except Exception as e:
            logger.warning(f"ScanLogger: Festplattenprüfung fehlgeschlagen: {e}")
            self._disk_ok = True

        return self._disk_ok

    # ------------------------------------------------------------------ #
    #  Log-Rotation & Konsolidierung                                      #
    # ------------------------------------------------------------------ #
    def _consolidate_and_rotate_logs(self):
        """
        Führt alte Tages-Logdateien (scans_*.jsonl) automatisch in die primäre
        Gesamtdatei `scans.jsonl` zusammen und entfernt alte Tages-Dateien.
        """
        try:
            target_single_log = os.path.join(self._log_dir, "scans.jsonl")
            daily_files = sorted(
                [f for f in os.listdir(self._log_dir) if f.startswith("scans_") and f.endswith(".jsonl")]
            )

            if not daily_files:
                return

            records_by_id = {}
            ordered_records = []

            # 1. Vorhandene Records aus scans.jsonl einlesen
            if os.path.exists(target_single_log):
                with open(target_single_log, "r", encoding="utf-8") as f:
                    for line in f:
                        line_str = line.strip()
                        if not line_str:
                            continue
                        try:
                            rec = json.loads(line_str)
                            sid = rec.get("scan_id") or rec.get("ts")
                            if sid and sid not in records_by_id:
                                records_by_id[sid] = line_str
                                ordered_records.append(line_str)
                        except Exception:
                            pass

            # 2. Records aus alten Tages-Dateien einlesen & anhängen (falls noch nicht vorhanden)
            for df in daily_files:
                df_path = os.path.join(self._log_dir, df)
                try:
                    with open(df_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line_str = line.strip()
                            if not line_str:
                                continue
                            try:
                                rec = json.loads(line_str)
                                sid = rec.get("scan_id") or rec.get("ts")
                                if sid and sid not in records_by_id:
                                    records_by_id[sid] = line_str
                                    ordered_records.append(line_str)
                            except Exception:
                                pass
                except Exception as e:
                    logger.warning(f"ScanLogger: Fehler beim Lesen von {df}: {e}")

            # 3. Konsolidierte scans.jsonl schreiben
            with open(target_single_log, "w", encoding="utf-8") as f:
                for l_str in ordered_records:
                    f.write(l_str + "\n")

            # 4. Alte Tages-Logdateien entfernen
            for df in daily_files:
                df_path = os.path.join(self._log_dir, df)
                try:
                    os.remove(df_path)
                    logger.info(f"ScanLogger: Tagesdatei in scans.jsonl konsolidiert & entfernt: {df}")
                except Exception as e:
                    logger.warning(f"ScanLogger: Fehler beim Löschen von {df}: {e}")

        except Exception as e:
            logger.error(f"ScanLogger: Log-Konsolidierungsfehler: {e}")

    def _get_log_filepath(self) -> str:
        """Gibt den Pfad zur primären Gesamt-Logdatei (scans.jsonl) zurück."""
        return os.path.join(self._log_dir, "scans.jsonl")

    def _get_image_dir(self) -> str:
        """Gibt den Pfad zum Bild-Ordner zurück."""
        os.makedirs(self._images_dir, exist_ok=True)
        return self._images_dir

    # ------------------------------------------------------------------ #
    #  Bild speichern                                                      #
    # ------------------------------------------------------------------ #
    def _save_image(self, frame: np.ndarray, scan_id: str) -> str | None:
        """
        Speichert das Kamerabild als JPEG.

        Args:
            frame: Das Kamerabild (BGR).
            scan_id: Die Scan-ID für den Dateinamen.

        Returns:
            Relativer Pfad zum gespeicherten Bild oder None bei Fehler.
        """
        try:
            img_dir = self._get_image_dir()
            filename = f"{scan_id}.jpg"
            filepath = os.path.join(img_dir, filename)

            cv2.imwrite(
                filepath, frame,
                [cv2.IMWRITE_JPEG_QUALITY, _IMAGE_JPEG_QUALITY]
            )
            self._stats["images_saved"] += 1

            # Relativen Pfad zurückgeben (ab log_dir)
            return os.path.relpath(filepath, self._log_dir)
        except Exception as e:
            logger.error(f"ScanLogger: Bild speichern fehlgeschlagen: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Haupt-Log-Methode                                                   #
    # ------------------------------------------------------------------ #
    def log_scan(
        self,
        scan_result: dict,
        frame: np.ndarray | None,
        timing: dict | None = None,
        detection_info: dict | None = None,
        meta: dict | None = None,
    ) -> dict | None:
        """
        Loggt einen Scan-Vorgang als JSONL-Record und speichert das Bild.

        Args:
            scan_result: Das Ergebnis von scanner.scan() bzw. _update_result().
            frame: Das Kamerabild (BGR numpy array). None = kein Bild speichern.
            timing: {"total_ms": int, "yolo_ms": int, "scan_ms": int}
            detection_info: {"yolo_conf": float, "crop_size": [w, h], "label_detected": bool}
            meta: {"camera_model": str, "camera_serial": str, "exposure_us": float,
                    "gain": float, "app_version": str, "port": int}

        Returns:
            Der geschriebene Record als dict, oder None bei Festplattenschutz.
        """
        # Festplattenschutz
        if not self._check_disk_usage():
            return None

        with self._lock:
            self._scan_counter += 1
            scan_id = _make_scan_id(self._scan_counter)

        if timing is None:
            timing = {}
        if detection_info is None:
            detection_info = {}
        if meta is None:
            meta = {}

        # Flags und Grade berechnen
        flags = _compute_flags(scan_result, detection_info, timing)
        grade = _compute_grade(scan_result, flags)

        # Bild speichern Logik: nur bei Fehlgeschlagen oder wenn der Scan >12s gedauert hat
        image_path = None
        resolution = None
        success = scan_result.get("success", False)
        is_slow = "SLOW_SCAN" in flags
        should_save_image = (not success) or is_slow

        if frame is not None and should_save_image:
            h, w = frame.shape[:2]
            resolution = [w, h]
            image_path = self._save_image(frame, scan_id)
            if image_path:
                flags.append("IMAGE_SAVED")

        # Internes Timing aus scanner.py extrahieren
        internal_timing = scan_result.get("_internal_timing", {})

        # JSONL-Record zusammenbauen
        record = {
            "ts": _now_iso(),
            "scan_id": scan_id,
            "session_id": self._session_id,
            "result": {
                "success": scan_result.get("success", False),
                "code": scan_result.get("result"),
                "method": scan_result.get("method"),
                "confidence": round(scan_result.get("confidence", 0.0), 4),
                "verified": scan_result.get("verified", False),
            },
            "timing": {
                "total_ms": timing.get("total_ms", scan_result.get("duration_ms", 0)),
                "yolo_ms": timing.get("yolo_ms", 0),
                "scan_ms": timing.get("scan_ms", 0),
                "ocr_ms": internal_timing.get("ocr_ms", 0),
                "dmtx_ms": internal_timing.get("dmtx_ms", 0),
            },
            "dmtx": {
                "text": scan_result.get("dmtx_result"),
            },
            "ocr": {
                "text": scan_result.get("ocr_result"),
                "partial": scan_result.get("ocr_partial_display"),
            },
            "detection": {
                "yolo_conf": round(detection_info.get("yolo_conf", 0.0), 4),
                "crop_size": detection_info.get("crop_size"),
                "label_detected": detection_info.get("label_detected", True),
            },
            "image": {
                "resolution": resolution,
                "path": image_path,
            },
            "quality": {
                "grade": grade,
                "flags": flags,
            },
            "meta": meta,
        }

        # In JSONL-Datei schreiben
        try:
            log_path = self._get_log_filepath()
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        except Exception as e:
            logger.error(f"ScanLogger: JSONL schreiben fehlgeschlagen: {e}")

        # Session-Statistiken aktualisieren
        self._update_stats(scan_result, grade, timing)

        return record

    # ------------------------------------------------------------------ #
    #  Session-Statistiken                                                 #
    # ------------------------------------------------------------------ #
    def _update_stats(self, scan_result: dict, grade: str, timing: dict):
        """Aktualisiert die Session-Statistiken im RAM."""
        self._stats["total_scans"] += 1

        if scan_result.get("success", False):
            self._stats["successful"] += 1
        else:
            self._stats["failed"] += 1

        self._stats["grade_distribution"][grade] = (
            self._stats["grade_distribution"].get(grade, 0) + 1
        )

        duration = timing.get("total_ms", scan_result.get("duration_ms", 0))
        self._stats["total_duration_ms"] += duration

        method = scan_result.get("method", "Unbekannt")
        self._stats["methods"][method] = self._stats["methods"].get(method, 0) + 1

    def get_session_stats(self) -> dict:
        """
        Gibt die aktuellen Session-Statistiken zurück.

        Returns:
            dict mit allen Session-Metriken.
        """
        stats = self._stats.copy()
        total = stats["total_scans"]
        if total > 0:
            stats["success_rate"] = round(stats["successful"] / total, 4)
            stats["avg_scan_ms"] = round(stats["total_duration_ms"] / total, 1)
        else:
            stats["success_rate"] = 0.0
            stats["avg_scan_ms"] = 0.0
        stats["end_time"] = _now_iso()
        return stats

    def save_session_summary(self):
        """
        Schreibt eine Session-Zusammenfassung als JSON-Datei.
        Wird typischerweise beim App-Shutdown aufgerufen.
        """
        if not self._check_disk_usage():
            logger.warning("ScanLogger: Session-Summary nicht gespeichert (Festplattenschutz).")
            return

        stats = self.get_session_stats()

        try:
            summary_path = os.path.join(
                self._log_dir, f"session_{self._session_id}.json"
            )
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            logger.info(
                f"ScanLogger: Session-Summary gespeichert: {summary_path} "
                f"(Scans: {stats['total_scans']}, "
                f"Erfolg: {stats.get('success_rate', 0):.1%})"
            )
        except Exception as e:
            logger.error(f"ScanLogger: Session-Summary fehlgeschlagen: {e}")

        # Auch als letzten JSONL-Record loggen
        try:
            log_path = self._get_log_filepath()
            session_record = {
                "ts": _now_iso(),
                "type": "SESSION_END",
                "session_id": self._session_id,
                "stats": stats,
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(session_record, ensure_ascii=False, separators=(",", ":")) + "\n")
        except Exception:
            pass
