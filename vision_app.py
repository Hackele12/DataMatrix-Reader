"""
vision_app.py — DataMatrixReader: Live-Ansicht, Scan und TCP-Trigger für eine oder mehrere IDS-Kameras.

Jede Kamera hat einen eigenen Reiter. „Start Stream" verbindet die gewählte Kamera und öffnet den TCP-Port
des Reiters (Kamera 1: 9500, Kamera 2: 9501, ...). Protokoll: "+" → STX <Code> CR LF EOT.
Streams, die beim Beenden oder bei einem Absturz liefen, starten beim nächsten Programmstart automatisch.
Start über Start_DataDetector.bat (launcher.py); dessen Watchdog startet die App nach einem Absturz neu.
"""

import os
import sys
import json
import ctypes
import logging
import threading
import time
import socket
import struct
import tkinter as tk
from logging.handlers import RotatingFileHandler
from tkinter import messagebox
import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import numpy as np

# --- IDS peak SDK ---
try:
    from ids_peak import ids_peak
    from ids_peak_ipl import ids_peak_ipl
    IDS_AVAILABLE = True
except ImportError:
    IDS_AVAILABLE = False

# --- Windows Crash-Dialog unterdruecken (24/7 Betrieb) ---
# Verhindert "Python funktioniert nicht mehr" Popups
try:
    SEM_NOGPFAULTERRORBOX = 0x0002
    SEM_FAILCRITICALERRORS = 0x0001
    SEM_NOOPENFILEERRORBOX = 0x8000
    ctypes.windll.kernel32.SetErrorMode(
        SEM_NOGPFAULTERRORBOX | SEM_FAILCRITICALERRORS | SEM_NOOPENFILEERRORBOX
    )
except Exception:
    pass

# --- Logging Setup (RotatingFileHandler für 24/7 Betrieb) (W1) ---
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Rotating File Handler erstellen: 5MB maximal, 3 Backups
file_handler = RotatingFileHandler(
    'app_debug.log',
    maxBytes=5 * 1024 * 1024,
    backupCount=3,
    encoding='utf-8'
)
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

# Stream Handler für Konsolenausgabe
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)
logger.info("========== APP START ==========")

# --- Globaler Crash-Handler für Unhandled Exceptions (W2) ---
def global_crash_handler(exctype, value, traceback_obj):
    import traceback
    try:
        # Fehler in crash.log schreiben
        with open("crash.log", "a", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write(f"CRASH OCCURRED: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n")
            traceback.print_exception(exctype, value, traceback_obj, file=f)
            f.write("\n")
    except Exception:
        pass
    
    # In normales Log schreiben
    logger.critical("Unbehandelter Ausnahmefehler aufgetreten!", exc_info=(exctype, value, traceback_obj))
    
    # Standard-Verhalten aufrufen
    sys.__excepthook__(exctype, value, traceback_obj)

sys.excepthook = global_crash_handler

# Auch Thread-Exceptions abfangen
def thread_crash_handler(args):
    global_crash_handler(args.exc_type, args.exc_value, args.exc_traceback)

threading.excepthook = thread_crash_handler

# --- Kritische Umgebungsvariablen VOR allen anderen Imports ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- Scanner, Scan-Logger und Horden-DB laden (YOLO/ultralytics erst beim Stream-Start) ---
try:
    import scanner
    import horde_db
    import yolo_detector
    from scan_logger import ScanLogger, resolve_log_directory
    logger.info("Scanner Modul geladen.")
except Exception as e:
    logger.error(f"Scanner Importfehler: {e}")
    sys.exit(1)

# --- App-Version ---
APP_VERSION = "21.0"
CONFIG_FILE = "config.json"
DEFAULT_LOG_DIR = r"U:\Temp\DataMatrixReader.logFiles"
DEFAULT_TCP_PORT = 9500          # Kamera 1; jede weitere Kamera einen Port höher
DEFAULT_CAMERA_SLOTS = 2
AUTOSTART_ATTEMPTS = 10          # GigE-Kameras sind nach einem Absturz erst nach dem Heartbeat-Timeout wieder frei
RETRY_DELAY_S = 5.0
TRAINING_DIR = "training_data"
AUTO_TRAIN_DIR = "auto_training_data"
AUTO_TRAIN_MAX = 1000            # Maximale Anzahl Auto-Training-Bilder (Festplattenschutz)
SINGLE_INSTANCE_MUTEX = "Local\\DataDetector_DataMatrixReader"
EXTRA_SCAN_FRAMES = 2            # weitere Kamerabilder, wenn das erste keinen Reed-Solomon-bestätigten Code liefert
NEXT_FRAME_TIMEOUT_S = 0.4       # Wartezeit auf ein neues Kamerabild

# --- UI Styling (Light Theme) ---
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

ACCENT   = "#2563EB"
SUCCESS  = "#16A34A"
WARN     = "#D97706"
DANGER   = "#DC2626"
BG_SIDE  = "#F1F5F9"
BG_CARD  = "#FFFFFF"
BG_MAIN  = "#E2E8F0"
BG_INPUT = "#FFFFFF"
TXT_DARK = "#1E293B"
TXT_MID  = "#475569"
TXT_LIGHT= "#94A3B8"
BORDER   = "#CBD5E1"


def _load_config() -> dict:
    """Laedt die gespeicherte Konfiguration."""
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_config(cfg: dict):
    """Speichert die Konfiguration."""
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(cfg, f, indent=4)
    except Exception as e:
        logger.error(f"Config speichern fehlgeschlagen: {e}")


def _count(seq) -> int:
    """Länge einer IDS-peak-Liste (Vektor mit size() oder Python-Sequenz)."""
    return seq.size() if hasattr(seq, 'size') else len(seq)


def _ip_to_int(ip: str) -> int:
    """IPv4-Adresse als 32-Bit-Integer (GigE Vision Standard)."""
    return struct.unpack("!I", socket.inet_aton(ip))[0]


def _register_unicast_ip(dm, camera_ip: str) -> None:
    """Registriert die Kamera-IP für die GigE-Unicast-Suche auf allen Netzwerk-Interfaces (ohne USB/WLAN)."""
    ip_int = _ip_to_int(camera_ip)
    systems = dm.Systems()
    for sys_idx in range(_count(systems)):
        system = systems[sys_idx]
        sys_name = system.DisplayName()
        if "U3V" in sys_name or "USB" in sys_name:
            continue
        interfaces = system.Interfaces()
        for if_idx in range(_count(interfaces)):
            iface_desc = interfaces[if_idx]
            iface_name = iface_desc.DisplayName()
            if "Wi-Fi" in iface_name:
                continue
            try:
                nodemaps = iface_desc.OpenedInterface().NodeMaps()
                if _count(nodemaps) > 0:
                    nodemap = nodemaps[0]
                    if nodemap.HasNode("GevDiscoveryUnicastIPAddressToAdd"):
                        nodemap.FindNode("GevDiscoveryUnicastIPAddressToAdd").SetValue(ip_int)
                        nodemap.FindNode("GevDiscoveryUnicastIPAddressAdd").Execute()
                        logger.info(f"Unicast-IP {camera_ip} für Interface '{iface_name}' registriert.")
            except Exception as e_iface:
                logger.warning(f"Unicast-Setup auf '{iface_name}' fehlgeschlagen: {e_iface}")


# Kamerasuche und Öffnen aller Reiter nacheinander (DeviceManager ist nicht für parallele Updates gedacht)
_IDS_LOCK = threading.Lock()


def _configured_camera_ips(cfg: dict) -> list[str]:
    """Kamera-IPs aus config.json (global und pro Kamera) für die Unicast-Suche."""
    ips = [cfg.get("camera_ip")] + [c.get("camera_ip") for c in cfg.get("cameras") or [] if isinstance(c, dict)]
    return list(dict.fromkeys(ip for ip in ips if ip))


def _register_camera_ips(dm, camera_ips: list[str]) -> None:
    """Unicast-Suche für konfigurierte Kamera-IPs (z.B. Kameras in einem anderen Subnetz)."""
    if not camera_ips:
        return
    dm.Update()  # Erste Update-Runde, damit Interfaces geöffnet werden
    for camera_ip in camera_ips:
        try:
            _register_unicast_ip(dm, camera_ip)
        except Exception as e:
            logger.warning(f"Unicast-Konfiguration für {camera_ip} fehlgeschlagen: {e}")


class IDSFrameGrabber:
    """
    IDS peak SDK Frame-Grabber:
    Läuft in einem eigenen Thread und holt Bilder direkt von der IDS GigE-Kamera.
    Konvertiert Monochrom-Rohdaten automatisch zu BGR8 für OpenCV-Kompatibilität.
    Die IDS-Bibliothek wird einmal pro Programm initialisiert (main).
    """
    def __init__(self):
        self.frame = None
        self.running = False
        self._lock = threading.Lock()
        self._thread = None
        self._device = None
        self._ds = None
        self._nodemap = None
        self._buffers = []
        self.model_name = ""
        self.serial = ""
        self.last_frame_time = time.time()  # Zeitstempel des letzten erfolgreichen Frames

        # --- Auto-Exposure Regelschleife (immer aktiv) ---
        self.auto_exposure_enabled = True
        self.auto_exposure_target = 130       # Ziel-Helligkeit (0-255)
        self.auto_exposure_deadzone = 5        # Hysterese ±5
        self.auto_exposure_min_us = 1000.0     # Min. Belichtung in µs (1ms)
        self.auto_exposure_max_us = 50000.0    # Max. Belichtung in µs (50ms)
        self.auto_exposure_max_gain = 12.0     # Max. Gain
        self._ae_last_time = 0.0               # Letzte Regelung (Throttle)
        self._current_brightness = 0           # Aktuelle Helligkeit (für UI)

    @staticmethod
    def list_cameras(camera_ips: list[str]) -> list[dict]:
        """
        Listet alle im System verfügbaren IDS-Kameras auf.
        
        Returns:
            list[dict]: Liste von Kamera-Dicts mit 'serial', 'model', 'display_name'.
        """
        cameras = []
        if not IDS_AVAILABLE:
            return cameras
        try:
            with _IDS_LOCK:
                dm = ids_peak.DeviceManager.Instance()
                _register_camera_ips(dm, camera_ips)
                dm.Update()
                devices = dm.Devices()
                for idx in range(_count(devices)):
                    desc = devices[idx]
                    model = desc.ModelName()
                    serial = desc.SerialNumber()
                    cameras.append({
                        "serial": serial,
                        "model": model,
                        "display_name": f"{model} ({serial})",
                        "index": idx
                    })
        except Exception as e:
            logger.warning(f"Fehler bei Kamera-Auflistung: {e}")
        return cameras

    def start(self, target_serial: str | None = None, camera_ips: list[str] | None = None) -> bool:
        """Kamera öffnen (per Seriennummer, ohne Vorgabe die erste freie), Buffer anlegen und Aufnahme starten."""
        if not IDS_AVAILABLE:
            logger.error("IDS peak SDK nicht installiert!")
            return False
        try:
            with _IDS_LOCK:
                dm = ids_peak.DeviceManager.Instance()
                _register_camera_ips(dm, camera_ips or [])
                dm.Update()
                devices = dm.Devices()
                dev_count = _count(devices)
                if dev_count == 0:
                    logger.error("Keine IDS-Kamera gefunden!")
                    return False

                desc = None
                if target_serial:
                    # Kein Ausweichen auf eine andere Kamera: sie könnte zu einem anderen Reiter/Port gehören
                    for idx in range(dev_count):
                        if devices[idx].SerialNumber() == target_serial:
                            desc = devices[idx]
                            logger.info(f"Ziel-Kamera gewählt: {desc.ModelName()} (S/N: {target_serial})")
                            break
                    if desc is None:
                        logger.error(f"Kamera S/N {target_serial} nicht gefunden!")
                        return False
                else:
                    for idx in range(dev_count):
                        if devices[idx].IsOpenable():
                            desc = devices[idx]
                            logger.info(f"Erste freie Kamera gewählt: {desc.ModelName()} (S/N: {desc.SerialNumber()})")
                            break
                    if desc is None:
                        logger.error("Keine freie IDS-Kamera gefunden!")
                        return False

                self.model_name = desc.ModelName()
                self.serial = desc.SerialNumber()

                self._device = desc.OpenDevice(ids_peak.DeviceAccessType_Control)
                self._nodemap = self._device.RemoteDevice().NodeMaps()[0]
                self._ds = self._device.DataStreams()[0].OpenDataStream()

                payload = self._nodemap.FindNode("PayloadSize").Value()
                buf_count = max(self._ds.NumBuffersAnnouncedMinRequired(), 3)
                for _ in range(buf_count):
                    buf = self._ds.AllocAndAnnounceBuffer(payload)
                    self._ds.QueueBuffer(buf)
                    self._buffers.append(buf)

                if self._nodemap.HasNode("TLParamsLocked"):
                    self._nodemap.FindNode("TLParamsLocked").SetValue(True)
                self._ds.StartAcquisition()
                self._nodemap.FindNode("AcquisitionStart").Execute()

            self.running = True
            self._thread = threading.Thread(target=self._grab_loop, daemon=True)
            self._thread.start()
            return True
        except Exception as e:
            logger.error(f"IDS Kamera Start fehlgeschlagen: {e}")
            return False

    def _grab_loop(self):
        while self.running:
            try:
                buffer = self._ds.WaitForFinishedBuffer(5000)
                try:
                    ipl_img = ids_peak_ipl.Image.CreateFromSizeAndBuffer(
                        buffer.PixelFormat(), buffer.BasePtr(),
                        buffer.Size(), buffer.Width(), buffer.Height()
                    )
                    converted = ipl_img.ConvertTo(ids_peak_ipl.PixelFormatName_BGR8)
                    frame = converted.get_numpy_3D().copy()
                    with self._lock:
                        self.frame = frame
                        self.last_frame_time = time.time()

                    # --- Auto-Exposure: Helligkeit messen und nachregeln ---
                    if self.auto_exposure_enabled:
                        self._auto_exposure_step(frame)
                finally:
                    # KRITISCH: Buffer IMMER zurückgeben, auch bei Konvertierungsfehler!
                    # Sonst gehen nach wenigen Fehlern alle Buffer verloren und der
                    # Kamera-Stream blockiert permanent.
                    try:
                        self._ds.QueueBuffer(buffer)
                    except Exception:
                        pass
            except Exception:
                if self.running:
                    time.sleep(0.01)

    def get_frame(self):
        with self._lock:
            return self.frame.copy() if self.frame is not None else None

    def _auto_exposure_step(self, frame: np.ndarray):
        """
        Software Auto-Exposure Regelschleife (Fast-Reactive).
        Misst die mittlere Bildhelligkeit und passt Belichtungszeit und Gain
        automatisch an, um die Zielhelligkeit zu halten.
        Priorität: Belichtung zuerst (weniger Rauschen), Gain nur als Backup.
        Throttle: Maximal alle 40ms (25x pro Sekunde / jedes Frame).
        """
        now = time.time()
        if (now - self._ae_last_time) < 0.04:
            return  # Throttle: max 25x pro Sekunde
        self._ae_last_time = now

        try:
            # Helligkeit messen (Graustufen-Mittelwert)
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            mean_brightness = float(np.mean(gray))
            self._current_brightness = int(mean_brightness)

            error = self.auto_exposure_target - mean_brightness

            # Innerhalb der Totzone (±5) → keine Anpassung nötig
            if abs(error) <= self.auto_exposure_deadzone:
                return

            # Schneller Anpassungsfaktor (proportional, max ±40% pro Schritt)
            adjustment = 1.0 + (error / 255.0) * 0.9
            adjustment = max(0.60, min(1.40, adjustment))

            current_exp = getattr(self, 'exposure_us', 6000.0)
            current_gain = getattr(self, 'gain', 1.0)
            min_gain = 1.0

            if error > 0:
                # --- Bild zu dunkel: Aufhellen ---
                new_exp = current_exp * adjustment
                if new_exp <= self.auto_exposure_max_us:
                    self.set_exposure(new_exp)
                else:
                    if current_exp < self.auto_exposure_max_us:
                        self.set_exposure(self.auto_exposure_max_us)
                    new_gain = current_gain * adjustment
                    new_gain = min(new_gain, self.auto_exposure_max_gain)
                    self.set_gain(new_gain)
            else:
                # --- Bild zu hell: Abdunkeln ---
                if current_gain > min_gain + 0.05:
                    new_gain = current_gain * adjustment
                    new_gain = max(new_gain, min_gain)
                    self.set_gain(new_gain)
                else:
                    new_exp = current_exp * adjustment
                    new_exp = max(new_exp, self.auto_exposure_min_us)
                    self.set_exposure(new_exp)

        except Exception as e:
            logger.debug(f"Auto-Exposure Fehler: {e}")

    def set_exposure(self, exposure_us: float):
        """Belichtungszeit in Mikrosekunden setzen (live)."""
        try:
            node = self._nodemap.FindNode("ExposureTime")
            val = min(max(exposure_us, node.Minimum()), node.Maximum())
            node.SetValue(val)
            self.exposure_us = val
            logger.info(f"Belichtungszeit gesetzt: {val:.0f} us")
        except Exception as e:
            self.exposure_us = exposure_us
            logger.warning(f"Belichtungszeit konnte nicht gesetzt werden: {e}")

    def set_gain(self, gain: float):
        """Gain (Verstärkung) setzen (live)."""
        try:
            node = self._nodemap.FindNode("Gain")
            val = min(max(gain, node.Minimum()), node.Maximum())
            node.SetValue(val)
            self.gain = val
            logger.info(f"Gain gesetzt: {val:.2f}")
        except Exception as e:
            self.gain = gain
            logger.warning(f"Gain konnte nicht gesetzt werden: {e}")

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=2)
        try:
            self._nodemap.FindNode("AcquisitionStop").Execute()
        except Exception:
            pass
        try:
            self._ds.StopAcquisition(ids_peak.AcquisitionStopMode_Default)
        except Exception:
            pass
        try:
            if self._nodemap.HasNode("TLParamsLocked"):
                self._nodemap.FindNode("TLParamsLocked").SetValue(False)
        except Exception:
            pass
        try:
            self._ds.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
            for b in self._buffers:
                self._ds.RevokeBuffer(b)
        except Exception:
            pass
        self._buffers.clear()
        try:
            self._ds.Close()
        except Exception:
            pass
        try:
            self._device.Close()
        except Exception:
            pass
        self.frame = None
        logger.info(f"IDS Kamera {self.serial} sauber geschlossen.")


class CameraPanel(ctk.CTkFrame):
    """Reiter einer Kamera: Auswahl, Live-Ansicht und Scan; der TCP-Port ist offen, solange der Stream läuft."""

    def __init__(self, master, app: "DataMatrixReaderApp", slot: dict, tcp_port: int, tab_name: str):
        super().__init__(master, fg_color="transparent", corner_radius=0)
        self.app = app
        self._slot = slot  # Eintrag in config["cameras"]
        self.cam_id = slot["id"]
        self.cam_name = slot["name"]
        self.tcp_port = tcp_port
        self.tab_name = tab_name
        self.visible = False  # nur der sichtbare Reiter zeichnet die Live-Ansicht
        self._closed = False

        self.stream_running = False
        self._connecting = False
        self._stopping = False  # Guard gegen doppelten Stop
        self.active_serial: str | None = None  # Kamera, die gerade verbunden wird oder streamt
        self.grabber: IDSFrameGrabber | None = None
        self.scan_logger: ScanLogger | None = None  # wird beim ersten Stream-Start angelegt
        self._display_thread = None
        self._scan_running = False
        self._last_frame = None
        self._canvas_img_id = None  # Tracking für Canvas-Bild (Speicherleck-Fix)
        self._scan_counter = 0
        self._camera_map = {}  # Map: display_name -> serial

        # --- Smart Auto-Scan (Präsenzerkennung) ---
        self.auto_scan_enabled = bool(self._setting("auto_scan", False))
        self._presence_state = "EMPTY"  # "EMPTY" | "SCANNED"
        self._presence_counter = 0
        self._absence_counter = 0

        # --- YOLO Inferenz Throttling ---
        self._last_infer_time = 0.0
        self._last_detections = None  # Gecachtes Inferenz-Ergebnis

        # --- TCP-Server (nur während der Stream läuft) ---
        self._tcp_server: socket.socket | None = None
        self._tcp_stop: threading.Event | None = None
        self._tcp_scan_token = 0
        self._tcp_scan_lock = threading.Lock()

        self._build_ui()

    # ------------------------------------------------------------------ #
    #  Einstellungen dieser Kamera                                         #
    # ------------------------------------------------------------------ #
    def _setting(self, key: str, default=None):
        """Wert dieser Kamera; ohne eigenen Wert gilt der bisherige globale Wert aus config.json."""
        return self._slot.get(key, self.app.config_data.get(key, default))

    def _store(self, **values):
        self._slot.update(values)
        self.app.save_config()

    @property
    def saved_serial(self) -> str | None:
        return self._slot.get("selected_camera_serial")

    @property
    def autostart_requested(self) -> bool:
        """Stream lief beim letzten Programmende (oder Absturz) und wurde nicht per Stop beendet."""
        return bool(self._slot.get("stream_active", False))

    # ------------------------------------------------------------------ #
    #  UI Builder                                                          #
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # -- Sidebar --
        self.sidebar = ctk.CTkFrame(self, width=230, corner_radius=12, fg_color=BG_SIDE,
                                    border_width=1, border_color=BORDER)
        self.sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 12))
        self.sidebar.grid_propagate(False)
        self.sidebar.grid_rowconfigure(14, weight=1)

        ctk.CTkLabel(
            self.sidebar, text=self.cam_name,
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
            text_color=ACCENT
        ).grid(row=0, column=0, padx=20, pady=(24, 2))

        ctk.CTkLabel(
            self.sidebar, text=f"DataMatrixReader v{APP_VERSION}",
            font=ctk.CTkFont(size=11), text_color=TXT_LIGHT
        ).grid(row=1, column=0, padx=20, pady=(0, 16))

        # -- Kamera-Auswahl --
        ctk.CTkLabel(self.sidebar, text="Kamera-Auswahl:", anchor="w",
                     font=ctk.CTkFont(weight="bold"), text_color=TXT_DARK
        ).grid(row=2, column=0, padx=20, sticky="w")

        self.cam_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.cam_frame.grid(row=3, column=0, padx=20, pady=(2, 10), sticky="ew")
        self.cam_frame.grid_columnconfigure(0, weight=1)

        self.camera_optionmenu = ctk.CTkOptionMenu(
            self.cam_frame, values=["Suche Kameras..."],
            command=self._on_camera_selected,
            height=28, dynamic_resizing=False
        )
        self.camera_optionmenu.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self.cam_refresh_btn = ctk.CTkButton(
            self.cam_frame, text="⟳", width=28, height=28,
            fg_color=TXT_MID, hover_color="#64748B",
            command=self.app.refresh_cameras
        )
        self.cam_refresh_btn.grid(row=0, column=1, sticky="e")

        # -- Helligkeits-Regelung --
        ctk.CTkLabel(self.sidebar, text="Helligkeits-Regelung:", anchor="w",
                     font=ctk.CTkFont(weight="bold"), text_color=TXT_DARK
        ).grid(row=4, column=0, padx=20, sticky="w")

        self.settings_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.settings_frame.grid(row=5, column=0, padx=20, pady=(4, 10), sticky="ew")
        self.settings_frame.grid_columnconfigure(0, weight=1)

        saved_pct = self._setting("auto_exposure_target_pct")
        if saved_pct is None:
            saved_target = self._setting("auto_exposure_target", 130)
            saved_pct = max(0, min(100, int((saved_target - 40) / 180.0 * 100)))

        self.brightness_target_label = ctk.CTkLabel(
            self.settings_frame, text=f"Ziel-Helligkeit: {int(saved_pct)}%",
            font=ctk.CTkFont(size=11, weight="bold"), text_color=TXT_DARK, anchor="w"
        )
        self.brightness_target_label.grid(row=0, column=0, sticky="w", pady=(0, 2))

        self.brightness_slider = ctk.CTkSlider(
            self.settings_frame, from_=0, to=100, number_of_steps=100,
            command=self._on_brightness_slider_changed, width=190
        )
        self.brightness_slider.set(saved_pct)
        self.brightness_slider.grid(row=1, column=0, sticky="ew", pady=(2, 4))

        self.brightness_live_label = ctk.CTkLabel(
            self.settings_frame, text="☀ Live: --/255",
            font=ctk.CTkFont(size=10), text_color=TXT_LIGHT, anchor="w"
        )
        self.brightness_live_label.grid(row=2, column=0, sticky="w", pady=(0, 2))

        # -- Stream Start / Stop Button --
        self.start_btn = ctk.CTkButton(
            self.sidebar, text="▶  Start Stream", width=190,
            fg_color=ACCENT, hover_color="#1D4ED8",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self.toggle_stream
        )
        self.start_btn.grid(row=6, column=0, padx=20, pady=(4, 8))

        ctk.CTkFrame(self.sidebar, height=1, fg_color=BORDER).grid(
            row=7, column=0, padx=20, pady=8, sticky="ew"
        )

        # -- Aktionen --
        ctk.CTkLabel(self.sidebar, text="Aktionen:", anchor="w",
                     font=ctk.CTkFont(weight="bold"), text_color=TXT_DARK
        ).grid(row=8, column=0, padx=20, sticky="w")

        self.scan_btn = ctk.CTkButton(
            self.sidebar, text="◎  SCAN", width=190,
            fg_color=SUCCESS, hover_color="#15803D",
            font=ctk.CTkFont(size=14, weight="bold"),
            state="disabled",
            command=self.trigger_scan
        )
        self.scan_btn.grid(row=9, column=0, padx=20, pady=(4, 6))

        self.train_capture_btn = ctk.CTkButton(
            self.sidebar, text="◉  Capture (Training)", width=190,
            fg_color="#7C3AED", hover_color="#6D28D9",
            font=ctk.CTkFont(size=12, weight="bold"),
            state="disabled",
            command=self.capture_training_image
        )
        self.train_capture_btn.grid(row=10, column=0, padx=20, pady=(0, 6))

        self.auto_scan_switch = ctk.CTkSwitch(
            self.sidebar, text="Auto-Scan (Präsenz)",
            font=ctk.CTkFont(size=12, weight="bold"), text_color=TXT_DARK,
            command=self._on_auto_scan_toggled
        )
        if self.auto_scan_enabled:
            self.auto_scan_switch.select()
        self.auto_scan_switch.grid(row=11, column=0, padx=20, pady=(6, 2), sticky="w")

        self.save_all_scans_switch = ctk.CTkSwitch(
            self.sidebar, text="Bilder aller Codes speich.",
            font=ctk.CTkFont(size=11, weight="bold"), text_color=TXT_DARK,
            command=self._on_save_all_scans_toggled
        )
        if self._setting("save_all_scans", False):
            self.save_all_scans_switch.select()
        self.save_all_scans_switch.grid(row=12, column=0, padx=20, pady=(2, 6), sticky="w")

        self.tcp_info_label = ctk.CTkLabel(
            self.sidebar, text=f"TCP Port {self.tcp_port}: AUS",
            font=ctk.CTkFont(size=11, weight="bold"), text_color=TXT_LIGHT
        )
        self.tcp_info_label.grid(row=13, column=0, padx=20, pady=(4, 0))

        self.status_label = ctk.CTkLabel(
            self.sidebar, text="● Bereit", text_color=TXT_LIGHT,
            font=ctk.CTkFont(size=12)
        )
        self.status_label.grid(row=14, column=0, padx=20, pady=(8, 4), sticky="s")

        # Lade-Animation (indeterminate progress bar)
        self.loading_bar = ctk.CTkProgressBar(
            self.sidebar, width=190, height=6,
            fg_color=BORDER, progress_color=ACCENT, mode="indeterminate"
        )
        self.loading_bar.grid(row=15, column=0, padx=20, pady=(0, 16), sticky="s")
        self.loading_bar.grid_remove()  # Versteckt bis zum Laden

        # -- Hauptbereich --
        self.main_frame = ctk.CTkFrame(self, corner_radius=12, fg_color=BG_CARD,
                                       border_width=1, border_color=BORDER)
        self.main_frame.grid(row=0, column=1, pady=(0, 12), sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.main_frame, bg="#E2E8F0", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        self.fps_label = ctk.CTkLabel(
            self.main_frame, text="FPS: --",
            font=ctk.CTkFont(size=11), text_color=TXT_LIGHT
        )
        self.fps_label.place(relx=1.0, rely=0.0, anchor="ne", x=-12, y=12)

        # -- Ergebnis-Panel --
        self.result_panel = ctk.CTkFrame(self, corner_radius=12, fg_color=BG_CARD,
                                         height=130, border_width=1, border_color=BORDER)
        self.result_panel.grid(row=1, column=1, sticky="ew")
        self.result_panel.grid_columnconfigure(1, weight=1)
        self.result_panel.grid_propagate(False)

        ctk.CTkLabel(
            self.result_panel, text="SCAN ERGEBNIS",
            font=ctk.CTkFont(size=10, weight="bold"), text_color=TXT_LIGHT
        ).grid(row=0, column=0, columnspan=3, padx=16, pady=(10, 2), sticky="w")

        self.method_label = ctk.CTkLabel(
            self.result_panel, text="--",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=ACCENT, width=140, anchor="w"
        )
        self.method_label.grid(row=1, column=0, padx=(16, 8), pady=2, sticky="w")

        self.result_label = ctk.CTkLabel(
            self.result_panel, text="Noch kein Scan durchgefuehrt.",
            font=ctk.CTkFont(family="Consolas", size=16, weight="bold"),
            text_color=TXT_DARK, anchor="w", wraplength=700
        )
        self.result_label.grid(row=1, column=1, padx=8, pady=2, sticky="ew")

        self.conf_label = ctk.CTkLabel(
            self.result_panel, text="",
            font=ctk.CTkFont(size=11), text_color=TXT_LIGHT, anchor="e", width=100
        )
        self.conf_label.grid(row=1, column=2, padx=(8, 16), pady=2, sticky="e")

        self.detail_label = ctk.CTkLabel(
            self.result_panel, text="",
            font=ctk.CTkFont(size=10), text_color=TXT_LIGHT, anchor="w"
        )
        self.detail_label.grid(row=2, column=0, columnspan=2, padx=16, pady=(0, 2), sticky="w")

        self.auto_train_label = ctk.CTkLabel(
            self.result_panel,
            text=f"Auto-Training: {self.app.auto_train_count} Bilder",
            font=ctk.CTkFont(size=10), text_color=TXT_LIGHT, anchor="e"
        )
        self.auto_train_label.grid(row=2, column=2, padx=(8, 16), pady=(0, 2), sticky="e")

    def _on_brightness_slider_changed(self, val: float):
        """Wird aufgerufen wenn der Helligkeits-Schieberegler bewegt wird (0-100%)."""
        pct = int(val)
        target_br = int(40 + (pct / 100.0) * 180)
        self.brightness_target_label.configure(text=f"Ziel-Helligkeit: {pct}%")
        self._store(auto_exposure_target_pct=pct, auto_exposure_target=target_br)

        if self.grabber:
            self.grabber.auto_exposure_target = target_br

    # ------------------------------------------------------------------ #
    #  Kamera-Auswahl                                                      #
    # ------------------------------------------------------------------ #
    def update_camera_list(self, cameras: list[dict], taken: set[str]) -> str | None:
        """
        Aktualisiert das Dropdown: die gespeicherte Kamera bleibt gewählt (auch wenn sie gerade fehlt),
        sonst wird die erste noch keinem Reiter zugeordnete Kamera vorgeschlagen. → angezeigte S/N
        """
        self._camera_map = {cam["display_name"]: cam["serial"] for cam in cameras}
        names = list(self._camera_map)
        saved = self.saved_serial
        shown = next((name for name, serial in self._camera_map.items() if serial == saved), None)
        if shown is None and saved:
            shown = f"S/N {saved} (nicht gefunden)"
            self._camera_map[shown] = saved
            names.insert(0, shown)
        if shown is None:
            shown = next((name for name, serial in self._camera_map.items() if serial not in taken), None)

        self.camera_optionmenu.configure(values=names or ["Keine Kamera gefunden"])
        if shown is not None:
            self.camera_optionmenu.set(shown)
        else:
            self.camera_optionmenu.set("Kamera wählen..." if names else "Keine Kamera gefunden")
        return self._camera_map.get(shown)

    def _on_camera_selected(self, selected_display_name: str):
        serial = self._camera_map.get(selected_display_name)
        if not serial or serial == self.saved_serial:
            return

        logger.info(f"[{self.cam_name}] Kamera gewechselt zu: {selected_display_name} (S/N: {serial})")
        self._store(selected_camera_serial=serial)

        # Falls der Stream gerade läuft, neu starten
        if self.stream_running:
            self._stop_stream()
            self.after(500, self.start_stream)

    def _on_auto_scan_toggled(self):
        val = bool(self.auto_scan_switch.get())
        self.auto_scan_enabled = val
        self._store(auto_scan=val)
        logger.info(f"[{self.cam_name}] Auto-Scan Modus geändert: {val}")

    def _on_save_all_scans_toggled(self):
        val = bool(self.save_all_scans_switch.get())
        self._store(save_all_scans=val)
        if self.scan_logger:
            self.scan_logger.save_all_scans = val
        logger.info(f"[{self.cam_name}] Bilder-Speicher-Modus: "
                    f"{'ALLE Bilder speichern' if val else 'Nur FEHLER-Bilder speichern'}")

    # ------------------------------------------------------------------ #
    #  Stream Steuerung                                                    #
    # ------------------------------------------------------------------ #
    def toggle_stream(self):
        if not self.stream_running:
            self.start_stream()
        else:
            self._store(stream_active=False)  # bewusst gestoppt → kein Autostart beim nächsten Programmstart
            self._stop_stream()

    def start_stream(self, autostart: bool = False):
        if self.stream_running or self._connecting:
            return
        if autostart:
            serial = self.saved_serial
        else:
            serial = self._camera_map.get(self.camera_optionmenu.get()) or self.saved_serial
        if serial is None and self._camera_map:
            self._set_status("Bitte Kamera auswählen.", WARN)
            return
        owner = self.app.panel_using_camera(serial, exclude=self) if serial else None
        if owner is not None:
            self._set_status(f"Kamera läuft bereits in {owner.cam_name}!", DANGER)
            return

        self._connecting = True
        self.active_serial = serial
        # UI in Lade-Zustand versetzen
        self.start_btn.configure(state="disabled", text="...Verbinde")
        self._set_status("Lade Modell & Stream...", ACCENT)
        self.loading_bar.grid()          # Zeige Fortschrittsbalken
        self.loading_bar.start()         # Animation starten

        # Schwere Arbeit im Hintergrund-Thread
        threading.Thread(target=self._start_stream_worker, args=(serial, autostart), daemon=True).start()

    def _start_stream_worker(self, serial: str | None, autostart: bool):
        """Laeuft im Hintergrund-Thread: YOLO laden + Kamera verbinden (Autostart wartet auf die Kamera)."""
        try:
            # 1) YOLO Modell laden (kann 5-20 Sekunden dauern); ultralytics wird erst hier importiert,
            #    damit die GUI sofort startet. Alle Reiter teilen sich das Modell.
            if self.app.model is None:
                self.after(0, lambda: self._set_status("Lade KI-Modell...", ACCENT))
            self.app.ensure_model()

            if self.scan_logger is None:
                # Eigener Log-Unterordner pro Kamera (der Log Analyzer zeigt ihn als Kamera an)
                self.scan_logger = ScanLogger(log_dir=os.path.join(self.app.log_base_dir, self.cam_id),
                                              save_all_scans=bool(self._setting("save_all_scans", False)))

            # 2) Kamera-Stream verbinden
            attempts = AUTOSTART_ATTEMPTS if autostart else 1
            for attempt in range(1, attempts + 1):
                if self._closed:
                    return
                self.after(0, lambda: self._set_status("Verbinde mit Kamera...", WARN))
                grabber = self._open_camera(serial)
                if grabber is not None:
                    self.grabber = grabber
                    self.after(0, self._on_stream_connected)
                    return
                if attempt < attempts:
                    logger.warning(f"[{self.cam_name}] Autostart: Kamera nicht erreichbar "
                                   f"(Versuch {attempt}/{attempts}), neuer Versuch in {RETRY_DELAY_S:.0f} s.")
                    self.after(0, lambda a=attempt: self._set_status(
                        f"Warte auf Kamera... ({a}/{attempts})", WARN))
                    time.sleep(RETRY_DELAY_S)
            self.after(0, self._on_stream_failed)

        except Exception as e:
            logger.error(f"[{self.cam_name}] Stream-Start Fehler: {e}")
            self.after(0, self._on_stream_error, str(e))

    def _open_camera(self, serial: str | None) -> IDSFrameGrabber | None:
        grabber = IDSFrameGrabber()
        if not grabber.start(target_serial=serial, camera_ips=self.app.camera_ips):
            return None
        self._configure_grabber(grabber)
        return grabber

    def _configure_grabber(self, grabber: IDSFrameGrabber):
        """Übernimmt Belichtung, Gain und Auto-Exposure-Parameter aus der Config."""
        cfg = self.app.config_data
        try:
            exp_val = float(cfg.get("last_exposure", 20.0))
            gain_val = float(cfg.get("last_gain", 1.0))
            grabber.set_exposure(exp_val * 1000.0)
            grabber.set_gain(gain_val)
        except Exception:
            pass

        if cfg.get("auto_exposure_enabled", False):
            grabber.auto_exposure_enabled = True
            grabber.auto_exposure_target = int(self._setting("auto_exposure_target", 130))
            grabber.auto_exposure_deadzone = int(cfg.get("auto_exposure_deadzone", 10))
            grabber.auto_exposure_min_us = float(cfg.get("auto_exposure_min_ms", 1.0)) * 1000.0
            grabber.auto_exposure_max_us = float(cfg.get("auto_exposure_max_ms", 50.0)) * 1000.0
            grabber.auto_exposure_max_gain = float(cfg.get("auto_exposure_max_gain", 12.0))
            logger.info(f"[{self.cam_name}] Auto-Exposure aus Config aktiviert.")

    def _on_stream_connected(self):
        """Callback im Main-Thread: Stream erfolgreich verbunden → TCP-Port öffnen."""
        self.loading_bar.stop()
        self.loading_bar.grid_remove()
        self._connecting = False
        self.stream_running = True
        self._stopping = False
        self.active_serial = self.grabber.serial
        self._store(selected_camera_serial=self.grabber.serial, stream_active=True)
        self.start_btn.configure(state="normal", text="■  Stop Stream",
                                 fg_color=DANGER, hover_color="#B91C1C")
        self.scan_btn.configure(state="normal")
        self.train_capture_btn.configure(state="normal")
        self._set_status("● LIVE", SUCCESS)
        self._start_tcp_server()

        self._display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self._display_thread.start()

    def _on_stream_failed(self):
        """Callback im Main-Thread: Stream-Verbindung fehlgeschlagen."""
        self._reset_start_button()
        self._set_status("Stream Verbindung fehlgeschlagen!", DANGER)

    def _on_stream_error(self, msg):
        """Callback im Main-Thread: Allgemeiner Fehler beim Start."""
        self._reset_start_button()
        self._set_status(f"Fehler: {msg}", DANGER)

    def _reset_start_button(self):
        self.loading_bar.stop()
        self.loading_bar.grid_remove()
        self._connecting = False
        self.active_serial = None
        self.start_btn.configure(state="normal", text="▶  Start Stream",
                                 fg_color=ACCENT, hover_color="#1D4ED8")

    def _stop_stream(self):
        # Guard gegen doppelten Aufruf
        if self._stopping:
            return
        self._stopping = True
        self.stream_running = False
        self._stop_tcp_server()
        # Lade-Animation sicher beenden
        try:
            self.loading_bar.stop()
            self.loading_bar.grid_remove()
        except Exception:
            pass
        if self.grabber:
            self.grabber.stop()
            self.grabber = None
        self.active_serial = None
        self.start_btn.configure(
            state="normal", text="▶  Start Stream",
            fg_color=ACCENT, hover_color="#1D4ED8"
        )
        self.scan_btn.configure(state="disabled")
        self.train_capture_btn.configure(state="disabled")
        self._set_status("● Gestoppt", TXT_LIGHT)

    def shutdown(self):
        """Programmende: Stream und TCP-Port schließen, Session-Statistik speichern (Autostart-Merker bleibt)."""
        self._closed = True
        if self.scan_logger is not None:
            self.scan_logger.save_session_summary()
            stats = self.scan_logger.get_session_stats()
            logger.info(
                f"[{self.cam_name}] Session beendet: {stats['total_scans']} Scans, "
                f"Erfolg: {stats.get('success_rate', 0):.1%}"
            )
        self._stop_stream()

    # ------------------------------------------------------------------ #
    #  TCP Server (läuft, solange der Stream läuft)                        #
    # ------------------------------------------------------------------ #
    def _start_tcp_server(self):
        """Öffnet den TCP-Port dieser Kamera für externe Netzwerk-Trigger (z. B. '+')."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind(('0.0.0.0', self.tcp_port))
            server.listen(5)
        except OSError as e:
            server.close()
            logger.error(f"[{self.cam_name}] TCP-Server Start-Fehler auf Port {self.tcp_port}: {e}")
            self._set_tcp_state("FEHLER", DANGER)
            return
        server.settimeout(1.0)  # accept() prüft regelmäßig, ob der Stream gestoppt wurde
        self._tcp_server = server
        self._tcp_stop = threading.Event()
        threading.Thread(target=self._serve_tcp, args=(server, self._tcp_stop), daemon=True).start()
        logger.info(f"[{self.cam_name}] TCP-Server lauscht auf Port {self.tcp_port}")
        self._set_tcp_state("AKTIV", SUCCESS)

    def _stop_tcp_server(self):
        if self._tcp_stop is not None:
            self._tcp_stop.set()
            self._tcp_stop = None
        if self._tcp_server is not None:
            try:
                self._tcp_server.close()
            except OSError:
                pass
            self._tcp_server = None
            logger.info(f"[{self.cam_name}] TCP-Port {self.tcp_port} geschlossen.")
        self._set_tcp_state("AUS", TXT_LIGHT)

    def _set_tcp_state(self, state: str, color: str):
        self.tcp_info_label.configure(text=f"TCP Port {self.tcp_port}: {state}", text_color=color)

    def _serve_tcp(self, server: socket.socket, stop: threading.Event):
        while not stop.is_set():
            try:
                conn, addr = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break  # Socket wurde beim Stoppen geschlossen
            self._handle_tcp_client(conn, addr, stop)
        try:
            server.close()
        except OSError:
            pass

    def _handle_tcp_client(self, conn: socket.socket, addr, stop: threading.Event):
        conn.settimeout(1.0)
        logger.info(f"[{self.cam_name}] TCP-Client verbunden: {addr}")
        try:
            while not stop.is_set():
                try:
                    data = conn.recv(1024)
                except socket.timeout:
                    continue
                if not data:
                    break
                if b"+" in data:
                    with self._tcp_scan_lock:
                        self._tcp_scan_token += 1
                        my_token = self._tcp_scan_token
                    logger.info(f"[{self.cam_name}] Trigger '+' (Token {my_token}) empfangen. Starte Auswertung...")
                    code = self._process_tcp_trigger_scan(token_id=my_token)
                    if code is not None:
                        response_bytes = b"\x02" + code.encode("utf-8") + b"\r\n\x04"
                        conn.sendall(response_bytes)
                    else:
                        logger.info(f"[{self.cam_name}] Scan (Token {my_token}) wurde verworfen — keine Antwort gesendet.")
                else:
                    conn.sendall(b"\x02ERROR_UNKNOWN_COMMAND\r\n\x04")
        except Exception as e_c:
            logger.error(f"[{self.cam_name}] TCP-Kommunikationsfehler: {e_c}")
        finally:
            conn.close()

    @staticmethod
    def _wait_next_frame(grabber: IDSFrameGrabber, after_time: float) -> tuple[np.ndarray | None, float]:
        """Nächstes Kamerabild nach after_time → (Bild, Zeitstempel); (None, after_time) nach Timeout."""
        end = time.time() + NEXT_FRAME_TIMEOUT_S
        while time.time() < end:
            frame_time = grabber.last_frame_time
            if frame_time > after_time:
                frame = grabber.get_frame()
                if frame is not None:
                    return frame, frame_time
            time.sleep(0.005)
        return None, after_time

    def _scan_frames(self, first_frame: np.ndarray, detections: list[dict], grabber: IDSFrameGrabber | None,
                     cancellation_check=None) -> dict:
        """
        Schnelle DataMatrix-Stufen auf bis zu 1 + EXTRA_SCAN_FRAMES Kamerabildern (gleiche YOLO-Boxen, das Teil
        steht beim Trigger). Ein Modulabgleich-Ergebnis wird durch ein weiteres Bild bestätigt; widersprechen
        sich zwei Bilder, wird kein Code gemeldet. Ohne Treffer folgt die langsame Pipeline auf dem ersten Bild.
        """
        cfg = self.app.config_data
        extra_frames = int(cfg.get("scan_extra_frames", EXTRA_SCAN_FRAMES))
        require_confirmation = bool(cfg.get("soft_require_confirmation", False))
        deadline = scanner.new_deadline()
        frame, frame_time = first_frame, (grabber.last_frame_time if grabber is not None else 0.0)
        soft = None
        first_hints = {}
        for index in range(1 + extra_frames):
            # Bestätigungsbilder ohne die langsamen zxing-Varianten: die haben schon im ersten Bild nichts gefunden
            result = scanner.scan_fast(frame, detections, deadline, zxing_rest=soft is None,
                                       hints=first_hints if index == 0 else None)
            if result is not None:
                result["frames_used"] = index + 1
                if result.get("verified"):
                    if soft is not None and soft["result"] != result["result"]:
                        logger.warning(f"[{self.cam_name}] Modulabgleich '{soft['result']}' im 1. Bild, "
                                       f"Reed-Solomon '{result['result']}' im {index + 1}. Bild → verifizierter Code gilt.")
                    return result
                if soft is None:
                    soft = result
                elif soft["result"] == result["result"]:
                    soft["method_detail"] += f", bestätigt im {index + 1}. Bild"
                    soft["frames_used"] = index + 1
                    return soft
                else:
                    logger.warning(f"[{self.cam_name}] Widerspruch zwischen Kamerabildern: "
                                   f"'{soft['result']}' vs. '{result['result']}' → kein Code.")
                    return {"success": False, "result": f"Widerspruch: {soft['result']} vs {result['result']}",
                            "method": "Fehler", "confidence": 0.0, "dmtx_result": None, "ocr_result": None,
                            "verified": False, "frames_used": index + 1}
            if index == extra_frames or grabber is None or time.perf_counter() > deadline \
                    or (cancellation_check and cancellation_check()):
                break
            frame, frame_time = self._wait_next_frame(grabber, frame_time)
            if frame is None:
                break

        if soft is not None:
            if not require_confirmation:
                soft["method_detail"] += ", unbestätigt"
                return soft
            logger.warning(f"[{self.cam_name}] Modulabgleich '{soft['result']}' ohne Bestätigung → OCR-Gegenprobe.")
            module = soft.get("_module") or {}
            if soft.get("frames_used") == 1 and module.get("quad"):
                # Die OCR der Klarschrift darf den Modulabgleich bestätigen (zwei unabhängige Quellen)
                first_hints["module"] = {"best": soft["result"], "ncc": module["ncc"], "margin": module["margin"],
                                         "frame_t": module["frame_t"], "quad": np.float32(module["quad"])}
        return scanner.scan_2class(first_frame, detections, cancellation_check=cancellation_check,
                                   deadline=deadline, run_fast=False, module_hint=first_hints.get("module"))

    def _process_tcp_trigger_scan(self, token_id: int = 0) -> str | None:
        """Wird aufgerufen wenn per TCP ein Trigger '+' empfangen wird."""
        cancellation_check = lambda: (token_id > 0 and self._tcp_scan_token != token_id)

        # Nur ein aktuelles Kamerabild: während eines Reconnects kein veraltetes Bild auswerten
        grabber = self.grabber
        scan_snapshot = grabber.get_frame() if grabber is not None else None
        if scan_snapshot is None:
            logger.error(f"[{self.cam_name}] TCP: Kein Kamerabild verfügbar!")
            return "ERROR_NO_FRAME"

        start_time = time.time()
        yolo_detections, use_2class, detection_conf, detection_box, scan_frame = \
            self.app.detect_labels(scan_snapshot, self.cam_name)

        # Abbrechen & Verwerfen falls in der Zwischenzeit ein neuer Trigger empfangen wurde
        if cancellation_check():
            logger.warning(f"[{self.cam_name}] Scan (Token {token_id}) VOR Auswertung abgebrochen & VERWORFEN!")
            return None

        if use_2class:
            result = self._scan_frames(scan_snapshot, yolo_detections, grabber, cancellation_check)
        else:
            result = scanner.scan(scan_frame, cancellation_check=cancellation_check)

        # Abbrechen & Verwerfen falls während der Auswertung ein neuer Trigger empfangen wurde
        if cancellation_check() or result.get("cancelled"):
            logger.warning(f"[{self.cam_name}] Scan (Token {token_id}) NACH Auswertung VERWORFEN (neuer Trigger).")
            return None

        duration_ms = int((time.time() - start_time) * 1000)
        result["duration_ms"] = duration_ms
        logger.info(f"[{self.cam_name}] TCP-Scan fertig ({duration_ms}ms): {result}")

        # Logging (nur für gültige, nicht stornierte Scans)
        self._log_scan(result, scan_snapshot, grabber, detection_conf, detection_box, use_2class, scan_frame,
                       trigger="TCP")

        # Update GUI live in main thread
        self.after(0, self._update_result, result)

        if result["success"]:
            return result["result"]
        else:
            return "ERROR"

    def _log_scan(self, result: dict, snapshot: np.ndarray, grabber: IDSFrameGrabber | None,
                  detection_conf: float, detection_box, use_2class: bool, scan_frame: np.ndarray, trigger: str):
        """Scan-Logging (JSONL + Bild) im Log-Unterordner dieser Kamera."""
        if self.scan_logger is None:
            return
        duration_ms = result["duration_ms"]
        cfg = self.app.config_data
        exp_us = getattr(grabber, 'exposure_us', float(cfg.get("last_exposure", 20.0)) * 1000.0)
        gain_val = getattr(grabber, 'gain', float(cfg.get("last_gain", 1.0)))
        self.scan_logger.log_scan(
            scan_result=result,
            frame=snapshot,
            timing={"total_ms": duration_ms, "yolo_ms": 0, "scan_ms": duration_ms},
            detection_info={
                "yolo_conf": detection_conf,
                "crop_size": [scan_frame.shape[1], scan_frame.shape[0]] if detection_box else None,
                "label_detected": (use_2class or detection_box is not None),
            },
            meta={
                "camera_model": grabber.model_name if grabber else "",
                "camera_serial": grabber.serial if grabber else "",
                "exposure_us": exp_us,
                "gain": gain_val,
                "app_version": APP_VERSION,
                "cam_id": self.cam_id,
                "port": self.tcp_port,
                "trigger": trigger,
                "auto_exposure": getattr(grabber, 'auto_exposure_enabled', False) if grabber else False
            },
        )

    def _draw_detections(self, frame, detections):
        """Zeichnet Bounding Boxes auf das Frame (W5)."""
        if detections is None or not hasattr(detections, 'boxes'):
            return frame
        
        out_frame = frame.copy()
        try:
            for box in detections.boxes:
                # Box-Koordinaten holen
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                
                # Klasse und Konfidenz
                conf = float(box.conf[0].cpu().item())
                cls_id = int(box.cls[0].cpu().item())
                cls_name = detections.names.get(cls_id, f"Klasse_{cls_id}")
                
                # Farbkodierung: Grün (BGR: 50, 205, 50) für DataMatrix (Klasse 0), Orange (BGR: 0, 165, 255) für Text (Klasse 1), Blau für 1-Klassen/Sonstiges
                if self.app.is_2class and cls_id == 0:
                    color_bgr = (50, 205, 50)  # Lime Green for DataMatrix
                elif self.app.is_2class and cls_id == 1:
                    color_bgr = (0, 165, 255)  # Orange for Text
                else:
                    color_bgr = (235, 99, 37)  # ACCENT Blue
                
                # Rahmen zeichnen
                cv2.rectangle(out_frame, (x1, y1), (x2, y2), color_bgr, 2)
                
                # Label zeichnen
                label_text = f"{cls_name} {conf:.2f}"
                
                # Textgröße berechnen
                (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                # Text oberhalb der Box platzieren, falls genug Platz vorhanden ist, sonst unterhalb
                if y1 - h - 4 > 0:
                    cv2.rectangle(out_frame, (x1, y1 - h - 4), (x1 + w, y1), color_bgr, -1)
                    cv2.putText(out_frame, label_text, (x1, y1 - 2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    cv2.rectangle(out_frame, (x1, y2), (x1 + w, y2 + h + 4), color_bgr, -1)
                    cv2.putText(out_frame, label_text, (x1, y2 + h + 2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        except Exception as e:
            logger.debug(f"Fehler beim Zeichnen der Boxen: {e}")
        return out_frame

    # ------------------------------------------------------------------ #
    #  Display Loop (Thread-Safe + Throttled YOLO)                         #
    # ------------------------------------------------------------------ #
    def _display_loop(self):
        prev_time = time.time()
        _reconnect_logged = False  # Verhindert Log-Spam bei Kamera-Ausfall
        reconnect_attempts = 0
        max_reconnect_attempts = 10

        while self.stream_running:
            try:
                # --- Kamera-Ausfallschutz: Auto-Reconnect (W3) ---
                is_disconnected = False
                if self.grabber is None:
                    is_disconnected = True
                else:
                    time_since_frame = time.time() - self.grabber.last_frame_time
                    if time_since_frame > 5.0:
                        is_disconnected = True

                if is_disconnected:
                    if not _reconnect_logged:
                        logger.warning(f"[{self.cam_name}] Kamera: Verbindung verloren oder kein Frame. Starte Auto-Reconnect...")
                        self.after(0, lambda: self._set_status("Kamera getrennt. Wiederverbindung...", WARN))
                        _reconnect_logged = True
                    
                    # Alten Grabber stoppen
                    if self.grabber is not None:
                        try:
                            self.grabber.stop()
                        except Exception:
                            pass
                        self.grabber = None
                    
                    reconnect_attempts += 1
                    logger.info(f"[{self.cam_name}] Kamera Reconnect-Versuch {reconnect_attempts}/{max_reconnect_attempts}...")
                    
                    new_grabber = self._open_camera(self.saved_serial)
                    if new_grabber is not None:
                        if not self.stream_running:  # während des Verbindens gestoppt
                            new_grabber.stop()
                            break
                        self.grabber = new_grabber
                        logger.info(f"[{self.cam_name}] Kamera: Auto-Reconnect erfolgreich!")
                        self.after(0, lambda: self._set_status("● LIVE (wiederverbunden)", SUCCESS))
                        _reconnect_logged = False
                        reconnect_attempts = 0
                    else:
                        if reconnect_attempts >= max_reconnect_attempts:
                            logger.error(f"[{self.cam_name}] Kamera: Maximale Reconnect-Versuche erreicht. Beende Stream.")
                            self.after(0, lambda: self._set_status("Verbindung verloren!", DANGER))
                            break
                        
                        time.sleep(5.0)  # 5 Sekunden warten vor erneutem Versuch
                    continue
                else:
                    _reconnect_logged = False

                grabber = self.grabber
                frame = grabber.get_frame() if grabber else None
                if frame is None:
                    time.sleep(0.01)
                    continue

                self._last_frame = frame
                if not self.visible and not self.auto_scan_enabled:
                    time.sleep(0.1)  # Verdeckter Reiter ohne Auto-Scan: nur Frame aktuell halten
                    continue

                # YOLO Inferenz: max 2x pro Sekunde (Throttling)
                # Thread-Sperre verhindert gleichzeitige Nutzung durch Scans und andere Reiter (W4)
                now = time.time()
                model = self.app.model
                if model is not None and (now - self._last_infer_time) >= 0.5:
                    self._last_infer_time = now
                    with self.app.model_lock:
                        results = model.predict(frame, conf=yolo_detector.PREDICT_CONF, verbose=False)
                        self._last_detections = results[0]

                # --- Smart Auto-Scan (Präsenzerkennung: DataMatrix ODER Text erkannt) ---
                if self.auto_scan_enabled and not self._scan_running:
                    has_presence = False
                    with self.app.model_lock:
                        if self._last_detections is not None and hasattr(self._last_detections, 'boxes'):
                            has_presence = yolo_detector.has_label_presence(self._last_detections.boxes)
                    
                    if has_presence:
                        self._absence_counter = 0
                        if self._presence_state == "EMPTY":
                            self._presence_counter += 1
                            if self._presence_counter >= 2:  # 2 aufeinanderfolgende Frames stabil
                                self._presence_state = "SCANNED"
                                logger.info(f"[{self.cam_name}] Auto-Scan getriggert (DataMatrix oder Text erkannt)!")
                                self.after(0, self.trigger_scan)
                    else:
                        self._presence_counter = 0
                        self._absence_counter += 1
                        if self._absence_counter >= 3:  # 3 leere Frames -> wieder bereit für nächste Horde
                            self._presence_state = "EMPTY"

                if not self.visible:
                    time.sleep(0.1)
                    continue
                
                # Synchronisierte Kopie der Detektionen zum Zeichnen holen (W4)
                detections_to_draw = None
                with self.app.model_lock:
                    if self._last_detections is not None:
                        detections_to_draw = self._last_detections
                
                # Detections direkt auf das aktuelle Frame zeichnen, verhindert Springen (W5)
                display_frame = frame
                if detections_to_draw is not None:
                    display_frame = self._draw_detections(display_frame, detections_to_draw)

                img_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

                cw = self.canvas.winfo_width()
                ch = self.canvas.winfo_height()
                if cw > 10 and ch > 10:
                    h, w = img_rgb.shape[:2]
                    scale = min(cw / w, ch / h)
                    nw, nh = int(w * scale), int(h * scale)

                    img_pil = Image.fromarray(img_rgb).resize((nw, nh), Image.Resampling.BILINEAR)
                    photo = ImageTk.PhotoImage(image=img_pil)

                    # Thread-safe: UI-Updates ueber self.after()
                    self.after(0, self._update_canvas, photo, cw, ch)

                curr = time.time()
                fps = 1.0 / max(curr - prev_time, 1e-6)
                prev_time = curr
                self.after(0, lambda t=f"FPS: {fps:.1f}": self.fps_label.configure(text=t))

                # --- Auto-Exposure Helligkeits-Anzeige aktualisieren ---
                if grabber and getattr(grabber, 'auto_exposure_enabled', False):
                    brightness = getattr(grabber, '_current_brightness', 0)
                    exp_ms = getattr(grabber, 'exposure_us', 0) / 1000.0
                    gain_now = getattr(grabber, 'gain', 1.0)
                    live_text = f"\u2600 Live: {brightness}/255 | {exp_ms:.1f}ms | G{gain_now:.1f}"
                    self.after(0, lambda t=live_text: self.brightness_live_label.configure(text=t))

            except Exception as e:
                logger.error(f"[{self.cam_name}] Display-Loop Fehler: {e}")
                time.sleep(0.1)

        # Nur _stop_stream aufrufen, wenn wir nicht bereits beim Stoppen sind
        if not self._stopping:
            self.after(0, self._stop_stream)

    def _update_canvas(self, photo, cw, ch):
        """Thread-sicheres Canvas-Update (laeuft im Main-Thread)."""
        try:
            # KRITISCH: Altes Bild-Item löschen, bevor ein neues erstellt wird!
            # Ohne diesen Schritt sammeln sich tausende Canvas-Items im Speicher
            # und der RAM-Verbrauch wächst unbegrenzt (Speicherleck im 24/7-Betrieb).
            if self._canvas_img_id is not None:
                self.canvas.delete(self._canvas_img_id)
            self._canvas_img_id = self.canvas.create_image(
                cw // 2, ch // 2, image=photo, anchor="center"
            )
            self.canvas.image = photo
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    #  Scan Trigger                                                        #
    # ------------------------------------------------------------------ #
    def trigger_scan(self):
        if self._scan_running:
            return
        if self._last_frame is None:
            self._update_result({
                "success": False, "result": "Kein Frame verfuegbar.",
                "method": "Fehler", "confidence": 0.0,
                "dmtx_result": None, "ocr_result": None, "verified": False
            })
            return

        self._scan_running = True
        self.scan_btn.configure(state="disabled", text="...Scanne")
        self.loading_bar.grid()
        self.loading_bar.start()

        # Frame-Snapshot einfrieren: Wir verwenden das Frame, das zum Zeitpunkt des Scans aktuell war,
        # nicht self._last_frame (das sich im Hintergrund ständig ändert).
        threading.Thread(target=self._run_scan, args=(self._last_frame.copy(),), daemon=True).start()

    def _run_scan(self, scan_snapshot):
        logger.info(f"[{self.cam_name}] Scan gestartet...")
        start_time = time.time()
        grabber = self.grabber

        yolo_detections, use_2class, detection_conf, detection_box, scan_frame = \
            self.app.detect_labels(scan_snapshot, self.cam_name)

        if use_2class:
            result = self._scan_frames(scan_snapshot, yolo_detections, grabber)
        else:
            result = scanner.scan(scan_frame)
        duration_ms = int((time.time() - start_time) * 1000)
        result["duration_ms"] = duration_ms
        logger.info(f"[{self.cam_name}] Scan Ergebnis: {result} (Dauer: {duration_ms}ms)")

        # --- Scan-Logging (JSONL + Bild) ---
        self._log_scan(result, scan_snapshot, grabber, detection_conf, detection_box, use_2class, scan_frame,
                       trigger="GUI")

        # --- Horden-Datenbank Bildspeicherung (mit Späterkennungs-Schutz) ---
        if result.get("success") and result.get("result"):
            is_late = (duration_ms > 6000)
            horde_db.save_or_update_horde_image(
                code=result["result"],
                frame=scan_snapshot,
                is_late_scan=is_late,
                verified=result.get("verified", False),
                confidence=result.get("confidence", 1.0)
            )

        # --- Active Learning: Auto-Save bei unsicherer Erkennung ---
        if result["success"] and detection_box is not None:
            self._scan_counter += 1
            # Speichere das volle Bild wenn KI unsicher war (Konfidenz 0.15-0.60)
            # ODER bei jedem 20. erfolgreichen Scan als allgemeine Datenmasse
            if (0.15 <= detection_conf <= 0.60) or (self._scan_counter % 20 == 0):
                self.app.save_auto_training(scan_snapshot, detection_box)

        self.after(0, self._update_result, result)

    def _update_result(self, result: dict):
        success = result["success"]
        text = result["result"]
        method = result["method"]
        confidence = result["confidence"]
        dmtx = result.get("dmtx_result")
        ocr = result.get("ocr_result")
        verified = result.get("verified", False)
        duration_ms = result.get("duration_ms", 0)
        ocr_partial = result.get("ocr_partial_display")
        
        # Farbe bestimmen
        if verified:
            color = SUCCESS
        elif method == "Rekonstruiert":
            color = WARN  # Orange für rekonstruierte Ergebnisse
        elif success:
            color = ACCENT
        else:
            color = DANGER
            
        # Hauptergebnis: Nur das bereinigte Endergebnis anzeigen (max 4 Zeichen)
        self.result_label.configure(text=text, text_color=color)
        self.method_label.configure(
            text=f"[{method}]",
            text_color=color
        )
        conf_text = f"{confidence * 100:.0f}%" if confidence > 0 else ""
        self.conf_label.configure(text=conf_text)
        
        # Detail-Zeile: Diagnose-Info (DMTX + OCR Rohdaten + Dauer)
        detail_parts = []
        if dmtx is not None:
            detail_parts.append(f"DMTX: {dmtx}")
        if ocr_partial is not None and ocr_partial != ocr:
            # Zeige Partial-Display wenn es vom OCR-Ergebnis abweicht (z.B. "A?12")
            detail_parts.append(f"OCR: {ocr_partial} (teilweise)")
        elif ocr is not None:
            detail_parts.append(f"OCR: {ocr}")
        if duration_ms > 0:
            detail_parts.append(f"Dauer: {duration_ms}ms")
        detail_text = "  |  ".join(detail_parts) if detail_parts else ""
        self.detail_label.configure(text=detail_text)
        self.auto_train_label.configure(text=f"Auto-Training: {self.app.auto_train_count} Bilder")
        
        self._scan_running = False
        self.scan_btn.configure(state="normal", text="◎  SCAN")
        try:
            self.loading_bar.stop()
            self.loading_bar.grid_remove()
        except Exception:
            pass

    def capture_training_image(self):
        """Speichert das aktuelle Frame für das KI-Training."""
        if self._last_frame is None:
            self._set_status("Kein Bild zum Speichern!", WARN)
            return
            
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"train_data_{timestamp}.jpg"
        filepath = os.path.join(TRAINING_DIR, filename)

        cv2.imwrite(filepath, self._last_frame)
        self._set_status(f"Gespeichert: {filename}", SUCCESS)
        logger.info(f"Training image saved: {filepath}")

    # ------------------------------------------------------------------ #
    #  Hilfsfunktionen                                                     #
    # ------------------------------------------------------------------ #
    def _set_status(self, text: str, color: str):
        self.status_label.configure(text=text, text_color=color)


class DataMatrixReaderApp(ctk.CTk):
    """Hauptfenster: ein Reiter pro Kamera; alle Kameras teilen sich ein YOLO-Modell."""

    def __init__(self):
        super().__init__()
        self.title(f"DataMatrixReader  —  DataDetector v{APP_VERSION}")
        self.geometry("1280x800")
        self.minsize(900, 700)
        self.configure(fg_color=BG_MAIN)

        self.config_data = _load_config()
        self.camera_ips = _configured_camera_ips(self.config_data)
        self.log_base_dir = resolve_log_directory(self.config_data.get("log_dir", DEFAULT_LOG_DIR))
        if "scan_time_budget_s" in self.config_data:
            scanner.config.SCAN_TIME_BUDGET_S = float(self.config_data["scan_time_budget_s"])

        # --- YOLO-Modell (Thread-Sperre gegen gleichzeitige Nutzung durch Anzeige und Scans) ---
        self.model = None
        self.is_2class = False
        self.model_lock = threading.Lock()
        self._model_load_lock = threading.Lock()

        os.makedirs(TRAINING_DIR, exist_ok=True)

        # --- Active Learning State ---
        os.makedirs(os.path.join(AUTO_TRAIN_DIR, "images"), exist_ok=True)
        os.makedirs(os.path.join(AUTO_TRAIN_DIR, "labels"), exist_ok=True)
        self.auto_train_count = len(os.listdir(os.path.join(AUTO_TRAIN_DIR, "images")))
        self._auto_train_lock = threading.Lock()

        self.panels: list[CameraPanel] = []
        self._autostart_pending = True
        self.tabview = ctk.CTkTabview(
            self, fg_color=BG_MAIN, anchor="w",
            segmented_button_selected_color=ACCENT, segmented_button_selected_hover_color="#1D4ED8",
            segmented_button_font=ctk.CTkFont(size=13, weight="bold"),
            command=self._on_tab_changed
        )
        self.tabview.pack(fill="both", expand=True, padx=12, pady=(4, 12))
        for slot in self._camera_slots():
            self._add_panel(slot)
        self._on_tab_changed()

        # --- Kameras beim Start im Hintergrund auflisten (danach Autostart) ---
        self.after(200, self.refresh_cameras)

    def _camera_slots(self) -> list[dict]:
        """Kamera-Reiter aus config["cameras"] (mindestens zwei); die bisherige Einzelkamera gehört zu Kamera 1."""
        slots = [c for c in self.config_data.get("cameras") or [] if isinstance(c, dict)]
        while len(slots) < DEFAULT_CAMERA_SLOTS:
            slots.append({})
        for number, slot in enumerate(slots, start=1):
            slot.setdefault("id", f"cam{number}")
            slot.setdefault("name", f"Kamera {number}")
        legacy_serial = self.config_data.get("selected_camera_serial")
        if legacy_serial:
            slots[0].setdefault("selected_camera_serial", legacy_serial)
        self.config_data["cameras"] = slots
        return slots

    def _add_panel(self, slot: dict):
        base_port = int(self.config_data.get("tcp_port", DEFAULT_TCP_PORT))
        port = int(slot.get("port", base_port + len(self.panels)))
        tab_name = f"{slot['name']}  ·  Port {port}"
        tab = self.tabview.add(tab_name)
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        panel = CameraPanel(tab, self, slot, port, tab_name)
        panel.grid(row=0, column=0, sticky="nsew")
        self.panels.append(panel)

    def _on_tab_changed(self):
        current = self.tabview.get()
        for panel in self.panels:
            panel.visible = panel.tab_name == current

    def save_config(self):
        _save_config(self.config_data)

    # ------------------------------------------------------------------ #
    #  Kameras                                                             #
    # ------------------------------------------------------------------ #
    def refresh_cameras(self):
        """Sucht nach verfügbaren IDS-Kameras und aktualisiert die Dropdown-Menüs aller Reiter."""
        def _worker():
            cameras = IDSFrameGrabber.list_cameras(self.camera_ips)
            self.after(0, self._on_cameras_listed, cameras)
        threading.Thread(target=_worker, daemon=True).start()

    def _on_cameras_listed(self, cameras: list[dict]):
        # Jede weitere angeschlossene Kamera bekommt einen eigenen Reiter (Kamera 3: Port 9502, ...)
        if len(cameras) > len(self.panels):
            for number in range(len(self.panels) + 1, len(cameras) + 1):
                slot = {"id": f"cam{number}", "name": f"Kamera {number}"}
                self.config_data["cameras"].append(slot)
                self._add_panel(slot)
            self.save_config()

        taken = {panel.saved_serial for panel in self.panels if panel.saved_serial}
        for panel in self.panels:
            shown = panel.update_camera_list(cameras, taken)
            if shown:
                taken.add(shown)

        if self._autostart_pending:
            self._autostart_pending = False
            for panel in self.panels:
                if panel.autostart_requested:
                    logger.info(f"[{panel.cam_name}] Autostart: Stream lief beim letzten Programmende.")
                    panel.start_stream(autostart=True)

    def panel_using_camera(self, serial: str, exclude: CameraPanel) -> CameraPanel | None:
        return next((p for p in self.panels if p is not exclude and p.active_serial == serial), None)

    # ------------------------------------------------------------------ #
    #  YOLO-Modell und Active Learning (von allen Reitern genutzt)         #
    # ------------------------------------------------------------------ #
    def ensure_model(self):
        """Lädt das YOLO-Modell beim ersten Stream-Start."""
        with self._model_load_lock:
            if self.model is None:
                self.model, self.is_2class = yolo_detector.load_model()

    def detect_labels(self, snapshot: np.ndarray, cam_name: str) -> tuple[list[dict], bool, float, tuple | None, np.ndarray]:
        """
        YOLO-Detektion vor dem Scan (Thread-Sperre gegen gleichzeitige Nutzung durch Anzeige und andere Kameras).

        Returns:
            (Detektionen mit conf > 0.3, 2-Klassen-Modus, beste Konfidenz,
             Etikett-Box im 1-Klassen-Modus, zu scannendes Bild)
        """
        if self.model is None:
            return [], False, 0.0, None, snapshot
        with self.model_lock:
            results = self.model.predict(snapshot, conf=yolo_detector.PREDICT_CONF, verbose=False)
        detections = yolo_detector.extract_detections(results[0], min_conf=0.3) if results else []

        detected_classes = {d["cls"] for d in detections}
        if self.is_2class and (0 in detected_classes or 1 in detected_classes):
            detection_conf = max(d["conf"] for d in detections)
            logger.info(f"[{cam_name}] 2-Klassen-Modus: {len(detections)} Detections (Klassen: {detected_classes}, "
                        f"max Conf: {detection_conf:.2f})")
            return detections, True, detection_conf, None, snapshot
        if detections:
            best_det = max(detections, key=lambda d: d["conf"])
            scan_frame = scanner.deskew_crop(snapshot, best_det["box"], padding=60)
            logger.info(f"[{cam_name}] 1-Klassen KI Etikett gefunden! Konfidenz: {best_det['conf']:.2f}. "
                        f"Ausschneiden und Begradigen auf {scan_frame.shape[1]}x{scan_frame.shape[0]}.")
            return detections, False, best_det["conf"], best_det["box"], scan_frame
        logger.warning(f"[{cam_name}] KI hat kein Etikett gefunden, scanne gesamtes Bild.")
        return detections, False, 0.0, None, snapshot

    def save_auto_training(self, full_frame, box):
        """Speichert das volle Bild + YOLO-Label automatisch für späteres Nachtraining."""
        with self._auto_train_lock:
            if self.auto_train_count >= AUTO_TRAIN_MAX:
                return
            try:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                img_name = f"auto_{timestamp}_{self.auto_train_count}.jpg"
                lbl_name = f"auto_{timestamp}_{self.auto_train_count}.txt"

                img_path = os.path.join(AUTO_TRAIN_DIR, "images", img_name)
                lbl_path = os.path.join(AUTO_TRAIN_DIR, "labels", lbl_name)

                # Bild speichern (volles Kamerabild!)
                cv2.imwrite(img_path, full_frame)

                # YOLO-Label generieren (normierte Koordinaten: x_center, y_center, width, height)
                fh, fw = full_frame.shape[:2]
                x1, y1, x2, y2 = box
                x_center = ((x1 + x2) / 2.0) / fw
                y_center = ((y1 + y2) / 2.0) / fh
                w = (x2 - x1) / fw
                h = (y2 - y1) / fh

                with open(lbl_path, "w") as f:
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

                self.auto_train_count += 1
                logger.info(f"Active Learning: Bild #{self.auto_train_count} gespeichert ({img_name})")
            except Exception as e:
                logger.error(f"Auto-Save Fehler: {e}")

    def on_closing(self):
        for panel in self.panels:
            panel.shutdown()
        self.destroy()


def _already_running() -> bool:
    """Nur eine Instanz: Kameras und TCP-Ports lassen sich nicht von zwei Programmen gleichzeitig nutzen."""
    global _instance_mutex
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    _instance_mutex = kernel32.CreateMutexW(None, False, SINGLE_INSTANCE_MUTEX)
    return ctypes.get_last_error() == 183  # ERROR_ALREADY_EXISTS


def main():
    if _already_running():
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("DataMatrixReader", "Der DataMatrixReader läuft bereits.")
        root.destroy()
        return

    if IDS_AVAILABLE:
        try:
            ids_peak.Library.Initialize()
        except Exception as e:
            logger.error(f"IDS peak Initialisierung fehlgeschlagen: {e}")

    app = DataMatrixReaderApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()

    if IDS_AVAILABLE:
        try:
            ids_peak.Library.Close()
        except Exception:
            pass
    logger.info("========== APP ENDE ==========")


if __name__ == "__main__":
    main()
    logging.shutdown()
    # Hintergrund-Threads (Kamera, Torch) nicht abwarten: Exit-Code 0 heißt für den Watchdog „bewusst beendet"
    os._exit(0)
