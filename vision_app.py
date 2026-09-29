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
from logging.handlers import RotatingFileHandler

# Root Logger konfigurieren
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
    from scan_logger import ScanLogger
    logger.info("Scanner Modul geladen.")
except Exception as e:
    logger.error(f"Scanner Importfehler: {e}")
    sys.exit(1)

# --- App-Version ---
APP_VERSION = "21.0"
CONFIG_FILE = "config.json"

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


class IDSFrameGrabber:
    """
    IDS peak SDK Frame-Grabber:
    Läuft in einem eigenen Thread und holt Bilder direkt von der IDS GigE-Kamera.
    Konvertiert Monochrom-Rohdaten automatisch zu BGR8 für OpenCV-Kompatibilität.
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
    def list_cameras() -> list[dict]:
        """
        Listet alle im System verfügbaren IDS-Kameras auf.
        
        Returns:
            list[dict]: Liste von Kamera-Dicts mit 'serial', 'model', 'display_name'.
        """
        cameras = []
        if not IDS_AVAILABLE:
            return cameras
        try:
            ids_peak.Library.Initialize()
            dm = ids_peak.DeviceManager.Instance()

            camera_ip = _load_config().get("camera_ip")
            if camera_ip:
                try:
                    dm.Update()
                    _register_unicast_ip(dm, camera_ip)
                except Exception:
                    pass

            dm.Update()
            devices = dm.Devices()
            for idx in range(_count(devices)):
                desc = devices[idx]
                model = desc.ModelName()
                serial = desc.SerialNumber()
                display_name = f"{model} ({serial})"
                cameras.append({
                    "serial": serial,
                    "model": model,
                    "display_name": display_name,
                    "index": idx
                })
        except Exception as e:
            logger.warning(f"Fehler bei Kamera-Auflistung: {e}")
        finally:
            try:
                ids_peak.Library.Close()
            except Exception:
                pass
        return cameras

    def start(self, target_serial: str | None = None):
        """Kamera finden, öffnen, Buffer anlegen und Aufnahme starten."""
        if not IDS_AVAILABLE:
            logger.error("IDS peak SDK nicht installiert!")
            return False
        try:
            ids_peak.Library.Initialize()
            dm = ids_peak.DeviceManager.Instance()

            # --- Unicast-Erkennung für Netzwerk-Kameras konfigurieren ---
            camera_ip = _load_config().get("camera_ip")
            if camera_ip:
                logger.info(f"Konfiguriere Unicast-Suche für Kamera-IP: {camera_ip} (0x{_ip_to_int(camera_ip):08X})")
                # Erste Update-Runde, damit Interfaces geöffnet werden
                dm.Update()
                try:
                    _register_unicast_ip(dm, camera_ip)
                except Exception as e_systems:
                    logger.warning(f"Fehler bei der Unicast-Konfiguration: {e_systems}")

            dm.Update()
            devices = dm.Devices()
            dev_count = _count(devices)
            if dev_count == 0:
                logger.error("Keine IDS-Kamera gefunden!")
                ids_peak.Library.Close()
                return False

            desc = None
            if target_serial:
                for idx in range(dev_count):
                    d = devices[idx]
                    if d.SerialNumber() == target_serial:
                        desc = d
                        logger.info(f"Ziel-Kamera gewählt: {d.ModelName()} (S/N: {target_serial})")
                        break

            if desc is None:
                desc = devices[0]
                logger.info(f"Standard-Kamera (0) gewählt: {desc.ModelName()} (S/N: {desc.SerialNumber()})")

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
            self._cleanup_partial()
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
        try:
            ids_peak.Library.Close()
        except Exception:
            pass
        self.frame = None
        logger.info("IDS Kamera sauber geschlossen.")

    def _cleanup_partial(self):
        """Aufräumen nach fehlgeschlagenem Start."""
        try:
            ids_peak.Library.Close()
        except Exception:
            pass


class AIVisionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(f"AI Vision Core  —  DataDetector v{APP_VERSION}")
        self.geometry("1280x760")
        self.minsize(900, 600)
        self.configure(fg_color=BG_MAIN)

        self.stream_running = False
        self._stopping = False  # Guard gegen doppelten Stop
        self.model = None
        self._is_2class = False
        self._model_lock = threading.Lock()  # Thread-Sperre für YOLO-Modell
        self.grabber: IDSFrameGrabber | None = None
        self._display_thread = None
        self._scan_running = False
        self._last_frame = None
        self._canvas_img_id = None  # Tracking für Canvas-Bild (Speicherleck-Fix)

        # --- Smart Auto-Scan (Präsenzerkennung) ---
        self.auto_scan_enabled = False
        self._presence_state = "EMPTY"  # "EMPTY" | "SCANNED"
        self._presence_counter = 0
        self._absence_counter = 0

        # --- ROI / Zoom State ---
        self._roi = None
        self._drawing = False
        self._draw_start = None
        self._draw_rect_id = None
        self._display_scale = 1.0
        self._display_offset = (0, 0)

        # --- YOLO Inferenz Throttling ---
        self._last_infer_time = 0.0
        self._last_detections = None  # Gecachtes Inferenz-Ergebnis

        # --- Config laden (URL Persistenz) ---
        self._config = _load_config()

        # --- TCP Server Attribute VOR _build_ui() definieren ---
        self.tcp_port = int(self._config.get("tcp_port", 9500))
        self._tcp_running = False
        self._tcp_thread = None
        self._tcp_scan_token = 0
        self._tcp_scan_lock = threading.Lock()

        self.training_dir = "training_data"
        os.makedirs(self.training_dir, exist_ok=True)

        # --- Active Learning State ---
        self._auto_train_dir = "auto_training_data"
        self._auto_train_max = 1000  # Maximale Anzahl Auto-Training-Bilder (Festplattenschutz)
        os.makedirs(os.path.join(self._auto_train_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self._auto_train_dir, "labels"), exist_ok=True)
        self._auto_train_count = len(os.listdir(os.path.join(self._auto_train_dir, "images")))
        self._scan_counter = 0

        self._camera_map = {}  # Map: display_name -> serial

        self._build_ui()

        # --- Kameras beim Start im Hintergrund auflisten ---
        self.after(200, self.refresh_cameras)

        # --- Scan-Logger initialisieren ---
        master_log_dir = self._config.get("log_dir", r"U:\Temp\DataMatrixReader.logFiles")
        save_all = self._config.get("save_all_scans", False)
        self.scan_logger = ScanLogger(log_dir=master_log_dir, save_all_scans=save_all)

        # --- TCP Server (Hintergrund-Dienst) starten ---
        self._start_tcp_server()

    # ------------------------------------------------------------------ #
    #  UI Builder                                                          #
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # -- Sidebar --
        self.sidebar = ctk.CTkFrame(self, width=230, corner_radius=0, fg_color=BG_SIDE,
                                    border_width=1, border_color=BORDER)
        self.sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self.sidebar.grid_propagate(False)
        self.sidebar.grid_rowconfigure(14, weight=1)

        ctk.CTkLabel(
            self.sidebar, text="AI Vision Core",
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
            text_color=ACCENT
        ).grid(row=0, column=0, padx=20, pady=(24, 2))

        ctk.CTkLabel(
            self.sidebar, text=f"DataDetector v{APP_VERSION}",
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
            command=self.refresh_cameras
        )
        self.cam_refresh_btn.grid(row=0, column=1, sticky="e")

        # -- Helligkeits-Regelung --
        ctk.CTkLabel(self.sidebar, text="Helligkeits-Regelung:", anchor="w",
                     font=ctk.CTkFont(weight="bold"), text_color=TXT_DARK
        ).grid(row=4, column=0, padx=20, sticky="w")

        self.settings_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.settings_frame.grid(row=5, column=0, padx=20, pady=(4, 10), sticky="ew")
        self.settings_frame.grid_columnconfigure(0, weight=1)

        saved_pct = self._config.get("auto_exposure_target_pct")
        if saved_pct is None:
            saved_target = self._config.get("auto_exposure_target", 130)
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
        if self._config.get("auto_scan", False):
            self.auto_scan_enabled = True
            self.auto_scan_switch.select()
        self.auto_scan_switch.grid(row=11, column=0, padx=20, pady=(6, 2), sticky="w")

        self.save_all_scans_switch = ctk.CTkSwitch(
            self.sidebar, text="Bilder aller Codes speich.",
            font=ctk.CTkFont(size=11, weight="bold"), text_color=TXT_DARK,
            command=self._on_save_all_scans_toggled
        )
        if self._config.get("save_all_scans", False):
            self.save_all_scans_switch.select()
        self.save_all_scans_switch.grid(row=12, column=0, padx=20, pady=(2, 6), sticky="w")

        self.tcp_info_label = ctk.CTkLabel(
            self.sidebar, text=f"TCP Port {self.tcp_port}: AKTIV",
            font=ctk.CTkFont(size=11, weight="bold"), text_color=SUCCESS
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
        self.main_frame.grid(row=0, column=1, padx=(0, 16), pady=16, sticky="nsew")
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
        self.result_panel.grid(row=1, column=1, padx=(0, 16), pady=(0, 16), sticky="ew")
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
            text=f"Auto-Training: {self._auto_train_count} Bilder",
            font=ctk.CTkFont(size=10), text_color=TXT_LIGHT, anchor="e"
        )
        self.auto_train_label.grid(row=2, column=2, padx=(8, 16), pady=(0, 2), sticky="e")

    def _on_brightness_slider_changed(self, val: float):
        """Wird aufgerufen wenn der Helligkeits-Schieberegler bewegt wird (0-100%)."""
        pct = int(val)
        target_br = int(40 + (pct / 100.0) * 180)
        self.brightness_target_label.configure(text=f"Ziel-Helligkeit: {pct}%")
        self._config["auto_exposure_target_pct"] = pct
        self._config["auto_exposure_target"] = target_br
        _save_config(self._config)

        if self.grabber:
            self.grabber.auto_exposure_target = target_br

    # ------------------------------------------------------------------ #
    #  Stream Steuerung                                                    #
    # ------------------------------------------------------------------ #
    def toggle_stream(self):
        if not self.stream_running:
            self._start_stream()
        else:
            self._stop_stream()

    def refresh_cameras(self):
        """Sucht nach verfügbaren IDS-Kameras und aktualisiert das Dropdown-Menü."""
        def _worker():
            cameras = IDSFrameGrabber.list_cameras()
            self.after(0, lambda: self._update_camera_dropdown(cameras))
        threading.Thread(target=_worker, daemon=True).start()

    def _update_camera_dropdown(self, cameras: list[dict]):
        self._camera_map.clear()
        if not cameras:
            display_values = ["Keine Kamera gefunden"]
            self.camera_optionmenu.configure(values=display_values)
            self.camera_optionmenu.set("Keine Kamera gefunden")
            return

        display_values = []
        for cam in cameras:
            name = cam["display_name"]
            serial = cam["serial"]
            self._camera_map[name] = serial
            display_values.append(name)

        self.camera_optionmenu.configure(values=display_values)

        # Gespeicherte Kamera auswählen falls vorhanden
        saved_serial = self._config.get("selected_camera_serial")
        selected_name = display_values[0]
        if saved_serial:
            for name, serial in self._camera_map.items():
                if serial == saved_serial:
                    selected_name = name
                    break

        self.camera_optionmenu.set(selected_name)
        if selected_name in self._camera_map:
            self._config["selected_camera_serial"] = self._camera_map[selected_name]
            _save_config(self._config)

    def _on_camera_selected(self, selected_display_name: str):
        serial = self._camera_map.get(selected_display_name)
        if not serial:
            return
        if self._config.get("selected_camera_serial") == serial:
            return

        logger.info(f"Kamera gewechselt zu: {selected_display_name} (S/N: {serial})")
        self._config["selected_camera_serial"] = serial
        _save_config(self._config)

        # Falls der Stream gerade läuft, neu starten
        if self.stream_running:
            self._stop_stream()
            self.after(500, self._start_stream)


    def _on_auto_scan_toggled(self):
        val = bool(self.auto_scan_switch.get())
        self.auto_scan_enabled = val
        self._config["auto_scan"] = val
        _save_config(self._config)
        logger.info(f"Auto-Scan Modus geändert: {val}")

    def _on_save_all_scans_toggled(self):
        val = bool(self.save_all_scans_switch.get())
        self._config["save_all_scans"] = val
        _save_config(self._config)
        if self.scan_logger:
            self.scan_logger.save_all_scans = val
        logger.info(f"Bilder-Speicher-Modus: {'ALLE Bilder speichern' if val else 'Nur FEHLER-Bilder speichern'}")

    # ------------------------------------------------------------------ #
    #  TCP Server (Hintergrund-Dienst)                                    #
    # ------------------------------------------------------------------ #
    def _start_tcp_server(self):
        """Startet den TCP-Server-Thread für externe Netzwerk-Trigger (z. B. '+')."""
        self._tcp_running = True
        self._tcp_thread = threading.Thread(target=self._run_tcp_server, daemon=True)
        self._tcp_thread.start()

    def _run_tcp_server(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind(('0.0.0.0', self.tcp_port))
            server.listen(5)
            logger.info(f"[GUI TCP SERVER] Lauscht auf Port {self.tcp_port}")
            self.after(0, lambda: self._update_tcp_label(f"TCP Port {self.tcp_port}: AKTIV"))
        except Exception as e:
            logger.error(f"[GUI TCP SERVER] Start-Fehler auf Port {self.tcp_port}: {e}")
            self.after(0, lambda: self._update_tcp_label(f"TCP Port {self.tcp_port}: FEHLER"))
            return

        while self._tcp_running:
            try:
                conn, addr = server.accept()
                conn.settimeout(1.0)
                logger.info(f"[GUI TCP SERVER] Client verbunden: {addr}")
                try:
                    while self._tcp_running:
                        try:
                            data = conn.recv(1024)
                            if not data:
                                break
                            if b"+" in data:
                                with self._tcp_scan_lock:
                                    self._tcp_scan_token += 1
                                    my_token = self._tcp_scan_token
                                logger.info(f"[GUI TCP SERVER] Trigger '+' (Token {my_token}) empfangen. Starte Auswertung...")
                                code = self._process_tcp_trigger_scan(token_id=my_token)
                                if code is not None:
                                    response_bytes = b"\x02" + code.encode("utf-8") + b"\r\n\x04"
                                    conn.sendall(response_bytes)
                                else:
                                    logger.info(f"[GUI TCP SERVER] Scan (Token {my_token}) wurde verworfen — keine Antwort gesendet.")
                            else:
                                conn.sendall(b"\x02ERROR_UNKNOWN_COMMAND\r\n\x04")
                        except socket.timeout:
                            continue
                except Exception as e_c:
                    logger.error(f"[GUI TCP SERVER] Kommunikationsfehler: {e_c}")
                finally:
                    conn.close()
            except Exception as e_s:
                if self._tcp_running:
                    logger.error(f"[GUI TCP SERVER] Server-Loop beendet: {e_s}")
                break
        try:
            server.close()
        except Exception:
            pass

    def _update_tcp_label(self, text: str):
        if hasattr(self, "tcp_info_label"):
            self.tcp_info_label.configure(text=text)

    def _detect_labels(self, snapshot: np.ndarray) -> tuple[list[dict], bool, float, tuple | None, np.ndarray]:
        """
        YOLO-Detektion vor dem Scan (Thread-Sperre gegen gleichzeitige Nutzung durch die Display-Loop).

        Returns:
            (Detektionen mit conf > 0.3, 2-Klassen-Modus, beste Konfidenz,
             Etikett-Box im 1-Klassen-Modus, zu scannendes Bild)
        """
        if self.model is None:
            return [], False, 0.0, None, snapshot
        with self._model_lock:
            results = self.model.predict(snapshot, conf=yolo_detector.PREDICT_CONF, verbose=False)
        detections = yolo_detector.extract_detections(results[0], min_conf=0.3) if results else []

        detected_classes = {d["cls"] for d in detections}
        if self._is_2class and (0 in detected_classes or 1 in detected_classes):
            detection_conf = max(d["conf"] for d in detections)
            logger.info(f"2-Klassen-Modus: {len(detections)} Detections (Klassen: {detected_classes}, "
                        f"max Conf: {detection_conf:.2f})")
            return detections, True, detection_conf, None, snapshot
        if detections:
            best_det = max(detections, key=lambda d: d["conf"])
            scan_frame = scanner.deskew_crop(snapshot, best_det["box"], padding=60)
            logger.info(f"1-Klassen KI Etikett gefunden! Konfidenz: {best_det['conf']:.2f}. Ausschneiden und "
                        f"Begradigen auf {scan_frame.shape[1]}x{scan_frame.shape[0]}.")
            return detections, False, best_det["conf"], best_det["box"], scan_frame
        logger.warning("KI hat kein Etikett gefunden, scanne gesamtes Bild.")
        return detections, False, 0.0, None, snapshot

    def _process_tcp_trigger_scan(self, token_id: int = 0) -> str | None:
        """Wird aufgerufen wenn per TCP ein Trigger '+' empfangen wird."""
        cancellation_check = lambda: (token_id > 0 and self._tcp_scan_token != token_id)

        frame = None
        if self.grabber is not None:
            frame = self.grabber.get_frame()
        if frame is None and self._last_frame is not None:
            frame = self._last_frame.copy()

        if frame is None:
            logger.error("[GUI TCP SERVER] Kein Kamerabild verfügbar!")
            return "ERROR_NO_FRAME"

        start_time = time.time()
        scan_snapshot = frame.copy()
        yolo_detections, use_2class, detection_conf, detection_box, scan_frame = self._detect_labels(scan_snapshot)

        # Abbrechen & Verwerfen falls in der Zwischenzeit ein neuer Trigger empfangen wurde
        if cancellation_check():
            logger.warning(f"[GUI TCP SERVER] Scan (Token {token_id}) VOR Auswertung abgebrochen & VERWORFEN!")
            return None

        if use_2class:
            result = scanner.scan_2class(scan_snapshot, yolo_detections, cancellation_check=cancellation_check)
        else:
            result = scanner.scan(scan_frame, cancellation_check=cancellation_check)

        # Abbrechen & Verwerfen falls während der Auswertung ein neuer Trigger empfangen wurde
        if cancellation_check() or result.get("cancelled"):
            logger.warning(f"[GUI TCP SERVER] Scan (Token {token_id}) NACH Auswertung VERWORFEN (neuer Trigger).")
            return None

        duration_ms = int((time.time() - start_time) * 1000)
        result["duration_ms"] = duration_ms
        logger.info(f"[GUI TCP SERVER] Scan fertig ({duration_ms}ms): {result}")

        # Logging (nur für gültige, nicht stornierte Scans)
        if self.scan_logger is not None:
            timing_info = {"total_ms": duration_ms, "yolo_ms": 0, "scan_ms": duration_ms}
            detection_info = {
                "yolo_conf": detection_conf,
                "crop_size": [scan_frame.shape[1], scan_frame.shape[0]] if detection_box else None,
                "label_detected": (use_2class or detection_box is not None),
            }
            exp_val = float(self._config.get("last_exposure", 20.0))
            gain_val = float(self._config.get("last_gain", 1.0))
            # Bei Auto-Exposure: Tatsächliche Werte vom Grabber verwenden
            if self.grabber and getattr(self.grabber, 'auto_exposure_enabled', False):
                exp_val = getattr(self.grabber, 'exposure_us', exp_val * 1000.0) / 1000.0
                gain_val = getattr(self.grabber, 'gain', gain_val)
            meta_info = {
                "camera_model": self.grabber.model_name if self.grabber else "",
                "camera_serial": self.grabber.serial if self.grabber else "",
                "exposure_us": exp_val * 1000.0,
                "gain": gain_val,
                "app_version": APP_VERSION,
                "port": self.tcp_port,
                "trigger": "TCP",
                "auto_exposure": getattr(self.grabber, 'auto_exposure_enabled', False) if self.grabber else False
            }
            self.scan_logger.log_scan(
                scan_result=result,
                frame=scan_snapshot,
                timing=timing_info,
                detection_info=detection_info,
                meta=meta_info,
            )

        # Update GUI live in main thread
        self.after(0, self._update_result, result)

        if result["success"]:
            return result["result"]
        else:
            return "ERROR"

    def _start_stream(self):
        # UI in Lade-Zustand versetzen
        self.start_btn.configure(state="disabled", text="...Verbinde")
        self._set_status("Lade Modell & Stream...", ACCENT)
        self.loading_bar.grid()          # Zeige Fortschrittsbalken
        self.loading_bar.start()         # Animation starten

        # Schwere Arbeit im Hintergrund-Thread
        threading.Thread(target=self._start_stream_worker, daemon=True).start()

    def _start_stream_worker(self):
        """Laeuft im Hintergrund-Thread: YOLO laden + Kamera verbinden."""
        try:
            # 1) YOLO Modell laden (kann 5-20 Sekunden dauern); ultralytics wird erst hier importiert,
            #    damit die GUI sofort startet.
            if self.model is None:
                self.after(0, lambda: self._set_status("Lade KI-Modell...", ACCENT))
                self.model, self._is_2class = yolo_detector.load_model()

            # 2) Kamera-Stream verbinden
            self.after(0, lambda: self._set_status("Verbinde mit Kamera...", WARN))
            grabber = IDSFrameGrabber()
            if not grabber.start(target_serial=self._config.get("selected_camera_serial")):
                self.after(0, self._on_stream_failed)
                return

            self.grabber = grabber
            self._configure_grabber(grabber)
            self.after(0, self._on_stream_connected)

        except Exception as e:
            logger.error(f"Stream-Start Fehler: {e}")
            self.after(0, lambda: self._on_stream_error(str(e)))

    def _configure_grabber(self, grabber: IDSFrameGrabber):
        """Übernimmt Belichtung, Gain und Auto-Exposure-Parameter aus der Config."""
        try:
            exp_val = float(self._config.get("last_exposure", 20.0))
            gain_val = float(self._config.get("last_gain", 1.0))
            grabber.set_exposure(exp_val * 1000.0)
            grabber.set_gain(gain_val)
        except Exception:
            pass

        if self._config.get("auto_exposure_enabled", False):
            grabber.auto_exposure_enabled = True
            grabber.auto_exposure_target = int(self._config.get("auto_exposure_target", 130))
            grabber.auto_exposure_deadzone = int(self._config.get("auto_exposure_deadzone", 10))
            grabber.auto_exposure_min_us = float(self._config.get("auto_exposure_min_ms", 1.0)) * 1000.0
            grabber.auto_exposure_max_us = float(self._config.get("auto_exposure_max_ms", 50.0)) * 1000.0
            grabber.auto_exposure_max_gain = float(self._config.get("auto_exposure_max_gain", 12.0))
            logger.info("Auto-Exposure aus Config aktiviert.")

    def _on_stream_connected(self):
        """Callback im Main-Thread: Stream erfolgreich verbunden."""
        self.loading_bar.stop()
        self.loading_bar.grid_remove()
        self.stream_running = True
        self._stopping = False
        self.start_btn.configure(state="normal", text="■  Stop Stream",
                                 fg_color=DANGER, hover_color="#B91C1C")
        self.scan_btn.configure(state="normal")
        self.train_capture_btn.configure(state="normal")
        self._set_status("● LIVE", SUCCESS)

        self._display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self._display_thread.start()

    def _on_stream_failed(self):
        """Callback im Main-Thread: Stream-Verbindung fehlgeschlagen."""
        self.loading_bar.stop()
        self.loading_bar.grid_remove()
        self.start_btn.configure(state="normal", text="▶  Start Stream",
                                 fg_color=ACCENT, hover_color="#1D4ED8")
        self._set_status("Stream Verbindung fehlgeschlagen!", DANGER)

    def _on_stream_error(self, msg):
        """Callback im Main-Thread: Allgemeiner Fehler beim Start."""
        self.loading_bar.stop()
        self.loading_bar.grid_remove()
        self.start_btn.configure(state="normal", text="▶  Start Stream",
                                 fg_color=ACCENT, hover_color="#1D4ED8")
        self._set_status(f"Fehler: {msg}", DANGER)

    def _stop_stream(self):
        # Guard gegen doppelten Aufruf
        if self._stopping:
            return
        self._stopping = True
        self.stream_running = False
        # Lade-Animation sicher beenden
        try:
            self.loading_bar.stop()
            self.loading_bar.grid_remove()
        except Exception:
            pass
        if self.grabber:
            self.grabber.stop()
            self.grabber = None
        self.start_btn.configure(
            state="normal", text="▶  Start Stream",
            fg_color=ACCENT, hover_color="#1D4ED8"
        )
        self.scan_btn.configure(state="disabled")
        self.train_capture_btn.configure(state="disabled")
        self._set_status("● Gestoppt", TXT_LIGHT)

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
                if self._is_2class and cls_id == 0:
                    color_bgr = (50, 205, 50)  # Lime Green for DataMatrix
                elif self._is_2class and cls_id == 1:
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
                        logger.warning("Kamera: Verbindung verloren oder kein Frame. Starte Auto-Reconnect...")
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
                    logger.info(f"Kamera Reconnect-Versuch {reconnect_attempts}/{max_reconnect_attempts}...")
                    
                    new_grabber = IDSFrameGrabber()
                    selected_serial = self._config.get("selected_camera_serial")
                    if new_grabber.start(target_serial=selected_serial):
                        self.grabber = new_grabber
                        self._configure_grabber(new_grabber)
                        logger.info("Kamera: Auto-Reconnect erfolgreich!")
                        self.after(0, lambda: self._set_status("● LIVE (wiederverbunden)", SUCCESS))
                        _reconnect_logged = False
                        reconnect_attempts = 0
                    else:
                        # Neuen Grabber sauber stoppen bei Fehlschlag
                        try:
                            new_grabber.stop()
                        except Exception:
                            pass
                        self.grabber = None
                        
                        if reconnect_attempts >= max_reconnect_attempts:
                            logger.error("Kamera: Maximale Reconnect-Versuche erreicht. Beende Stream.")
                            self.after(0, lambda: self._set_status("Verbindung verloren!", DANGER))
                            break
                        
                        time.sleep(5.0)  # 5 Sekunden warten vor erneutem Versuch
                    continue
                else:
                    _reconnect_logged = False

                frame = self.grabber.get_frame() if self.grabber else None
                if frame is None:
                    time.sleep(0.01)
                    continue

                self._last_frame = frame.copy()

                display_frame = self._crop_to_roi(frame)

                # YOLO Inferenz: max 2x pro Sekunde (Throttling)
                # Thread-Sperre verhindert gleichzeitige Nutzung durch Scan-Thread (W4)
                now = time.time()
                if self.model is not None and (now - self._last_infer_time) >= 0.5:
                    self._last_infer_time = now
                    with self._model_lock:
                        results = self.model.predict(display_frame, conf=yolo_detector.PREDICT_CONF, verbose=False)
                        self._last_detections = results[0]

                # --- Smart Auto-Scan (Präsenzerkennung: DataMatrix ODER Text erkannt) ---
                if self.auto_scan_enabled and not self._scan_running:
                    has_presence = False
                    with self._model_lock:
                        if self._last_detections is not None and hasattr(self._last_detections, 'boxes'):
                            has_presence = yolo_detector.has_label_presence(self._last_detections.boxes)
                    
                    if has_presence:
                        self._absence_counter = 0
                        if self._presence_state == "EMPTY":
                            self._presence_counter += 1
                            if self._presence_counter >= 2:  # 2 aufeinanderfolgende Frames stabil
                                self._presence_state = "SCANNED"
                                logger.info("Auto-Scan getriggert (DataMatrix oder Text erkannt)!")
                                self.after(0, self.trigger_scan)
                    else:
                        self._presence_counter = 0
                        self._absence_counter += 1
                        if self._absence_counter >= 3:  # 3 leere Frames -> wieder bereit für nächste Horde
                            self._presence_state = "EMPTY"
                
                # Synchronisierte Kopie der Detektionen zum Zeichnen holen (W4)
                detections_to_draw = None
                with self._model_lock:
                    if self._last_detections is not None:
                        detections_to_draw = self._last_detections
                
                # Detections direkt auf das aktuelle Frame zeichnen, verhindert Springen (W5)
                if detections_to_draw is not None:
                    display_frame = self._draw_detections(display_frame, detections_to_draw)

                img_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

                cw = self.canvas.winfo_width()
                ch = self.canvas.winfo_height()
                if cw > 10 and ch > 10:
                    h, w = img_rgb.shape[:2]
                    scale = min(cw / w, ch / h)
                    nw, nh = int(w * scale), int(h * scale)
                    self._display_scale = scale
                    self._display_offset = ((cw - nw) // 2, (ch - nh) // 2)

                    img_pil = Image.fromarray(img_rgb).resize((nw, nh), Image.Resampling.BILINEAR)
                    photo = ImageTk.PhotoImage(image=img_pil)

                    # Thread-safe: UI-Updates ueber self.after()
                    self.after(0, self._update_canvas, photo, cw, ch)

                curr = time.time()
                fps = 1.0 / max(curr - prev_time, 1e-6)
                prev_time = curr
                self.after(0, self.fps_label.configure, {"text": f"FPS: {fps:.1f}"})

                # --- Auto-Exposure Helligkeits-Anzeige aktualisieren ---
                if self.grabber and getattr(self.grabber, 'auto_exposure_enabled', False):
                    brightness = getattr(self.grabber, '_current_brightness', 0)
                    exp_ms = getattr(self.grabber, 'exposure_us', 0) / 1000.0
                    gain_now = getattr(self.grabber, 'gain', 1.0)
                    self.after(0, self.brightness_live_label.configure,
                              {"text": f"\u2600 Live: {brightness}/255 | {exp_ms:.1f}ms | G{gain_now:.1f}"})

            except Exception as e:
                logger.error(f"Display-Loop Fehler: {e}")
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
    def _crop_to_roi(self, frame: np.ndarray) -> np.ndarray:
        """Schneidet das Frame auf den gesetzten ROI zu (auf die Bildgrenzen begrenzt)."""
        if self._roi is None:
            return frame
        rx0, ry0, rx1, ry1 = self._roi
        fh, fw = frame.shape[:2]
        rx0 = max(0, min(rx0, fw - 1))
        ry0 = max(0, min(ry0, fh - 1))
        rx1 = max(rx0 + 1, min(rx1, fw))
        ry1 = max(ry0 + 1, min(ry1, fh))
        return frame[ry0:ry1, rx0:rx1]

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

        # Wenn ein ROI gesetzt ist, nur diesen Bereich scannen
        frame = self._crop_to_roi(self._last_frame.copy())
        if self._roi is not None:
            logger.info(f"Scanne ROI-Ausschnitt: {frame.shape[1]}x{frame.shape[0]} Pixel")

        threading.Thread(target=self._run_scan, args=(frame,), daemon=True).start()

    def _run_scan(self, frame):
        logger.info("Scan gestartet...")
        start_time = time.time()

        # Frame-Snapshot einfrieren: Wir verwenden das Frame, das zum
        # Zeitpunkt des Scans aktuell war, nicht self._last_frame
        # (das sich im Hintergrund ständig ändert).
        scan_snapshot = frame.copy()
        yolo_detections, use_2class, detection_conf, detection_box, scan_frame = self._detect_labels(scan_snapshot)

        if use_2class:
            result = scanner.scan_2class(scan_snapshot, yolo_detections)
        else:
            result = scanner.scan(scan_frame)
        duration_ms = int((time.time() - start_time) * 1000)
        result["duration_ms"] = duration_ms
        logger.info(f"Scan Ergebnis: {result} (Dauer: {duration_ms}ms)")

        # --- Scan-Logging (JSONL + Bild) ---
        if self.scan_logger is not None:
            timing_info = {
                "total_ms": duration_ms,
                "yolo_ms": 0,
                "scan_ms": duration_ms,
            }
            detection_info = {
                "yolo_conf": detection_conf,
                "crop_size": [scan_frame.shape[1], scan_frame.shape[0]] if detection_box else None,
                "label_detected": detection_box is not None,
            }
            exp_val = getattr(self.grabber, 'exposure_us', 6000.0) / 1000.0 if self.grabber else 6.0
            gain_val = getattr(self.grabber, 'gain', 1.0) if self.grabber else 1.0

            meta_info = {
                "camera_model": self.grabber.model_name if self.grabber else "",
                "camera_serial": self.grabber.serial if self.grabber else "",
                "exposure_us": exp_val * 1000.0,
                "gain": gain_val,
                "app_version": APP_VERSION,
                "auto_exposure": getattr(self.grabber, 'auto_exposure_enabled', False) if self.grabber else False
            }
            self.scan_logger.log_scan(
                scan_result=result,
                frame=scan_snapshot,
                timing=timing_info,
                detection_info=detection_info,
                meta=meta_info,
            )

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
            should_save = (0.15 <= detection_conf <= 0.60) or (self._scan_counter % 20 == 0)

            # Speicherplatz-Schutz: Maximal _auto_train_max Bilder speichern
            if should_save and self._auto_train_count < self._auto_train_max:
                self._auto_save_training(scan_snapshot, detection_box)

        self.after(0, self._update_result, result)

    def _auto_save_training(self, full_frame, box):
        """Speichert das volle Bild + YOLO-Label automatisch für späteres Nachtraining."""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            img_name = f"auto_{timestamp}_{self._auto_train_count}.jpg"
            lbl_name = f"auto_{timestamp}_{self._auto_train_count}.txt"
            
            img_path = os.path.join(self._auto_train_dir, "images", img_name)
            lbl_path = os.path.join(self._auto_train_dir, "labels", lbl_name)
            
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
            
            self._auto_train_count += 1
            logger.info(f"Active Learning: Bild #{self._auto_train_count} gespeichert ({img_name})")
            self.after(0, lambda: self.auto_train_label.configure(
                text=f"Auto-Training: {self._auto_train_count} Bilder"
            ))
            
        except Exception as e:
            logger.error(f"Auto-Save Fehler: {e}")

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
        
        self._scan_running = False
        self.scan_btn.configure(state="normal", text="◎  SCAN")
        try:
            self.loading_bar.stop()
            self.loading_bar.grid_remove()
        except Exception:
            pass

    def capture_training_image(self):
        """Speichert das aktuelle (ggf. gezoomte) Frame für das KI-Training."""
        if self._last_frame is None:
            self._set_status("Kein Bild zum Speichern!", WARN)
            return
            
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"train_data_{timestamp}.jpg"
        filepath = os.path.join(self.training_dir, filename)
        
        # Gezoomtes Frame verwenden, falls ROI aktiv ist
        frame_to_save = self._crop_to_roi(self._last_frame.copy())

        cv2.imwrite(filepath, frame_to_save)
        self._set_status(f"Gespeichert: {filename}", SUCCESS)
        logger.info(f"Training image saved: {filepath}")

    # ------------------------------------------------------------------ #
    #  Hilfsfunktionen                                                     #
    # ------------------------------------------------------------------ #
    def _set_status(self, text: str, color: str):
        self.status_label.configure(text=text, text_color=color)

    def on_closing(self):
        self._tcp_running = False
        # Session-Statistiken speichern
        if hasattr(self, 'scan_logger') and self.scan_logger is not None:
            self.scan_logger.save_session_summary()
            stats = self.scan_logger.get_session_stats()
            logger.info(
                f"Session beendet: {stats['total_scans']} Scans, "
                f"Erfolg: {stats.get('success_rate', 0):.1%}"
            )
        self._stop_stream()
        self.destroy()


if __name__ == "__main__":
    app = AIVisionApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
