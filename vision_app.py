import os
import sys
import json
import ctypes
import logging
import threading
import time
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

# --- YOLO wird LAZY geladen (erst beim Stream-Start im Hintergrund-Thread) ---
# Dadurch startet die GUI sofort, ohne auf PyTorch/YOLO zu warten.
YOLO = None  # Wird in _start_stream_worker importiert

# --- Scanner Modul laden ---
try:
    import scanner
    logger.info("Scanner Modul geladen.")
except Exception as e:
    logger.error(f"Scanner Importfehler: {e}")
    sys.exit(1)

# --- Scan-Logger laden ---
try:
    from scan_logger import ScanLogger
    logger.info("ScanLogger-Modul geladen.")
except Exception as e:
    logger.error(f"ScanLogger Importfehler: {e}")
    ScanLogger = None

# --- App-Version ---
APP_VERSION = "3.3"
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
            
            config = _load_config()
            camera_ip = config.get("camera_ip")
            if camera_ip:
                import struct as _struct
                import socket as _socket
                try:
                    ip_int = _struct.unpack("!I", _socket.inet_aton(camera_ip))[0]
                    dm.Update()
                    systems = dm.Systems()
                    sys_count = systems.size() if hasattr(systems, 'size') else len(systems)
                    for sys_idx in range(sys_count):
                        system = systems[sys_idx]
                        sys_name = system.DisplayName()
                        if "U3V" in sys_name or "USB" in sys_name:
                            continue
                        interfaces = system.Interfaces()
                        if_count = interfaces.size() if hasattr(interfaces, 'size') else len(interfaces)
                        for if_idx in range(if_count):
                            iface_desc = interfaces[if_idx]
                            if "Wi-Fi" in iface_desc.DisplayName():
                                continue
                            try:
                                opened = iface_desc.OpenedInterface()
                                nodemaps = opened.NodeMaps()
                                nm_count = nodemaps.size() if hasattr(nodemaps, 'size') else len(nodemaps)
                                if nm_count > 0:
                                    nodemap = nodemaps[0]
                                    if nodemap.HasNode("GevDiscoveryUnicastIPAddressToAdd"):
                                        nodemap.FindNode("GevDiscoveryUnicastIPAddressToAdd").SetValue(ip_int)
                                        nodemap.FindNode("GevDiscoveryUnicastIPAddressAdd").Execute()
                            except Exception:
                                pass
                except Exception:
                    pass
            
            dm.Update()
            devices = dm.Devices()
            dev_count = devices.size() if hasattr(devices, 'size') else len(devices)
            for idx in range(dev_count):
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
            config = _load_config()
            camera_ip = config.get("camera_ip")
            if camera_ip:
                import struct as _struct
                import socket as _socket
                # IP-Adresse in 32-Bit Integer konvertieren (GigE Vision Standard)
                ip_int = _struct.unpack("!I", _socket.inet_aton(camera_ip))[0]
                logger.info(f"Konfiguriere Unicast-Suche für Kamera-IP: {camera_ip} (0x{ip_int:08X})")
                
                # Erste Update-Runde, damit Interfaces geöffnet werden
                dm.Update()
                
                try:
                    systems = dm.Systems()
                    sys_count = systems.size() if hasattr(systems, 'size') else len(systems)
                    for sys_idx in range(sys_count):
                        system = systems[sys_idx]
                        sys_name = system.DisplayName()
                        if "U3V" in sys_name or "USB" in sys_name:
                            continue
                        interfaces = system.Interfaces()
                        if_count = interfaces.size() if hasattr(interfaces, 'size') else len(interfaces)
                        for if_idx in range(if_count):
                            iface_desc = interfaces[if_idx]
                            iface_name = iface_desc.DisplayName()
                            if "Wi-Fi" in iface_name:
                                continue
                            try:
                                opened = iface_desc.OpenedInterface()
                                nodemaps = opened.NodeMaps()
                                nm_count = nodemaps.size() if hasattr(nodemaps, 'size') else len(nodemaps)
                                if nm_count > 0:
                                    nodemap = nodemaps[0]
                                    if nodemap.HasNode("GevDiscoveryUnicastIPAddressToAdd"):
                                        nodemap.FindNode("GevDiscoveryUnicastIPAddressToAdd").SetValue(ip_int)
                                        nodemap.FindNode("GevDiscoveryUnicastIPAddressAdd").Execute()
                                        logger.info(f"Unicast-IP {camera_ip} für Interface '{iface_name}' registriert.")
                            except Exception as e_iface:
                                logger.warning(f"Unicast-Setup auf '{iface_name}' fehlgeschlagen: {e_iface}")
                except Exception as e_systems:
                    logger.warning(f"Fehler bei der Unicast-Konfiguration: {e_systems}")

            dm.Update()
            devices = dm.Devices()
            dev_count = devices.size() if hasattr(devices, 'size') else len(devices)
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
        if ScanLogger is not None:
            master_log_dir = self._config.get("log_dir", r"U:\Temp\DataMatrixReader.logFiles")
            self.scan_logger = ScanLogger(log_dir=master_log_dir)
        else:
            self.scan_logger = None
            logger.warning("ScanLogger nicht verfügbar — Scan-Logging deaktiviert.")

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
        self.sidebar.grid_rowconfigure(12, weight=1)

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

        # -- Kamera Einstellungen --
        ctk.CTkLabel(self.sidebar, text="Kamera Einstellungen:", anchor="w",
                     font=ctk.CTkFont(weight="bold"), text_color=TXT_DARK
        ).grid(row=4, column=0, padx=20, sticky="w")

        self.settings_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.settings_frame.grid(row=5, column=0, padx=20, pady=(4, 16), sticky="ew")
        self.settings_frame.grid_columnconfigure(0, weight=1)
        self.settings_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(self.settings_frame, text="Belichtung (ms):", font=ctk.CTkFont(size=11), text_color=TXT_DARK).grid(row=0, column=0, sticky="w", pady=2)
        self.exposure_entry = ctk.CTkEntry(self.settings_frame, width=70, height=24)
        self.exposure_entry.grid(row=0, column=1, sticky="e", pady=2)
        saved_exp = self._config.get("last_exposure", 20.0)
        self.exposure_entry.insert(0, str(saved_exp))
        self.exposure_entry.bind("<Return>", self._update_camera_settings)
        self.exposure_entry.bind("<FocusOut>", self._update_camera_settings)

        ctk.CTkLabel(self.settings_frame, text="Gain:", font=ctk.CTkFont(size=11), text_color=TXT_DARK).grid(row=1, column=0, sticky="w", pady=2)
        self.gain_entry = ctk.CTkEntry(self.settings_frame, width=70, height=24)
        self.gain_entry.grid(row=1, column=1, sticky="e", pady=2)
        saved_gain = self._config.get("last_gain", 1.0)
        self.gain_entry.insert(0, str(saved_gain))
        self.gain_entry.bind("<Return>", self._update_camera_settings)
        self.gain_entry.bind("<FocusOut>", self._update_camera_settings)

        self.start_btn = ctk.CTkButton(
            self.sidebar, text="▶  Start Stream", width=190,
            fg_color=ACCENT, hover_color="#1D4ED8",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self.toggle_stream
        )
        self.start_btn.grid(row=6, column=0, padx=20, pady=6)

        ctk.CTkFrame(self.sidebar, height=1, fg_color=BORDER).grid(
            row=7, column=0, padx=20, pady=12, sticky="ew"
        )

        # -- Zoom --
        ctk.CTkLabel(self.sidebar, text="Digital Zoom:", anchor="w",
                     font=ctk.CTkFont(weight="bold"), text_color=TXT_DARK
        ).grid(row=6, column=0, padx=20, sticky="w")

        ctk.CTkLabel(self.sidebar,
                     text="Ziehe ein Rechteck\nim Live-Bild zum Zoomen",
                     font=ctk.CTkFont(size=11), text_color=TXT_LIGHT,
                     justify="left"
        ).grid(row=7, column=0, padx=20, pady=(2, 4), sticky="w")

        self.reset_zoom_btn = ctk.CTkButton(
            self.sidebar, text="⟳  Zoom Reset", width=190,
            fg_color=TXT_MID, hover_color="#64748B",
            font=ctk.CTkFont(size=12),
            state="disabled",
            command=self.reset_zoom
        )
        self.reset_zoom_btn.grid(row=8, column=0, padx=20, pady=(0, 6))

        ctk.CTkFrame(self.sidebar, height=1, fg_color=BORDER).grid(
            row=9, column=0, padx=20, pady=12, sticky="ew"
        )

        # -- Aktionen --
        ctk.CTkLabel(self.sidebar, text="Aktionen:", anchor="w",
                     font=ctk.CTkFont(weight="bold"), text_color=TXT_DARK
        ).grid(row=10, column=0, padx=20, sticky="w")

        self.scan_btn = ctk.CTkButton(
            self.sidebar, text="◎  SCAN", width=190,
            fg_color=SUCCESS, hover_color="#15803D",
            font=ctk.CTkFont(size=14, weight="bold"),
            state="disabled",
            command=self.trigger_scan
        )
        self.scan_btn.grid(row=11, column=0, padx=20, pady=(4, 6))

        self.train_capture_btn = ctk.CTkButton(
            self.sidebar, text="◉  Capture (Training)", width=190,
            fg_color="#7C3AED", hover_color="#6D28D9",
            font=ctk.CTkFont(size=12, weight="bold"),
            state="disabled",
            command=self.capture_training_image
        )
        self.train_capture_btn.grid(row=12, column=0, padx=20, pady=(0, 6))

        self.zoom_info_label = ctk.CTkLabel(
            self.sidebar, text="Zoom: 1.0x",
            font=ctk.CTkFont(size=11), text_color=TXT_LIGHT
        )
        self.zoom_info_label.grid(row=13, column=0, padx=20, pady=(4, 0))

        self.status_label = ctk.CTkLabel(
            self.sidebar, text="● Bereit", text_color=TXT_LIGHT,
            font=ctk.CTkFont(size=12)
        )
        self.status_label.grid(row=14, column=0, padx=20, pady=(20, 4), sticky="s")

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

        self.canvas = tk.Canvas(self.main_frame, bg="#E2E8F0", highlightthickness=0, cursor="crosshair")
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)

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

    # ------------------------------------------------------------------ #
    #  Maus-Zoom: Rechteck zeichnen                                        #
    # ------------------------------------------------------------------ #
    def _on_mouse_down(self, event):
        """Maus-Klick: Beginn des Rechteck-Zeichnens."""
        if not self.stream_running:
            return
        self._drawing = True
        self._draw_start = (event.x, event.y)
        # Altes Rechteck löschen
        if self._draw_rect_id:
            self.canvas.delete(self._draw_rect_id)
            self._draw_rect_id = None

    def _on_mouse_drag(self, event):
        """Maus wird gezogen: Rechteck live mitzeichnen."""
        if not self._drawing or not self._draw_start:
            return
        if self._draw_rect_id:
            self.canvas.delete(self._draw_rect_id)
        x0, y0 = self._draw_start
        self._draw_rect_id = self.canvas.create_rectangle(
            x0, y0, event.x, event.y,
            outline="#3B82F6", width=2, dash=(6, 4)
        )

    def _on_mouse_up(self, event):
        """Maus losgelassen: ROI berechnen und Zoom aktivieren."""
        if not self._drawing or not self._draw_start:
            return
        self._drawing = False

        x0_canvas, y0_canvas = self._draw_start
        x1_canvas, y1_canvas = event.x, event.y

        # Sicherstellen, dass x0 < x1 und y0 < y1
        x0_canvas, x1_canvas = min(x0_canvas, x1_canvas), max(x0_canvas, x1_canvas)
        y0_canvas, y1_canvas = min(y0_canvas, y1_canvas), max(y0_canvas, y1_canvas)

        # Zu kleines Rechteck ignorieren (Klick ohne Ziehen)
        if (x1_canvas - x0_canvas) < 20 or (y1_canvas - y0_canvas) < 20:
            if self._draw_rect_id:
                self.canvas.delete(self._draw_rect_id)
                self._draw_rect_id = None
            return

        # Canvas-Koordinaten → Original-Bild-Koordinaten umrechnen
        off_x, off_y = self._display_offset
        scale = self._display_scale

        if scale <= 0:
            return

        # Zuerst: Wenn wir bereits einen ROI haben, sind die Koordinaten relativ zum ROI
        img_x0 = int((x0_canvas - off_x) / scale)
        img_y0 = int((y0_canvas - off_y) / scale)
        img_x1 = int((x1_canvas - off_x) / scale)
        img_y1 = int((y1_canvas - off_y) / scale)

        # Wenn ein ROI bereits gesetzt ist, auf das volle Bild umrechnen
        if self._roi is not None:
            roi_x0, roi_y0, _, _ = self._roi
            img_x0 += roi_x0
            img_y0 += roi_y0
            img_x1 += roi_x0
            img_y1 += roi_y0

        # Begrenzen auf Bildgröße
        if self._last_frame is not None:
            fh, fw = self._last_frame.shape[:2]
            img_x0 = max(0, min(img_x0, fw - 1))
            img_y0 = max(0, min(img_y0, fh - 1))
            img_x1 = max(0, min(img_x1, fw))
            img_y1 = max(0, min(img_y1, fh))

        if (img_x1 - img_x0) < 10 or (img_y1 - img_y0) < 10:
            return

        self._roi = (img_x0, img_y0, img_x1, img_y1)
        logger.info(f"ROI gesetzt: {self._roi}")

        # Zoom-Faktor berechnen und anzeigen
        if self._last_frame is not None:
            fh, fw = self._last_frame.shape[:2]
            roi_w = img_x1 - img_x0
            zoom_factor = fw / roi_w if roi_w > 0 else 1.0
            self.zoom_info_label.configure(text=f"Zoom: {zoom_factor:.1f}x")

        self.reset_zoom_btn.configure(state="normal")

        # Zeichnungsrechteck entfernen (wird jetzt durch den Zoom ersetzt)
        if self._draw_rect_id:
            self.canvas.delete(self._draw_rect_id)
            self._draw_rect_id = None

    def reset_zoom(self):
        """Zoom zurücksetzen auf Vollbild."""
        self._roi = None
        self.reset_zoom_btn.configure(state="disabled")
        self.zoom_info_label.configure(text="Zoom: 1.0x")
        logger.info("Zoom zurückgesetzt.")

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

    def _update_camera_settings(self, event=None):
        if not self.stream_running or not self.grabber:
            return
        try:
            exp_val = float(self.exposure_entry.get())
            gain_val = float(self.gain_entry.get())
            self.grabber.set_exposure(exp_val * 1000.0)  # ms -> us
            self.grabber.set_gain(gain_val)
            
            # Save to config
            self._config["last_exposure"] = exp_val
            self._config["last_gain"] = gain_val
            _save_config(self._config)
        except ValueError:
            pass

    def _start_stream(self):
        try:
            exp_val = float(self.exposure_entry.get())
            gain_val = float(self.gain_entry.get())
            self._config["last_exposure"] = exp_val
            self._config["last_gain"] = gain_val
            _save_config(self._config)
        except ValueError:
            self._set_status("Ungültige Werte!", DANGER)
            return

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
            # 1) YOLO Modell laden (kann 5-20 Sekunden dauern)
            #    Import findet hier statt, damit die GUI sofort startet!
            global YOLO
            if YOLO is None:
                self.after(0, lambda: self._set_status("Lade KI-Modell...", ACCENT))
                from ultralytics import YOLO as _YOLO
                YOLO = _YOLO
                logger.info("YOLO (ultralytics) lazy-importiert.")

            if self.model is None:
                if getattr(sys, 'frozen', False):
                    app_dir = os.path.dirname(sys.executable)
                else:
                    app_dir = os.path.dirname(os.path.abspath(__file__))
                
                model_path_2class = os.path.join(app_dir, "runs", "detect", "training_runs_v2", "horde_2class", "weights", "best.pt")
                model_path_1class = os.path.join(app_dir, "runs", "detect", "training_runs", "horde_model", "weights", "best.pt")
                
                if os.path.exists(model_path_2class):
                    self.model = YOLO(model_path_2class)
                    self._is_2class = True
                    logger.info(f"2-Klassen YOLO Modell geladen (datamatrix+text): {model_path_2class}")
                elif os.path.exists(model_path_1class):
                    self.model = YOLO(model_path_1class)
                    self._is_2class = False
                    logger.info(f"1-Klassen YOLO Modell geladen (Horde): {model_path_1class}")
                else:
                    if getattr(sys, 'frozen', False):
                        base_model_path = os.path.join(sys._MEIPASS, "yolov10n.pt")
                    else:
                        base_model_path = os.path.join(app_dir, "yolov10n.pt")
                    
                    self.model = YOLO(base_model_path)
                    self._is_2class = False
                    logger.warning(f"Kein trainiertes Modell gefunden, nutze Standard {base_model_path}")

            # 2) Kamera-Stream verbinden
            self.after(0, lambda: self._set_status("Verbinde mit Kamera...", WARN))
            selected_serial = self._config.get("selected_camera_serial")
            grabber = IDSFrameGrabber()
            if not grabber.start(target_serial=selected_serial):
                self.after(0, self._on_stream_failed)
                return

            self.grabber = grabber
            # Set initial camera settings
            try:
                exp_val = float(self._config.get("last_exposure", 20.0))
                gain_val = float(self._config.get("last_gain", 1.0))
                self.grabber.set_exposure(exp_val * 1000.0)
                self.grabber.set_gain(gain_val)
            except Exception:
                pass

            self.after(0, self._on_stream_connected)

        except Exception as e:
            logger.error(f"Stream-Start Fehler: {e}")
            self.after(0, lambda: self._on_stream_error(str(e)))

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
                        try:
                            exp_val = float(self._config.get("last_exposure", 20.0))
                            gain_val = float(self._config.get("last_gain", 1.0))
                            self.grabber.set_exposure(exp_val * 1000.0)
                            self.grabber.set_gain(gain_val)
                        except Exception:
                            pass
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

                display_frame = frame
                if self._roi is not None:
                    rx0, ry0, rx1, ry1 = self._roi
                    fh, fw = frame.shape[:2]
                    rx0 = max(0, min(rx0, fw - 1))
                    ry0 = max(0, min(ry0, fh - 1))
                    rx1 = max(rx0 + 1, min(rx1, fw))
                    ry1 = max(ry0 + 1, min(ry1, fh))
                    display_frame = frame[ry0:ry1, rx0:rx1]

                # YOLO Inferenz: max 2x pro Sekunde (Throttling)
                # Thread-Sperre verhindert gleichzeitige Nutzung durch Scan-Thread (W4)
                now = time.time()
                if self.model is not None and (now - self._last_infer_time) >= 0.5:
                    self._last_infer_time = now
                    with self._model_lock:
                        results = self.model.predict(display_frame, conf=0.15, verbose=False)
                        self._last_detections = results[0]
                
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
        frame = self._last_frame.copy()
        if self._roi is not None:
            rx0, ry0, rx1, ry1 = self._roi
            fh, fw = frame.shape[:2]
            rx0 = max(0, min(rx0, fw - 1))
            ry0 = max(0, min(ry0, fh - 1))
            rx1 = max(rx0 + 1, min(rx1, fw))
            ry1 = max(ry0 + 1, min(ry1, fh))
            frame = frame[ry0:ry1, rx0:rx1]
            logger.info(f"Scanne ROI-Ausschnitt: {rx1-rx0}x{ry1-ry0} Pixel")

        threading.Thread(target=self._run_scan, args=(frame,), daemon=True).start()

    def _run_scan(self, frame):
        logger.info("Scan gestartet...")
        start_time = time.time()

        # Frame-Snapshot einfrieren: Wir verwenden das Frame, das zum
        # Zeitpunkt des Scans aktuell war, nicht self._last_frame
        # (das sich im Hintergrund ständig ändert).
        scan_snapshot = frame.copy()
        scan_frame = scan_snapshot
        detection_conf = 0.0
        detection_box = None

        # KI-basiertes Zuschneiden (Cropping) / 2-Klassen Erkennung vor dem Scannen
        # Thread-Sperre verhindert gleichzeitige YOLO-Nutzung durch Display-Loop
        yolo_detections = []
        use_2class = False
        if self.model is not None:
            with self._model_lock:
                results = self.model.predict(scan_snapshot, conf=0.15, verbose=False)
            if results and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if conf > 0.3:
                        yolo_detections.append({
                            "cls": cls_id,
                            "box": (x1, y1, x2, y2),
                            "conf": conf,
                        })

                detected_classes = set(d["cls"] for d in yolo_detections)
                if self._is_2class and (0 in detected_classes or 1 in detected_classes):
                    use_2class = True
                    detection_conf = max(d["conf"] for d in yolo_detections) if yolo_detections else 0.0
                    logger.info(f"2-Klassen-Modus: {len(yolo_detections)} Detections (Klassen: {detected_classes}, max Conf: {detection_conf:.2f})")
                elif yolo_detections:
                    best_det = max(yolo_detections, key=lambda d: d["conf"])
                    detection_conf = best_det["conf"]
                    detection_box = best_det["box"]
                    scan_frame = scanner.deskew_crop(scan_snapshot, detection_box, padding=60)
                    logger.info(f"1-Klassen KI Etikett gefunden! Konfidenz: {detection_conf:.2f}. Ausschneiden und Begradigen auf {scan_frame.shape[1]}x{scan_frame.shape[0]}.")
                else:
                    logger.warning("KI hat kein Etikett gefunden, scanne gesamtes Bild.")
            else:
                logger.warning("KI hat kein Etikett gefunden, scanne gesamtes Bild.")

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
            exp_val = 20.0
            try:
                exp_val = float(self.exposure_entry.get())
            except Exception:
                exp_val = float(self._config.get("last_exposure", 20.0))

            gain_val = 1.0
            try:
                gain_val = float(self.gain_entry.get())
            except Exception:
                gain_val = float(self._config.get("last_gain", 1.0))

            meta_info = {
                "camera_model": self.grabber.model_name if self.grabber else "",
                "camera_serial": self.grabber.serial if self.grabber else "",
                "exposure_us": exp_val * 1000.0,
                "gain": gain_val,
                "app_version": APP_VERSION,
            }
            self.scan_logger.log_scan(
                scan_result=result,
                frame=scan_snapshot,
                timing=timing_info,
                detection_info=detection_info,
                meta=meta_info,
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
        frame_to_save = self._last_frame.copy()
        if self._roi is not None:
            rx0, ry0, rx1, ry1 = self._roi
            fh, fw = frame_to_save.shape[:2]
            rx0 = max(0, min(rx0, fw - 1))
            ry0 = max(0, min(ry0, fh - 1))
            rx1 = max(rx0 + 1, min(rx1, fw))
            ry1 = max(ry0 + 1, min(ry1, fh))
            frame_to_save = frame_to_save[ry0:ry1, rx0:rx1]

        cv2.imwrite(filepath, frame_to_save)
        self._set_status(f"Gespeichert: {filename}", SUCCESS)
        logger.info(f"Training image saved: {filepath}")

    # ------------------------------------------------------------------ #
    #  Hilfsfunktionen                                                     #
    # ------------------------------------------------------------------ #
    def _set_status(self, text: str, color: str):
        self.status_label.configure(text=text, text_color=color)

    def on_closing(self):
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
