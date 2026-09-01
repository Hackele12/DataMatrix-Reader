"""
vision_app_v4_network.py — Unified Multi-Camera Headless Industrial TCP/IP Scanner (v4.0)

Diese Version steuert BEIDE IDS-GigE-Kameras (Cam1: 172.31.146.192 @ Port 9500, Cam2: 172.31.146.193 @ Port 9501)
in EINEM einzigen Python-Prozess:
1. Lädt das lokale YOLO-KI-Modell EINMALIG für alle Kameras (speicherschonend).
2. Verknüpft jede Kamera zielgerichtet über ihre IP-Adresse (kein vertauschter Zugriff).
3. Startet dedizierte TCP-Server-Threads auf den jeweiligen Ports (z.B. 9500 & 9501).
4. Bei Empfang von "+" verarbeitet der jeweilige Port das Bild der zugehörigen Kamera
   mit Dual-Validation (DataMatrix + EasyOCR) und antwortet mit Carriage Return (\r).
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
import warnings
from logging.handlers import RotatingFileHandler
import cv2
import numpy as np

# PyTorch/User-Warnings auf CPU unterdrücken für sauberere Konsolenausgabe
warnings.filterwarnings("ignore", category=UserWarning)

# --- Config Laden ---
CONFIG_FILE = "config.json"
if len(sys.argv) > 1:
    CONFIG_FILE = sys.argv[1]

# --- Logging einrichten ---
logger = logging.getLogger("VisionNetworkApp")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = RotatingFileHandler("vision_network_main.log", maxBytes=5*1024*1024, backupCount=5, encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# --- Windows Crash-Dialog unterdrücken ---
try:
    if sys.platform == "win32":
        SEM_NOGPFAULTERRORBOX = 0x0002
        SEM_FAILCRITICALERRORS = 0x0001
        SEM_NOOPENFILEERRORBOX = 0x8000
        ctypes.windll.kernel32.SetErrorMode(
            SEM_NOGPFAULTERRORBOX | SEM_FAILCRITICALERRORS | SEM_NOOPENFILEERRORBOX
        )
except Exception as e:
    logger.warning(f"Konnte Windows-Fehlermodi nicht setzen: {e}")

# --- IDS peak SDK laden ---
try:
    from ids_peak import ids_peak
    from ids_peak_ipl import ids_peak_ipl
    IDS_AVAILABLE = True
    logger.info("IDS peak SDK erfolgreich geladen.")
except ImportError:
    IDS_AVAILABLE = False
    logger.error("IDS peak SDK konnte nicht geladen werden! Kamera-Funktion nicht verfügbar.")

# --- Scanner Modul laden ---
try:
    import scanner
    logger.info("Scanner-Modul (Dual-Validation) erfolgreich geladen.")
except Exception as e:
    logger.error(f"Scanner Importfehler: {e}")
    sys.exit(1)

# --- Scan-Logger laden ---
try:
    import scan_logger
    from scan_logger import ScanLogger, resolve_log_directory
    logger.info("ScanLogger-Modul geladen.")
except Exception as e:
    logger.error(f"ScanLogger Importfehler: {e}")
    ScanLogger = None
    resolve_log_directory = lambda p: p


def load_master_config(config_path: str) -> tuple[list[dict], str]:
    """Lädt die Liste aller Kamera-Konfigurationen und den Master-Log-Pfad aus config.json."""
    default_log_dir = r"U:\Temp\DataMatrixReader.logFiles"
    default_cams = [
        {
            "id": "cam1",
            "name": "Kamera 1",
            "camera_ip": "172.31.146.192",
            "port": 9500,
            "last_exposure": 6.0,
            "last_gain": 2.0,
            "log_file": "vision_network_cam1.log"
        },
        {
            "id": "cam2",
            "name": "Kamera 2",
            "camera_ip": "172.31.146.193",
            "port": 9501,
            "last_exposure": 6.0,
            "last_gain": 2.0,
            "log_file": "vision_network_cam2.log"
        }
    ]
    if not os.path.exists(config_path):
        logger.warning(f"Config-Datei '{config_path}' nicht gefunden. Verwende Default Multi-Cam Config.")
        return default_cams, default_log_dir

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            master_log_dir = data.get("log_dir", default_log_dir)
            if "cameras" in data and isinstance(data["cameras"], list):
                return data["cameras"], master_log_dir
            elif "camera_ip" in data:
                return [data], master_log_dir
    except Exception as e:
        logger.error(f"Fehler beim Lesen der Config '{config_path}': {e}")
    return default_cams, default_log_dir


class IDSFrameGrabber:
    """Frame-Grabber für eine spezifische IDS-Kamera (gezielt per IP gematcht)."""
    def __init__(self, camera_ip: str, cam_name: str = "Kamera", other_configured_ips: list[str] | None = None):
        self.camera_ip = camera_ip
        self.cam_name = cam_name
        self.other_configured_ips = set(other_configured_ips or [])
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
        self.exposure_us = 6000.0
        self.gain = 2.0
        self.last_frame_time = time.time()

    def start(self) -> bool:
        if not IDS_AVAILABLE:
            logger.error(f"[{self.cam_name}] Start unmöglich: IDS peak SDK fehlt.")
            return False
        try:
            dm = ids_peak.DeviceManager.Instance()
            
            # Unicast IP-Adresse im DeviceManager registrieren
            if self.camera_ip:
                try:
                    ip_int = struct.unpack("!I", socket.inet_aton(self.camera_ip))[0]
                    dm.Update()
                    for sys_obj in dm.Systems():
                        for iface in sys_obj.Interfaces():
                            try:
                                op = iface.OpenedInterface()
                                for nm in op.NodeMaps():
                                    if nm.HasNode("GevDiscoveryUnicastIPAddressToAdd"):
                                        nm.FindNode("GevDiscoveryUnicastIPAddressToAdd").SetValue(ip_int)
                                        nm.FindNode("GevDiscoveryUnicastIPAddressAdd").Execute()
                            except Exception:
                                pass
                except Exception as e_u:
                    logger.warning(f"[{self.cam_name}] Unicast IP Setup ({self.camera_ip}): {e_u}")

            dm.Update()
            devices = list(dm.Devices())
            logger.info(f"[{self.cam_name}] IDS Device Manager ergab {len(devices)} Gerät(e).")

            openable_candidates = []
            for idx, desc in enumerate(devices):
                openable = desc.IsOpenable()
                model_n = desc.ModelName()
                sn = desc.SerialNumber()
                logger.info(f"  [{self.cam_name}] Gerät {idx}: {model_n} (S/N: {sn}), IsOpenable={openable}")
                if openable:
                    openable_candidates.append(desc)

            matched_desc = None
            matched_device = None
            matched_nodemap = None

            # 1. Stufe: Exakte IP-Übereinstimmung
            if self.camera_ip and openable_candidates:
                for desc in openable_candidates:
                    try:
                        dev = desc.OpenDevice(ids_peak.DeviceAccessType_Control)
                        nm = dev.RemoteDevice().NodeMaps()[0]
                        dev_ip = None
                        if nm.HasNode("GevCurrentIPAddress"):
                            ip_val = nm.FindNode("GevCurrentIPAddress").Value()
                            dev_ip = socket.inet_ntoa(struct.pack("!I", ip_val))
                        elif nm.HasNode("GevDeviceIPAddress"):
                            ip_val = nm.FindNode("GevDeviceIPAddress").Value()
                            dev_ip = socket.inet_ntoa(struct.pack("!I", ip_val))

                        logger.info(f"[{self.cam_name}] Prüfe Kamera S/N {desc.SerialNumber()}: IP={dev_ip}")
                        if dev_ip == self.camera_ip:
                            matched_desc = desc
                            matched_device = dev
                            matched_nodemap = nm
                            logger.info(f"[{self.cam_name}] Exact IP Match! IP {dev_ip} (S/N: {desc.SerialNumber()})")
                            break
                        else:
                            del dev
                    except Exception as e_open:
                        logger.warning(f"[{self.cam_name}] Fehler beim IP-Check der Kamera S/N {desc.SerialNumber()}: {e_open}")

            # 2. Stufe (Fallback): Wenn keine exakte IP gefunden, wähle erste freie IDS-Kamera (überspringe reservierte IPs)
            if matched_device is None and openable_candidates:
                logger.warning(f"[{self.cam_name}] Keine Kamera mit exakter IP '{self.camera_ip}' gefunden. Suche freie Fallback IDS-Kamera...")
                for desc in openable_candidates:
                    try:
                        dev = desc.OpenDevice(ids_peak.DeviceAccessType_Control)
                        nm = dev.RemoteDevice().NodeMaps()[0]
                        dev_ip = None
                        if nm.HasNode("GevCurrentIPAddress"):
                            ip_val = nm.FindNode("GevCurrentIPAddress").Value()
                            dev_ip = socket.inet_ntoa(struct.pack("!I", ip_val))
                        elif nm.HasNode("GevDeviceIPAddress"):
                            ip_val = nm.FindNode("GevDeviceIPAddress").Value()
                            dev_ip = socket.inet_ntoa(struct.pack("!I", ip_val))

                        if dev_ip and dev_ip in self.other_configured_ips:
                            logger.info(f"[{self.cam_name}] Überspringe Kamera S/N {desc.SerialNumber()} (IP {dev_ip}), da sie für eine andere Kamera reserviert ist.")
                            del dev
                            continue

                        matched_desc = desc
                        matched_device = dev
                        matched_nodemap = nm
                        logger.info(f"[{self.cam_name}] Fallback Kamera-Match erfolgreich! S/N: {desc.SerialNumber()} ({desc.ModelName()})")
                        break
                    except Exception as e_fb:
                        logger.warning(f"[{self.cam_name}] Fallback Open fehlgeschlagen für S/N {desc.SerialNumber()}: {e_fb}")

            if matched_device is None:
                logger.error(f"[{self.cam_name}] Keine benutzbare IDS-Kamera im Netzwerk/System verfügbar!")
                return False

            self._device = matched_device
            self._nodemap = matched_nodemap
            self.model_name = matched_desc.ModelName()
            self.serial = matched_desc.SerialNumber()

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
            logger.info(f"[{self.cam_name}] Kamera erfolgreich gestartet. S/N: {self.serial}, Model: {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"[{self.cam_name}] Start fehlgeschlagen: {e}")
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
                    try:
                        self._ds.QueueBuffer(buffer)
                    except Exception:
                        pass
            except Exception:
                if self.running:
                    time.sleep(0.01)

    def get_frame(self) -> np.ndarray | None:
        with self._lock:
            return self.frame.copy() if self.frame is not None else None

    def set_exposure(self, exposure_us: float):
        try:
            node = self._nodemap.FindNode("ExposureTime")
            val = min(max(exposure_us, node.Minimum()), node.Maximum())
            node.SetValue(val)
            self.exposure_us = val
            logger.info(f"[{self.cam_name}] Belichtungszeit gesetzt auf: {val:.0f} us")
        except Exception as e:
            self.exposure_us = exposure_us
            logger.warning(f"[{self.cam_name}] Belichtungszeit Fehler: {e}")

    def set_gain(self, gain: float):
        try:
            node = self._nodemap.FindNode("Gain")
            val = min(max(gain, node.Minimum()), node.Maximum())
            node.SetValue(val)
            self.gain = val
            logger.info(f"[{self.cam_name}] Gain gesetzt auf: {val:.2f}")
        except Exception as e:
            self.gain = gain
            logger.warning(f"[{self.cam_name}] Gain Fehler: {e}")

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
        logger.info(f"[{self.cam_name}] Kamera-Stream geschlossen.")


class FallbackFrameGrabber:
    """Fallback Grabber falls keine physische IDS-Kamera angeschlossen ist (Offline / Simulation)."""
    def __init__(self, cam_name: str = "Demo"):
        self.cam_name = cam_name
        self.running = False
        self.model_name = "Fallback-Simulation"
        self.serial = "OFFLINE-0000"
        self.frame = None
        self.exposure_us = 6000.0
        self.gain = 2.0
        self.last_frame_time = time.time()
        self._thread = None
        self._lock = threading.Lock()

    def start(self) -> bool:
        self.running = True
        self._thread = threading.Thread(target=self._sim_loop, daemon=True)
        self._thread.start()
        logger.info(f"[{self.cam_name}] Fallback-Frame-Grabber aktiv (TCP Server lauscht auf Anfragen).")
        return True

    def _sim_loop(self):
        test_img = None
        td_dir = "training_data"
        if os.path.exists(td_dir):
            imgs = [f for f in os.listdir(td_dir) if f.lower().endswith(('.jpg', '.png'))]
            if imgs:
                test_img = cv2.imread(os.path.join(td_dir, imgs[0]))

        if test_img is None:
            test_img = np.zeros((1080, 1440, 3), dtype=np.uint8)
            cv2.putText(test_img, f"DATA DETECTOR OFFLINE STREAM: {self.cam_name}", (100, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.putText(test_img, "Keine physische IDS Kamera verbunden", (100, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        while self.running:
            with self._lock:
                self.frame = test_img.copy()
                self.last_frame_time = time.time()
            time.sleep(0.1)

    def get_frame(self) -> np.ndarray | None:
        with self._lock:
            return self.frame.copy() if self.frame is not None else None

    def set_exposure(self, exposure_us: float):
        self.exposure_us = exposure_us

    def set_gain(self, gain: float):
        self.gain = gain

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        logger.info(f"[{self.cam_name}] Fallback-Grabber gestoppt.")


def _check_dual_presence(boxes, dmx_min_conf=0.70, txt_min_conf=0.35, max_gap_ratio=2.5) -> bool:
    """
    Prüft VOR dem Auslösen des Auto-Scans, ob mindestens eine Klasse (DataMatrix oder Text)
    im Bild vorhanden ist.
    """
    if boxes is None or len(boxes) == 0:
        return False

    for b in boxes:
        cls_id = int(b.cls[0].cpu().item() if hasattr(b.cls[0], 'cpu') else b.cls[0])
        conf = float(b.conf[0].cpu().item() if hasattr(b.conf[0], 'cpu') else b.conf[0])
        if cls_id == 0 and conf >= dmx_min_conf:
            return True
        elif cls_id == 1 and conf >= txt_min_conf:
            return True

    return False



class CameraService:
    """Service für eine einzelne Kamera inklusive KI-Auswertung und TCP-Server."""
    def __init__(self, cam_cfg: dict, shared_yolo_model, master_log_dir: str = r"U:\Temp\DataMatrixReader.logFiles", is_2class: bool = False):
        self.cfg = cam_cfg
        self.cam_id = cam_cfg.get("id", "cam")
        self.cam_name = cam_cfg.get("name", self.cam_id)
        self.ip = cam_cfg.get("camera_ip", "")
        self.port = int(cam_cfg.get("port", 9500))
        self.model = shared_yolo_model
        self._is_2class = is_2class
        self._scan_counter = 0
        self._tcp_scan_token = 0
        self._tcp_scan_lock = threading.Lock()

        # Dedicated Scan Logger per Camera im dynamisch aufgelösten Log-Pfad
        if ScanLogger is not None:
            resolved_base = resolve_log_directory(master_log_dir)
            cam_log_dir = os.path.join(resolved_base, self.cam_id)
            self.scan_logger = ScanLogger(log_dir=cam_log_dir)
        else:
            self.scan_logger = None

        # Active Learning Setup per Camera
        self._auto_train_dir = os.path.join("auto_training_data", self.cam_id)
        self._auto_train_max = 1000
        os.makedirs(os.path.join(self._auto_train_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self._auto_train_dir, "labels"), exist_ok=True)
        self._auto_train_count = len(os.listdir(os.path.join(self._auto_train_dir, "images")))

    def start_camera(self, all_cams_cfg: list[dict] | None = None) -> bool:
        other_ips = []
        if all_cams_cfg:
            for c in all_cams_cfg:
                if c.get("id") != self.cam_id and c.get("camera_ip"):
                    other_ips.append(c.get("camera_ip"))

        self.grabber = IDSFrameGrabber(camera_ip=self.ip, cam_name=self.cam_name, other_configured_ips=other_ips)
        if not self.grabber.start():
            logger.warning(f"[{self.cam_name}] IDS-Kamera konnte nicht gestartet werden → Starte Fallback-Grabber (TCP Server bleibt auf Port {self.port} aktiv).")
            self.grabber = FallbackFrameGrabber(cam_name=self.cam_name)
        exp_val = float(self.cfg.get("last_exposure", 4.0))
        gain_val = float(self.cfg.get("last_gain", 2.0))
        self.grabber.set_exposure(exp_val * 1000.0)
        self.grabber.set_gain(gain_val)
        return True

    def process_scan(self, token_id: int = 0) -> str | None:
        """Führt eine Einzel-Auswertung aus. Kann storniert werden falls ein neuer Trigger eintrifft."""
        cancellation_check = lambda: (token_id > 0 and self._tcp_scan_token != token_id)

        start_time = time.time()
        self._scan_counter += 1

        scan_snapshot = None
        if self.grabber and getattr(self.grabber, "running", False):
            scan_snapshot = self.grabber.get_frame()

        if scan_snapshot is None:
            logger.error(f"[{self.cam_name}] Konnte kein Bild von der Kamera abrufen!")
            return "ERROR_NO_FRAME"

        detection_box = None
        detection_conf = 0.0
        scan_frame = scan_snapshot
        label_detected = False
        crop_size = None

        t_yolo_start = time.time()
        yolo_detections = []
        use_2class = False
        try:
            if self.model:
                results = self.model(scan_snapshot, verbose=False)
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
                        label_detected = True
                        detection_conf = max(d["conf"] for d in yolo_detections) if yolo_detections else 0.0
                    elif yolo_detections:
                        best_det = max(yolo_detections, key=lambda d: d["conf"])
                        detection_conf = best_det["conf"]
                        detection_box = best_det["box"]
                        label_detected = True
                        scan_frame = scanner.deskew_crop(scan_snapshot, detection_box, padding=60)
                        crop_size = [scan_frame.shape[1], scan_frame.shape[0]]
        except Exception as e:
            logger.error(f"[{self.cam_name}] Fehler bei KI-Auswertung: {e}")
        t_yolo_ms = int((time.time() - t_yolo_start) * 1000)

        # Abbrechen & Verwerfen falls ein neuer Trigger eingetroffen ist
        if cancellation_check():
            logger.warning(f"[{self.cam_name}] Scan (Token {token_id}) VOR Auswertung abgebrochen & VERWORFEN!")
            return None

        t_scan_start = time.time()
        if use_2class:
            result = scanner.scan_2class(scan_snapshot, yolo_detections, cancellation_check=cancellation_check)
        else:
            result = scanner.scan(scan_frame, cancellation_check=cancellation_check)
        t_scan_ms = int((time.time() - t_scan_start) * 1000)

        # Abbrechen & Verwerfen falls während der Auswertung ein neuer Trigger eingetroffen ist
        if cancellation_check() or result.get("cancelled"):
            logger.warning(f"[{self.cam_name}] Scan (Token {token_id}) NACH Auswertung VERWORFEN (neuer Trigger).")
            return None

        duration_ms = int((time.time() - start_time) * 1000)
        result["duration_ms"] = duration_ms
        logger.info(f"[{self.cam_name}] Scan fertig ({duration_ms}ms): {result}")

        # Logging (nur für gültige, nicht stornierte Scans)
        if self.scan_logger:
            exp_us = getattr(self.grabber, "exposure_us", float(self.cfg.get("last_exposure", 6.0)) * 1000.0) if self.grabber else float(self.cfg.get("last_exposure", 6.0)) * 1000.0
            gain_val = getattr(self.grabber, "gain", float(self.cfg.get("last_gain", 2.0))) if self.grabber else float(self.cfg.get("last_gain", 2.0))
            self.scan_logger.log_scan(
                scan_result=result,
                frame=scan_snapshot,
                timing={"total_ms": duration_ms, "yolo_ms": t_yolo_ms, "scan_ms": t_scan_ms},
                detection_info={
                    "yolo_conf": detection_conf,
                    "crop_size": crop_size,
                    "label_detected": label_detected,
                    "2class_mode": use_2class,
                    "detection_count": len(yolo_detections),
                },
                meta={
                    "camera_model": self.grabber.model_name if self.grabber else "",
                    "camera_serial": self.grabber.serial if self.grabber else "",
                    "exposure_us": exp_us,
                    "gain": gain_val,
                    "port": self.port
                }
            )

        if result["success"]:
            return result["result"]
        else:
            return "ERROR"

    def run_tcp_server(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('0.0.0.0', self.port))
        server.listen(5)
        logger.info(f"[{self.cam_name}] TCP SERVER LAUSCHT AUF PORT {self.port}")

        while True:
            try:
                conn, addr = server.accept()
                conn.settimeout(1.0)
                logger.info(f"[{self.cam_name}] Client verbunden: {addr}")
                try:
                    while True:
                        try:
                            data = conn.recv(1024)
                            if not data:
                                break
                            if b"+" in data:
                                with self._tcp_scan_lock:
                                    self._tcp_scan_token += 1
                                    my_token = self._tcp_scan_token
                                logger.info(f"[{self.cam_name}] Trigger '+' (Token {my_token}) empfangen. Starte Auswertung...")
                                code = self.process_scan(token_id=my_token)
                                if code is not None:
                                    response_bytes = b"\x02" + code.encode("utf-8") + b"\r\n\x04"
                                    conn.sendall(response_bytes)
                                else:
                                    logger.info(f"[{self.cam_name}] Scan (Token {my_token}) verworfen — keine Antwort gesendet.")
                            else:
                                conn.sendall(b"\x02ERROR_UNKNOWN_COMMAND\r\n\x04")
                        except socket.timeout:
                            continue
                except Exception as e:
                    logger.error(f"[{self.cam_name}] Kommunikationsfehler: {e}")
                finally:
                    conn.close()
            except Exception as e:
                logger.error(f"[{self.cam_name}] Server-Loop beendet: {e}")
                break
        try:
            server.close()
        except Exception:
            pass

    def run_auto_scan_loop(self):
        presence_state = "EMPTY"
        presence_counter = 0
        absence_counter = 0
        logger.info(f"[{self.cam_name}] Auto-Scan Präsenzerkennung aktiv (DMX oder Text erkannt)!")
        
        while True:
            try:
                time.sleep(0.3)
                if not self.grabber or not getattr(self.grabber, "running", False):
                    continue
                frame = self.grabber.get_frame()
                if frame is None:
                    continue
                
                has_presence = False
                if self.model:
                    results = self.model(frame, verbose=False)
                    if results and len(results[0].boxes) > 0:
                        has_presence = _check_dual_presence(results[0].boxes)
                
                if has_presence:
                    absence_counter = 0
                    if presence_state == "EMPTY":
                        presence_counter += 1
                        if presence_counter >= 2:
                            presence_state = "SCANNED"
                            logger.info(f"[{self.cam_name}] Auto-Scan getriggert (DataMatrix oder Text erkannt)!")
                            self.process_scan()
                else:
                    presence_counter = 0
                    absence_counter += 1
                    if absence_counter >= 3:
                        presence_state = "EMPTY"
            except Exception as e:
                logger.error(f"[{self.cam_name}] Auto-Scan Fehler: {e}")
                time.sleep(1.0)

    def stop(self):
        if self.grabber:
            self.grabber.stop()


def main():
    print(f"==================================================")
    print(f"    DATA DETECTOR MULTI-CAMERA SERVER (v4.0)")
    print(f"    Konfigurationsdatei: {CONFIG_FILE}")
    print(f"==================================================\n")

    cams_cfg, master_log_dir = load_master_config(CONFIG_FILE)
    logger.info(f"Master Log-Verzeichnis: {master_log_dir}")
    logger.info(f"Geladene Kameras ({len(cams_cfg)}): {[c.get('name') for c in cams_cfg]}")

    # 1) IDS SDK global initialisieren
    if IDS_AVAILABLE:
        try:
            ids_peak.Library.Initialize()
        except Exception as e:
            logger.error(f"IDS SDK Initialisierung fehlgeschlagen: {e}")

    # 2) YOLO Modell EINMALIG im Hauptthread laden (bevorzugt 2-Klassen-Modell)
    shared_yolo_model = None
    _is_2class_model = False
    try:
        from ultralytics import YOLO as _YOLO
        app_dir = os.path.dirname(os.path.abspath(__file__))
        # Bevorzugt: 2-Klassen-Modell (datamatrix + text)
        model_path_2class = os.path.join(app_dir, "runs", "detect", "training_runs_v2", "horde_2class", "weights", "best.pt")
        # Fallback: 1-Klassen-Modell (Horde)
        model_path_1class = os.path.join(app_dir, "runs", "detect", "training_runs", "horde_model", "weights", "best.pt")
        if os.path.exists(model_path_2class):
            shared_yolo_model = _YOLO(model_path_2class)
            _is_2class_model = True
            logger.info(f"2-Klassen YOLO Modell geladen (datamatrix+text): {model_path_2class}")
        elif os.path.exists(model_path_1class):
            shared_yolo_model = _YOLO(model_path_1class)
            logger.info(f"1-Klassen YOLO Modell geladen (Horde): {model_path_1class}")
        else:
            base_path = os.path.join(app_dir, "yolov10n.pt")
            shared_yolo_model = _YOLO(base_path)
            logger.warning(f"Standard YOLO Modell geladen: {base_path}")
    except Exception as e:
        logger.error(f"YOLO konnte nicht geladen werden: {e}")

    # 3) Services für alle Kameras initialisieren und starten
    services = []
    server_threads = []

    for cam_cfg in cams_cfg:
        srv = CameraService(cam_cfg, shared_yolo_model, master_log_dir=master_log_dir, is_2class=_is_2class_model)
        if srv.start_camera(all_cams_cfg=cams_cfg):
            services.append(srv)
            t = threading.Thread(target=srv.run_tcp_server, daemon=True)
            t.start()
            server_threads.append(t)
            if cam_cfg.get("auto_scan", False):
                t_auto = threading.Thread(target=srv.run_auto_scan_loop, daemon=True)
                t_auto.start()
        else:
            logger.error(f"Kamera '{cam_cfg.get('name')}' ({cam_cfg.get('camera_ip')}) konnte nicht gestartet werden!")

    if not services:
        logger.error("Keine Kamera erfolgreich gestartet. Programm wird beendet.")
        if IDS_AVAILABLE:
            ids_peak.Library.Close()
        sys.exit(1)

    logger.info("Multi-Kamera System bereit und lauscht auf allen konfigurierten Ports!")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Beende Multi-Kamera Server...")
    finally:
        for srv in services:
            srv.stop()
        if IDS_AVAILABLE:
            try:
                ids_peak.Library.Close()
            except Exception:
                pass
        logger.info("Alle Systeme sauber heruntergefahren.")


if __name__ == "__main__":
    main()
