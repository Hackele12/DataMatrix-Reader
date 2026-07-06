"""
vision_app_v4_network.py — Headless Industrial TCP/IP Scanner (v4.0)

Diese Version läuft komplett ohne grafische Benutzeroberfläche (GUI) im Hintergrund.
Sie ist für den produktiven 24/7-Dauerbetrieb optimiert:
1. Startet die IDS-Peak-Kamera und hält sie betriebsbereit.
2. Lädt das lokale YOLO-KI-Modell zur Etiketten-Erkennung.
3. Lauscht als TCP-Server auf Port 9500.
4. Bei Empfang von "+" wird ein Foto geschossen, per KI + Dual-Validation
   (DataMatrix + EasyOCR) ausgewertet und das Ergebnis mit Carriage Return (\r) zurückgesendet.

"""

import os
import sys
import json
import ctypes
import logging
import threading
import time
import socket
from logging.handlers import RotatingFileHandler
import cv2
import numpy as np
import warnings

# PyTorch/User-Warnings auf CPU unterdrücken für sauberere Konsolenausgabe
warnings.filterwarnings("ignore", category=UserWarning)


# --- Config & CLI Arguments laden ---
CONFIG_FILE = "config.json"
if len(sys.argv) > 1:
    CONFIG_FILE = sys.argv[1]

# Default-Werte
PORT = 9500
log_file = "vision_network_v4.log"

try:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            cfg_temp = json.load(f)
            PORT = cfg_temp.get("port", PORT)
            log_file = cfg_temp.get("log_file", f"vision_network_{PORT}.log")
except Exception as e:
    print(f"[WARN] Fehler beim Vorab-Laden der Config '{CONFIG_FILE}': {e}")

# --- Logging einrichten ---
logger = logging.getLogger("VisionNetworkApp")
logger.setLevel(logging.INFO)

# Formatter für Log-Meldungen
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Konsole Handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Datei Handler (rotierend, maximal 5 Dateien à 5 MB)
file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5, encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# --- Windows Crash-Dialog unterdrücken (24/7 Betrieb) ---
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

YOLO = None


def _load_config() -> dict:
    """Lädt die Kameraeinstellungen."""
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                cfg = json.load(f)
                logger.info(f"Konfiguration geladen: {cfg}")
                return cfg
    except Exception as e:
        logger.error(f"Config laden fehlgeschlagen: {e}")
    return {"last_exposure": 4.0, "last_gain": 2.0}


class IDSFrameGrabber:
    """
    IDS peak SDK Frame-Grabber:
    Holt im Hintergrund-Thread kontinuierlich Live-Bilder der Kamera.
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
        self.last_frame_time = time.time()

    def start(self) -> bool:
        if not IDS_AVAILABLE:
            logger.error("Kamera-Start unmöglich: IDS peak SDK fehlt.")
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
            if dm.Devices().empty():
                logger.error("Keine IDS-Kamera im Netzwerk/USB gefunden!")
                ids_peak.Library.Close()
                return False

            desc = dm.Devices()[0]
            self.model_name = desc.ModelName()
            self.serial = desc.SerialNumber()
            logger.info(f"Verbinde mit IDS Kamera: {self.model_name} (S/N: {self.serial})")

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
            logger.info("IDS Kamera erfolgreich gestartet und erfasst Live-Bilder.")
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
            logger.info(f"Belichtungszeit gesetzt auf: {val:.0f} us")
        except Exception as e:
            logger.warning(f"Belichtungszeit konnte nicht gesetzt werden: {e}")

    def set_gain(self, gain: float):
        try:
            node = self._nodemap.FindNode("Gain")
            val = min(max(gain, node.Minimum()), node.Maximum())
            node.SetValue(val)
            logger.info(f"Gain gesetzt auf: {val:.2f}")
        except Exception as e:
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
        logger.info("IDS Kamera geschlossen.")

    def _cleanup_partial(self):
        try:
            ids_peak.Library.Close()
        except Exception:
            pass


class HeadlessScanner:
    def __init__(self):
        self.config = _load_config()
        self.grabber = None
        self.model = None
        self._scan_counter = 0
        
        # Active Learning Setup
        self._auto_train_dir = "auto_training_data"
        self._auto_train_max = 1000
        os.makedirs(os.path.join(self._auto_train_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self._auto_train_dir, "labels"), exist_ok=True)
        self._auto_train_count = len(os.listdir(os.path.join(self._auto_train_dir, "images")))
        logger.info(f"Active Learning initialisiert: {self._auto_train_count} Bilder vorhanden.")

    def initialize_system(self) -> bool:
        """Lädt YOLO-Modell und startet die Kamera."""
        logger.info("System-Initialisierung gestartet...")
        
        # 1) YOLO laden
        try:
            from ultralytics import YOLO as _YOLO
            global YOLO
            YOLO = _YOLO
            
            app_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(app_dir, "runs", "detect", "training_runs", "horde_model", "weights", "best.pt")
            
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
                logger.info(f"Trained YOLO Modell geladen: {model_path}")
            else:
                base_model_path = os.path.join(app_dir, "yolov10n.pt")
                self.model = YOLO(base_model_path)
                logger.warning(f"Kein trainiertes Modell gefunden, nutze Standard YOLO: {base_model_path}")
        except Exception as e:
            logger.error(f"YOLO-Modell konnte nicht geladen werden: {e}")
            return False

        # 2) Kamera starten
        self.grabber = IDSFrameGrabber()
        if not self.grabber.start():
            logger.error("Kamera-Verbindung fehlgeschlagen!")
            return False

        # Einstellungen anwenden
        try:
            exp_val = float(self.config.get("last_exposure", 4.0))
            gain_val = float(self.config.get("last_gain", 2.0))
            self.grabber.set_exposure(exp_val * 1000.0) # ms -> us
            self.grabber.set_gain(gain_val)
        except Exception as e:
            logger.warning(f"Fehler beim Anwenden der Kamera-Settings: {e}")

        logger.info("System erfolgreich initialisiert und bereit für Scans!")
        return True

    def process_scan(self) -> str:
        """Führt einen Scan-Vorgang aus und gibt das Ergebnis zurück."""
        start_time = time.time()
        
        # 1) Bild abholen
        scan_snapshot = self.grabber.get_frame()
        if scan_snapshot is None:
            logger.error("Konnte kein Bild von der Kamera abrufen!")
            return "ERROR_NO_FRAME"

        # 2) KI-Detektion (YOLO)
        detection_box = None
        detection_conf = 0.0
        scan_frame = scan_snapshot
        
        try:
            results = self.model(scan_snapshot, verbose=False)
            if results and len(results[0].boxes) > 0:
                best_box = max(results[0].boxes, key=lambda b: float(b.conf[0]))
                detection_conf = float(best_box.conf[0])
                x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                detection_box = (x1, y1, x2, y2)
                
                # Mit Rand ausschneiden
                padding = 20
                fh, fw = scan_snapshot.shape[:2]
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(fw, x2 + padding)
                y2 = min(fh, y2 + padding)
                
                scan_frame = scan_snapshot[y1:y2, x1:x2]
                logger.info(f"KI: Etikett gefunden (Konfidenz: {detection_conf:.2f}). Ausschneiden auf {x2-x1}x{y2-y1}.")
            else:
                logger.warning("KI: Kein Etikett gefunden, scanne komplettes Bild.")
        except Exception as e:
            logger.error(f"Fehler bei KI-Auswertung: {e}. Scanne komplettes Bild.")

        # 3) Dual-Validation ausführen (scanner.py)
        result = scanner.scan(scan_frame)
        duration_ms = int((time.time() - start_time) * 1000)
        logger.info(f"Scan abgeschlossen. Dauer: {duration_ms}ms, Ergebnis: {result}")

        # 4) Active Learning (Auto-Save)
        if result["success"] and detection_box is not None:
            self._scan_counter += 1
            should_save = (0.15 <= detection_conf <= 0.60) or (self._scan_counter % 20 == 0)
            if should_save and self._auto_train_count < self._auto_train_max:
                self._auto_save_training(scan_snapshot, detection_box)

        if result["success"]:
            return result["result"]
        else:
            return "ERROR"

    def _auto_save_training(self, full_frame, box):
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            img_name = f"auto_{timestamp}_{self._auto_train_count}.jpg"
            lbl_name = f"auto_{timestamp}_{self._auto_train_count}.txt"

            img_path = os.path.join(self._auto_train_dir, "images", img_name)
            lbl_path = os.path.join(self._auto_train_dir, "labels", lbl_name)

            cv2.imwrite(img_path, full_frame)

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
        except Exception as e:
            logger.error(f"Active Learning Speichern fehlgeschlagen: {e}")

    def shutdown(self):
        if self.grabber:
            self.grabber.stop()
            logger.info("Kamerasystem heruntergefahren.")


def start_tcp_server(scanner_system: HeadlessScanner):
    """Startet den TCP Server auf Port 9500 und wartet auf Verbindungen."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # SO_REUSEADDR setzen, um "Address already in use" beim Neustart zu vermeiden
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # Binden an alle IPs des Rechners auf Port 9500
    server.bind(('0.0.0.0', PORT))
    server.listen(5)
    logger.info(f"==================================================")
    logger.info(f"TCP SERVER LAUSCHT AUF PORT {PORT}")
    logger.info(f"==================================================")

    while True:
        try:
            conn, addr = server.accept()
            # Setze ein kurzes Timeout (z. B. 1.0 Sekunde) für recv, damit wir regelmäßig
            # auf Server-Beendigung reagieren können, ohne die Verbindung zu blockieren.
            conn.settimeout(1.0)
            logger.info(f"Verbindung hergestellt von Client: {addr}")
            
            try:
                # Schleife für persistente Kommunikation über dieselbe Verbindung
                while True:
                    try:
                        data = conn.recv(1024)
                        if not data:
                            logger.info(f"Client {addr} hat die Verbindung getrennt (EOF).")
                            break
                        
                        logger.info(f"Empfangene Rohdaten: {repr(data)}")
                        
                        # Prüfen, ob das Triggersignal '+' in den empfangenen Daten enthalten ist
                        if b"+" in data:
                            logger.info("Trigger-Signal '+' empfangen. Starte Auswertung...")
                            # Scan durchführen
                            code = scanner_system.process_scan()
                            
                            # Antwort im Protokollformat vorbereiten: <STX>[CODE]<CR><LF><EOT>
                            response_bytes = b"\x02" + code.encode("utf-8") + b"\r\n\x04"
                            logger.info(f"Sende Antwort an Client: {repr(response_bytes)}")
                            
                            conn.sendall(response_bytes)
                        else:
                            logger.warning("Unbekannter Befehl oder falsches Format empfangen.")
                            response_bytes = b"\x02ERROR_UNKNOWN_COMMAND\r\n\x04"
                            conn.sendall(response_bytes)
                    except socket.timeout:
                        # Timeout ist bei einer persistenten Verbindung normal, solange der Client verbunden ist
                        continue
            except Exception as e:
                logger.error(f"Fehler bei der Kommunikation mit {addr}: {e}")
            finally:
                conn.close()
                logger.info(f"Verbindung zu {addr} geschlossen.\n")
                
        except KeyboardInterrupt:
            logger.info("Server wird beendet (KeyboardInterrupt)...")
            break
        except Exception as e:
            logger.error(f"Fehler im accept-Loop des Servers: {e}")
            time.sleep(1)

    try:
        server.close()
    except Exception:
        pass


if __name__ == "__main__":
    print(f"               HEADLESS TCP SCANNER (v4.0)")
    print(f"   Konfiguration: {CONFIG_FILE}")
    print(f"   Logdatei:      {log_file}")
    print(f"   TCP-Port:      {PORT}\n")

    scanner_app = HeadlessScanner()
    if not scanner_app.initialize_system():
        logger.error("Konnte das Kamerasystem oder das Modell nicht initialisieren. Programm beendet.")
        sys.exit(1)

    try:
        start_tcp_server(scanner_app)
    finally:
        logger.info("Räume Ressourcen auf...")
        scanner_app.shutdown()
        logger.info("Programm sauber beendet.")
