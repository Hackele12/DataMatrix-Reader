"""YOLO-Etikettdetektion: Modellauswahl und Umwandlung der Detektionen für scanner.scan_2class()."""

import logging
import os
import sys

logger = logging.getLogger(__name__)

APP_DIR = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))

# Trainierte Modelle in Prioritätsreihenfolge: 2-Klassen (datamatrix + text), dann 1-Klassen (Etikett)
MODEL_2CLASS = os.path.join("runs", "detect", "training_runs_v2", "horde_2class", "weights", "best.pt")
MODEL_1CLASS = os.path.join("runs", "detect", "training_runs", "horde_model", "weights", "best.pt")
BASE_MODEL = "yolov10n.pt"

# Vorhersage-Schwelle aller Aufrufer (Anzeige, Scan, Benchmark)
PREDICT_CONF = 0.15


def find_model_path(app_dir: str = APP_DIR) -> tuple[str, bool]:
    """Bestes verfügbares Modell → (Pfad, is_2class). Fallback ist das untrainierte Basismodell."""
    path_2class = os.path.join(app_dir, MODEL_2CLASS)
    if os.path.exists(path_2class):
        return path_2class, True
    path_1class = os.path.join(app_dir, MODEL_1CLASS)
    if os.path.exists(path_1class):
        return path_1class, False
    base_dir = sys._MEIPASS if getattr(sys, "frozen", False) else app_dir
    return os.path.join(base_dir, BASE_MODEL), False


def load_model(app_dir: str = APP_DIR):
    """Lädt das beste verfügbare YOLO-Modell → (Modell, is_2class)."""
    from ultralytics import YOLO

    path, is_2class = find_model_path(app_dir)
    if path.endswith(BASE_MODEL):
        logger.warning(f"Kein trainiertes Modell gefunden, nutze Basismodell: {path}")
    else:
        logger.info(f"{'2-Klassen' if is_2class else '1-Klassen'} YOLO-Modell geladen: {path}")
    return YOLO(path), is_2class


def extract_detections(result, min_conf: float | None = None) -> list[dict]:
    """Ultralytics-Ergebnis → [{"cls", "conf", "box": (x1, y1, x2, y2)}]; optional nur conf > min_conf."""
    detections = []
    for box in result.boxes:
        conf = float(box.conf[0])
        if min_conf is not None and conf <= min_conf:
            continue
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
        detections.append({"cls": int(box.cls[0]), "conf": conf, "box": (x1, y1, x2, y2)})
    return detections


def has_label_presence(boxes, dmx_min_conf: float = 0.70, txt_min_conf: float = 0.35) -> bool:
    """Auto-Scan-Auslöser: mindestens eine sichere DataMatrix- (Klasse 0) oder Text-Detektion (Klasse 1)."""
    if boxes is None or len(boxes) == 0:
        return False
    for b in boxes:
        cls_id = int(b.cls[0].cpu().item() if hasattr(b.cls[0], 'cpu') else b.cls[0])
        conf = float(b.conf[0].cpu().item() if hasattr(b.conf[0], 'cpu') else b.conf[0])
        if (cls_id == 0 and conf >= dmx_min_conf) or (cls_id == 1 and conf >= txt_min_conf):
            return True
    return False
