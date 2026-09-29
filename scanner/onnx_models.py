"""Trainierte KI-Modelle (ONNX Runtime): PACC-Zeichenklassifikator und MicroUNet-DataMatrix-Binarisierer."""

import logging
import os

import cv2
import numpy as np

from . import config
from .config import HORDEN_PATTERN
from .image_ops import to_gray

logger = logging.getLogger(__name__)

_PACC_PREFIX_CLASSES = ['A', 'B', 'P', 'W']
_sessions: dict[str, object] = {}


def _load_session(filename: str, description: str):
    """Lädt eine ONNX-Session einmalig (Lazy-Loading, 1 Thread); None, wenn Datei oder Runtime fehlt."""
    if filename not in _sessions:
        session = None
        try:
            import onnxruntime as ort
            model_path = os.path.join(config.PROJECT_DIR, 'models', filename)
            if os.path.exists(model_path):
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                opts.inter_op_num_threads = 1
                session = ort.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])
                logger.info(f"{description} (ONNX) erfolgreich geladen.")
            else:
                logger.warning(f"{description} ONNX Modell nicht gefunden: {model_path}")
        except Exception as e:
            logger.warning(f"Fehler beim Laden von {description} ONNX: {e}")
        _sessions[filename] = session
    return _sessions[filename]


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def predict_pacc(image: np.ndarray) -> tuple[str | None, float]:
    """Position-Aware Char Classifier (< 3 ms) → (Code, mittlere Konfidenz) oder (None, 0.0)."""
    if not config.USE_PACC:
        return None, 0.0
    session = _load_session('char_classifier.onnx', "PACC Char-Classifier")
    if session is None or image is None or image.size == 0:
        return None, 0.0

    try:
        resized = cv2.resize(to_gray(image), (128, 32), interpolation=cv2.INTER_AREA)
        inp = (resized.astype(np.float32) / 255.0)[np.newaxis, np.newaxis, :, :]
        outputs = session.run(None, {"input": inp})

        probs = [_softmax(out[0]) for out in outputs]
        indices = [int(np.argmax(p)) for p in probs]
        code = _PACC_PREFIX_CLASSES[indices[0]] + ''.join(str(i) for i in indices[1:])
        avg_conf = float(np.mean([p[i] for p, i in zip(probs, indices)]))

        if HORDEN_PATTERN.match(code):
            return code, avg_conf
    except Exception as e:
        logger.debug(f"PACC Inferenz Fehler: {e}")

    return None, 0.0


def unet_binarize(image: np.ndarray) -> np.ndarray | None:
    """MicroUNet-Binarisierung (~8-10 ms): sauberes Schwarz-Weiß-Bild der DataMatrix (uint8 0/255) oder None."""
    session = _load_session('unet_binarizer.onnx', "MicroUNet DataMatrix-Binarisierer")
    if session is None or image is None or image.size == 0:
        return None

    try:
        gray = to_gray(image)
        h, w = gray.shape[:2]
        resized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
        inp = (resized.astype(np.float32) / 255.0)[np.newaxis, np.newaxis, :, :]
        out_map = session.run(None, {"input": inp})[0][0, 0]

        sig_map = 1.0 / (1.0 + np.exp(-out_map))
        bin_128 = np.where(sig_map >= 0.5, 255, 0).astype(np.uint8)
        return cv2.resize(bin_128, (w, h), interpolation=cv2.INTER_NEAREST)
    except Exception as e:
        logger.debug(f"MicroUNet Binarizer Fehler: {e}")
        return None
