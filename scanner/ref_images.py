"""
Pipeline 3: Referenzbild-Abgleich. Vergleicht den DataMatrix-Bereich per Hamming-Distanz
gegen alle vorgenerierten Codes in generated_codes/ (packbits + XOR + Popcount).
"""

import logging
import os
import time
from functools import lru_cache

import cv2
import numpy as np

from . import config
from .image_ops import otsu, rect_kernel, to_gray

logger = logging.getLogger(__name__)

_REF_IMG_SIZE = (100, 100)  # Normgröße für den Vergleich
_POPCOUNT_LUT = np.array([bin(i).count('1') for i in range(256)], dtype=np.int32)


def _ref_img_dir() -> str:
    return os.path.join(config.APP_DIR, 'generated_codes')


def _binarize_01(gray: np.ndarray) -> np.ndarray:
    return (otsu(gray) // 255).astype(np.uint8)  # 0 = schwarz, 1 = weiß


@lru_cache(maxsize=1)
def load_reference_images() -> tuple[list[str], np.ndarray, np.ndarray]:
    """Lädt alle PNG-Referenzbilder einmalig als binarisierte Vektoren → (Codes, Matrix (N, 10000), gepackt (N, 1250))."""
    norm_w, norm_h = _REF_IMG_SIZE
    ref_dir = _ref_img_dir()
    if not os.path.isdir(ref_dir):
        logger.warning(f"Referenzbild-Ordner '{ref_dir}' nicht gefunden. Pipeline 3 deaktiviert.")
        empty = np.empty((0, norm_w * norm_h), dtype=np.uint8)
        return [], empty, np.packbits(empty, axis=1)

    t0 = time.time()
    codes = []
    rows = []
    for fname in sorted(f for f in os.listdir(ref_dir) if f.lower().endswith('.png')):
        img = cv2.imread(os.path.join(ref_dir, fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        resized = cv2.resize(img, (norm_w, norm_h), interpolation=cv2.INTER_AREA)
        rows.append(_binarize_01(resized).flatten())
        codes.append(os.path.splitext(fname)[0])

    matrix = np.array(rows, dtype=np.uint8) if rows else np.empty((0, norm_w * norm_h), dtype=np.uint8)
    logger.info(
        f"Pipeline 3: {len(codes)} Referenzbilder aus '{ref_dir}' geladen "
        f"(Matrix: {matrix.shape}, {time.time() - t0:.2f}s)"
    )
    return codes, matrix, np.packbits(matrix, axis=1)


def extract_dmx_region(frame: np.ndarray) -> list[np.ndarray]:
    """
    Binarisierte 100x100-Varianten des DMX-Bereichs: größte annähernd quadratische Kontur und
    das Gesamtbild, jeweils in allen 4 Rotationen (Ecksortierung ist nicht eindeutig).
    """
    gray = to_gray(frame)
    h, w = gray.shape[:2]
    norm_w, norm_h = _REF_IMG_SIZE
    variants = []

    closed = cv2.morphologyEx(cv2.bitwise_not(otsu(gray)), cv2.MORPH_CLOSE, rect_kernel(15), iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best_contour = None
    best_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 400 or area > (h * w) * 0.70:
            continue
        rect_w, rect_h = cv2.minAreaRect(c)[1]
        if rect_w == 0 or rect_h == 0 or max(rect_w, rect_h) / min(rect_w, rect_h) > 1.6:
            continue
        if area > best_area:
            best_area = area
            best_contour = c

    if best_contour is not None:
        box = np.float32(cv2.boxPoints(cv2.minAreaRect(best_contour)))
        center = box.mean(axis=0)
        sorted_box = box[np.argsort(np.arctan2(box[:, 1] - center[1], box[:, 0] - center[0]))]
        dst = np.float32([[0, 0], [norm_w, 0], [norm_w, norm_h], [0, norm_h]])
        warped = cv2.warpPerspective(gray, cv2.getPerspectiveTransform(sorted_box, dst), (norm_w, norm_h))
        base = _binarize_01(warped)
        variants.extend(np.rot90(base, k) for k in range(4))

    # Fallback: Gesamtbild (falls das Bild bereits ein sauberer DMX-Ausschnitt ist)
    fallback = _binarize_01(cv2.resize(gray, (norm_w, norm_h), interpolation=cv2.INTER_AREA))
    variants.extend(np.rot90(fallback, k) for k in range(4))
    return variants


def scan_reference_image_pipeline(frame: np.ndarray) -> dict:
    """
    Hamming-Abgleich aller Rotationsvarianten gegen die Referenzbilder.

    Returns:
        dict mit status ("matched" | "blocked"), text, confidence, method_detail.
    """
    blocked_result = {"status": "blocked", "text": None, "confidence": 0.0, "method_detail": "RefImg: nicht erkannt"}

    try:
        codes, _, ref_packed = load_reference_images()
        if len(codes) == 0:
            return blocked_result

        variants = extract_dmx_region(frame)
        if not variants:
            logger.debug("Pipeline 3: Keine DMX-Region im Bild gefunden.")
            return blocked_result

        n_pixels = _REF_IMG_SIZE[0] * _REF_IMG_SIZE[1]
        variant_packed = np.packbits(np.array([v.flatten() for v in variants], dtype=np.uint8), axis=1)

        best_score = -1.0
        best_code = None
        second_score = 0.0
        second_code = None
        for packed in variant_packed:
            hamming = np.sum(_POPCOUNT_LUT[np.bitwise_xor(ref_packed, packed)], axis=1)
            scores = 1.0 - (hamming.astype(np.float32) / n_pixels)

            top2_idx = np.argpartition(scores, -2)[-2:]
            top2_sorted = top2_idx[np.argsort(scores[top2_idx])[::-1]]
            best_idx = top2_sorted[0]
            second_idx = top2_sorted[1] if len(top2_sorted) > 1 else best_idx

            if float(scores[best_idx]) > best_score:
                best_score = float(scores[best_idx])
                best_code = codes[best_idx]
                second_score = float(scores[second_idx])
                second_code = codes[second_idx]

            if best_score >= 0.95:
                break

        margin = best_score - second_score
        logger.info(
            f"Pipeline 3 RefImg: Bester='{best_code}' Score={best_score:.3f}, "
            f"Zweiter='{second_code}' Score={second_score:.3f}, Margin={margin:.3f}"
        )

        if best_score >= 0.70 and margin >= 0.02:
            return {
                "status": "matched",
                "text": best_code,
                "confidence": min(1.0, max(0.60, best_score)),
                "method_detail": f"RefImg Hamming-Match (Score={best_score:.3f}, Margin={margin:.3f})",
            }
        return blocked_result

    except Exception as e:
        logger.warning(f"Pipeline 3 RefImg Fehler: {e}")
        return blocked_result
