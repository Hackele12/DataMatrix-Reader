"""
DataMatrix-Dekodierung: zxing-cpp (inkl. Dot-Peen-Varianten), pylibdmtx-Fallback,
Sichtbarkeitsanalyse und Rahmen-Rekonstruktion aus den inneren 8x8-Modulen.
"""

import logging
from itertools import islice

import cv2
import numpy as np
from PIL import Image as PILImage

from . import config
from .code_format import dmx_text_to_code
from .dmx_codec import frame_grid
from .dmx_grid import (collect_square_candidates, extract_observed_grid, find_label_crop, frame_pattern_ok,
                       generate_synthetic_dmtx, orient_corners, warp_and_sample)
from .image_ops import (clahe, crop_with_margin, disk, otsu, preprocess_etch_denoise, preprocess_faded_contrast,
                        preprocess_niblack, preprocess_ridge_enhancement, preprocess_sauvola, preprocess_tophat,
                        rect_kernel, sharpen, to_gray)
from .onnx_models import unet_binarize
from .results import dmx_blocked

logger = logging.getLogger(__name__)

# --- Lazy-Loading der Decoder-Bibliotheken ---
_zxing_module = None
_zxing_checked = False
_dmtx_available = False
_dmtx_loaded = False


def _zxing():
    """zxing-cpp-Modul oder None (einmalig geprüft)."""
    global _zxing_module, _zxing_checked
    if not _zxing_checked:
        try:
            import zxingcpp
            _zxing_module = zxingcpp
            logger.info("zxing-cpp (C++20 High-Speed Reader) erfolgreich geladen.")
        except ImportError:
            logger.info("zxing-cpp nicht verfügbar — verwende pylibdmtx als Fallback.")
        _zxing_checked = True
    return _zxing_module


def _load_dmtx() -> bool:
    global _dmtx_available, _dmtx_loaded
    if not _dmtx_loaded:
        try:
            import setuptools  # noqa: F401 (distutils-Kompatibilität für pylibdmtx ab Python 3.12)
            from pylibdmtx.pylibdmtx import decode  # noqa: F401
            _dmtx_available = True
            logger.info("pylibdmtx erfolgreich geladen.")
        except Exception as e:
            logger.warning(f"pylibdmtx konnte nicht geladen werden: {e}. DataMatrix-Scan deaktiviert.")
            _dmtx_available = False
        _dmtx_loaded = True
    return _dmtx_available


# --------------------------------------------------------------------------- #
#  zxing-cpp                                                                   #
# --------------------------------------------------------------------------- #

def try_zxing_dmtx(image: np.ndarray) -> str | None:
    """zxing-cpp-Dekodierung (< 5 ms) mit try_rotate/try_invert und drei Binarisierern → Code oder None."""
    zx = _zxing()
    if zx is None:
        return None
    try:
        gray = np.ascontiguousarray(to_gray(image))
        for binarizer in (zx.Binarizer.LocalAverage, zx.Binarizer.GlobalHistogram, zx.Binarizer.FixedThreshold):
            res = zx.read_barcode(gray, formats=zx.BarcodeFormat.DataMatrix, try_rotate=True,
                                  try_downscale=True, try_invert=True, binarizer=binarizer)
            if res and res.valid and res.text:
                candidate = dmx_text_to_code(res.text.strip())
                if candidate is not None:
                    logger.info(f"zxing-cpp DataMatrix erkannt (Binarisierer {binarizer}): '{candidate}'")
                    return candidate
    except Exception as e:
        logger.debug(f"zxing-cpp Fehler: {e}")
    return None


def _zxing_decode_strict(zx, gray: np.ndarray) -> str | None:
    """Ein zxing-cpp-Versuch mit zwei Binarisierern und strikter Formatprüfung."""
    img = np.ascontiguousarray(gray)
    for binarizer in (zx.Binarizer.LocalAverage, zx.Binarizer.GlobalHistogram):
        res = zx.read_barcode(img, formats=zx.BarcodeFormat.DataMatrix, try_rotate=True,
                              try_downscale=True, binarizer=binarizer)
        if res and res.valid:
            code = dmx_text_to_code(res.text)
            if code:
                return code
    return None


def _dotpeen_variants(gray: np.ndarray):
    """Vorverarbeitungen für Punkt-DataMatrix (genadelt/gelasert), die Einzelpunkte zu vollen Modulen verbinden."""
    for scale in (1.0, 0.75, 0.5, 0.35):
        base = gray if scale == 1.0 else cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        yield f"s{scale}", base
        for k in (3, 5, 7):
            yield f"s{scale}_median{k}", cv2.medianBlur(base, k)
            yield f"s{scale}_erode{k}", cv2.erode(base, disk(k))
        yield f"s{scale}_clahe_median5", cv2.medianBlur(clahe(base, 3.0, tile=4), 5)
    for k in (7, 9, 11, 13):
        merged = cv2.dilate(cv2.erode(gray, disk(k)), disk(k // 2))
        for scale in (0.5, 0.35):
            small = cv2.resize(merged, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            yield f"merge{k}_s{scale}", small
            yield f"merge{k}_s{scale}_adaptive", cv2.adaptiveThreshold(
                small, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 5)


def _fullframe_variants(gray: np.ndarray):
    def half(img):
        return cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    yield "full_raw", gray
    yield "full_s0.5", half(gray)
    for k in (5, 9):
        yield f"full_median{k}_s0.5", half(cv2.medianBlur(gray, k))
    for k in (5, 7):
        yield f"full_erode{k}_s0.5", half(cv2.erode(gray, disk(k)))


def locate_dmx_regions(gray: np.ndarray, max_regions: int = 3) -> list[tuple[int, int, int, int]]:
    """YOLO-unabhängige DMX-Suche: kompakte, annähernd quadratische Häufung dunkler Punkte."""
    scale = 360.0 / gray.shape[1]
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    blackhat = cv2.morphologyEx(small, cv2.MORPH_BLACKHAT, rect_kernel(21))
    mask = cv2.morphologyEx(otsu(blackhat), cv2.MORPH_CLOSE, rect_kernel(7))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scored = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if not (15 <= bw <= 120 and 15 <= bh <= 120 and 0.7 < bw / bh < 1.4):
            continue
        fill = cv2.contourArea(contour) / float(bw * bh)
        if fill > 0.5:
            box = (int(x / scale), int(y / scale), int((x + bw) / scale), int((y + bh) / scale))
            scored.append((fill * bw * bh, box))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [box for _, box in scored[:max_regions]]


def zxing_dmx_stream(frame: np.ndarray, dmx_boxes=()):
    """
    Alle zxing-Versuche in fester Reihenfolge (lazy) → (Detail, Bild): Rohausschnitte der YOLO-Boxen und
    gefundenen DMX-Regionen, Dot-Peen-Varianten je Region, zuletzt Varianten des Gesamtbilds.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    regions = [("yolo", crop_with_margin(gray, box, 0.15)) for box in list(dmx_boxes)[:2]]
    regions += [("locator", crop_with_margin(gray, box, 0.5)) for box in locate_dmx_regions(gray)]
    regions = [(name, crop) for name, crop in regions if min(crop.shape[:2]) >= 20]
    for name, crop in regions:
        yield f"{name}_raw", crop
    for name, crop in regions:
        for variant_name, variant in _dotpeen_variants(crop):
            yield f"{name}_{variant_name}", variant
    for variant_name, variant in _fullframe_variants(gray):
        yield variant_name, variant


def decode_zxing_stream(stream, limit: int | None = None) -> tuple[str | None, str | None]:
    """Prüft die nächsten `limit` Versuche eines zxing_dmx_stream (alle bei None) → (Code, Detail)."""
    zx = _zxing()
    if zx is None:
        return None, None
    try:
        for detail, image in islice(stream, limit):
            code = _zxing_decode_strict(zx, image)
            if code:
                return code, detail
    except Exception as e:
        logger.warning(f"Dot-Peen DMX-Dekodierung Fehler: {e}")
    return None, None


def decode_dmx_dotpeen(frame: np.ndarray, dmx_boxes=()) -> tuple[str | None, str | None]:
    """Alle zxing-Versuche von zxing_dmx_stream in einem Durchlauf → (Code, Detail)."""
    if frame is None or frame.size == 0:
        return None, None
    return decode_zxing_stream(zxing_dmx_stream(frame, dmx_boxes))


# --------------------------------------------------------------------------- #
#  zxing-cpp + pylibdmtx                                                       #
# --------------------------------------------------------------------------- #

def try_decode_dmtx(pil_img, timeout_ms: int = 250) -> str | None:
    """
    Dekodiert ein PIL-Bild per zxing-cpp, sonst pylibdmtx. pylibdmtx wird auf binarisierten Bildern
    (≤ 10 Grauwerte) übersprungen, da libdmtx dort C-Heap/Stack-Faults verursacht.
    """
    if pil_img is None:
        return None
    try:
        img_arr = np.array(pil_img)
        if img_arr.size == 0:
            return None

        zx_res = try_zxing_dmtx(img_arr)
        if zx_res is not None:
            return zx_res

        if len(np.unique(img_arr)) <= 10 or not _load_dmtx():
            return None

        from pylibdmtx.pylibdmtx import decode
        decoded = decode(pil_img, timeout=timeout_ms)
        if decoded:
            return dmx_text_to_code(decoded[0].data.decode("utf-8", errors="ignore").strip())
    except Exception as e:
        logger.debug(f"DataMatrix-Dekodierung Fehler: {e}")
    return None


_BASIC_ZXING_CLAHE_LIMITS = (4.0, 10.0)


def _decode_basic_zxing(frame: np.ndarray) -> tuple[str | None, str | None]:
    """zxing-cpp auf dem Rohbild und mit CLAHE-Kontrastverstärkung → (Code, method_detail)."""
    code = try_zxing_dmtx(frame)
    if code is not None:
        logger.info(f"DMX Pipeline: zxing-cpp Fast-Path erfolgreich: '{code}'")
        return code, "DataMatrix direkt dekodiert (zxing-cpp)"

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    for clip_limit in _BASIC_ZXING_CLAHE_LIMITS:
        code = try_zxing_dmtx(clahe(gray, clip_limit))
        if code is not None:
            logger.info(f"DMX Pipeline: zxing-cpp + CLAHE {clip_limit} erfolgreich: '{code}'")
            return code, f"DataMatrix dekodiert (zxing-cpp + CLAHE {clip_limit})"
    return None, None


def _zxing_preprocessing_cascade(gray: np.ndarray):
    """
    Weitere zxing-cpp-Vorverarbeitungen (lazy, feste Reihenfolge). Rohbild und CLAHE 4/10 hat
    _decode_basic_zxing() bereits erfolglos versucht.
    """
    yield "MicroUNet Binarisierer", lambda: unet_binarize(gray)
    yield "TopHat", lambda: preprocess_tophat(gray)
    yield "FadedBoost", lambda: preprocess_faded_contrast(gray)
    yield "EtchDenoise", lambda: preprocess_etch_denoise(gray)
    yield "RidgeBoost", lambda: preprocess_ridge_enhancement(gray)
    yield "Sauvola W11", lambda: preprocess_sauvola(gray, window_size=11, k=0.2)
    yield "Niblack", lambda: preprocess_niblack(gray, window_size=21, k=-0.2)
    yield "Sauvola", lambda: preprocess_sauvola(gray)


def _label_filter_variants(label_crop: np.ndarray):
    """Filter-Varianten des Etiketts für den letzten pylibdmtx-Versuch (lazy)."""
    def otsu_erode():
        return cv2.erode(otsu(label_crop), rect_kernel(2), iterations=1)

    def adaptive_erode():
        adaptive = cv2.adaptiveThreshold(label_crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return cv2.erode(adaptive, rect_kernel(2), iterations=1)

    def upscale_sharpen(img):
        h, w = img.shape[:2]
        return sharpen(cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC))

    yield "Schärfen", lambda: sharpen(label_crop)
    yield "Otsu+Erosion", otsu_erode
    yield "MorphClose", lambda: cv2.morphologyEx(label_crop, cv2.MORPH_CLOSE, rect_kernel(3))
    yield "Adaptiv+Erosion", adaptive_erode
    yield "2x-Upscale", lambda: upscale_sharpen(label_crop)
    for clip_limit in (4.0, 8.0, 15.0):
        yield f"CLAHE-{clip_limit}", lambda c=clip_limit: clahe(label_crop, c)
    for clip_limit in (8.0, 15.0):
        yield f"CLAHE-{clip_limit}+Upscale", lambda c=clip_limit: upscale_sharpen(clahe(label_crop, c))


def _decode_label_candidates(label_crop: np.ndarray) -> str | None:
    """pylibdmtx auf quadratischen Konturkandidaten im Etikett: roh, Otsu und über Gitter-Rekonstruktion."""
    crop_h, crop_w = label_crop.shape[:2]
    crop_inv = cv2.bitwise_not(otsu(clahe(label_crop, 4.0)))
    candidates = []
    collect_square_candidates(crop_inv, (35, 15), 1.6, candidates, [])
    candidates.sort(key=lambda x: x[1], reverse=True)
    logger.debug(f"DataMatrix-Kandidaten gefunden: {len(candidates)}")

    for idx, (contour, _, rect) in enumerate(candidates):
        x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
        pad_c = 20
        dmx_crop = label_crop[max(0, y_c - pad_c):min(crop_h, y_c + h_c + pad_c),
                              max(0, x_c - pad_c):min(crop_w, x_c + w_c + pad_c)]
        if dmx_crop.size == 0 or dmx_crop.shape[0] < 5 or dmx_crop.shape[1] < 5:
            continue
        dmx_crop = np.ascontiguousarray(dmx_crop)

        res = try_decode_dmtx(PILImage.fromarray(dmx_crop), timeout_ms=350)
        if res is not None:
            logger.info(f"DataMatrix gefunden (Kandidat {idx} direkt): {res}")
            return res

        res = try_decode_dmtx(PILImage.fromarray(otsu(dmx_crop)), timeout_ms=350)
        if res is not None:
            logger.info(f"DataMatrix gefunden (Kandidat {idx} Otsu): {res}")
            return res

        oriented = orient_corners(label_crop, np.float32(cv2.boxPoints(rect)))
        if oriented is not None:
            cells = warp_and_sample(label_crop, oriented)
            if cells is not None:
                res = try_decode_dmtx(PILImage.fromarray(generate_synthetic_dmtx(cells)), timeout_ms=400)
                if res is not None:
                    logger.info(f"DataMatrix gefunden (Kandidat {idx} Rekonstruktion): {res}")
                    return res
    return None


def _read_with_pylibdmtx(gray: np.ndarray) -> str | None:
    """pylibdmtx-Kaskade: Gesamtbild, Upscale, CLAHE, Konturkandidaten und Filter-Varianten des Etiketts."""
    h, w = gray.shape[:2]
    if w > 1200:
        scale = 1200.0 / w
        gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = gray.shape[:2]
        logger.debug(f"DataMatrix-Fallback: Bild auf {w}x{h} herunterskaliert für pylibdmtx.")

    res = try_decode_dmtx(PILImage.fromarray(gray), timeout_ms=300)
    if res is not None:
        logger.info(f"DataMatrix gefunden (Direkt-Scan): {res}")
        return res

    if w < 600:
        gray_up = sharpen(cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC))
        res = try_decode_dmtx(PILImage.fromarray(gray_up), timeout_ms=400)
        if res is not None:
            logger.info(f"DataMatrix gefunden (Upscale 2x): {res}")
            return res

    for clip_limit in (4.0, 8.0, 15.0):
        res = try_decode_dmtx(PILImage.fromarray(clahe(gray, clip_limit)), timeout_ms=300)
        if res is not None:
            logger.info(f"DataMatrix gefunden (CLAHE clipLimit={clip_limit}): {res}")
            return res

    label_crop = find_label_crop(gray)
    res = _decode_label_candidates(label_crop)
    if res is not None:
        return res

    for name, make_variant in _label_filter_variants(label_crop):
        try:
            result = try_decode_dmtx(PILImage.fromarray(make_variant()), timeout_ms=300)
            if result is not None:
                logger.info(f"DataMatrix gefunden (Label-Fallback: {name}): {result}")
                return result
        except Exception as e:
            logger.debug(f"DMTX-Variante '{name}' Fehler: {e}")

    logger.info("Kein DataMatrix-Code gefunden (Kandidaten + alle Fallbacks fehlgeschlagen).")
    return None


def read_datamatrix(frame: np.ndarray) -> str | None:
    """
    Erweiterte DataMatrix-Lesekaskade nach erfolglosem _decode_basic_zxing():
    weitere zxing-cpp-Vorverarbeitungen, danach pylibdmtx-Fallbacks.
    """
    gray = to_gray(frame)

    if config.USE_ZXING_FASTPATH:
        for name, make_variant in _zxing_preprocessing_cascade(gray):
            variant = make_variant()
            if variant is None:
                continue
            code = try_zxing_dmtx(variant)
            if code is not None:
                logger.info(f"DataMatrix gefunden (zxing-cpp {name}): {code}")
                return code

    if not _load_dmtx():
        return None
    try:
        return _read_with_pylibdmtx(gray)
    except Exception as e:
        logger.warning(f"DataMatrix-Scan Fehler: {e}")
        return None


# --------------------------------------------------------------------------- #
#  Sichtbarkeitsanalyse & Rahmen-Rekonstruktion                                #
# --------------------------------------------------------------------------- #

def check_dmx_visibility(frame: np.ndarray) -> dict:
    """
    Dekodiert die DataMatrix (erweiterte Lesekaskade) oder analysiert, ob nur der Rahmen gestört ist.

    Returns:
        dict mit status ("clear" | "outer_only" | "blocked"), decoded_text, observed_grid, inner_8x8.
    """
    result = {"status": "blocked", "decoded_text": None, "observed_grid": None, "inner_8x8": None}

    direct_text = read_datamatrix(frame)
    if direct_text is not None:
        result["status"] = "clear"
        result["decoded_text"] = direct_text
        logger.info(f"DMX Sichtbarkeit: KLAR — direkt dekodiert: '{direct_text}'")
        return result

    observed = None
    for method in ("otsu", "adaptive", "mean"):
        observed = extract_observed_grid(frame, binarization_method=method)
        if observed is not None:
            result["observed_grid"] = observed
            logger.debug(f"DMX Sichtbarkeit: Grid extrahiert mit Methode '{method}'")
            break
    if observed is None:
        logger.info("DMX Sichtbarkeit: BLOCKED — kein Grid extrahierbar.")
        return result

    inner = observed[1:9, 1:9].copy()
    result["inner_8x8"] = inner
    inner_ratio = np.sum(inner == 0) / inner.size
    if not 0.15 <= inner_ratio <= 0.85:
        logger.info(f"DMX Sichtbarkeit: BLOCKED — Innere 8×8 nicht plausibel (Schwarz-Anteil: {inner_ratio:.0%})")
        return result

    # Datenbereich plausibel → Rahmen-Rekonstruktion versuchen (unabhängig vom Rahmen-Zustand)
    frame_ok_count = sum(frame_pattern_ok(observed, min_finder=8, min_timing=7))
    result["status"] = "outer_only"
    if frame_ok_count < 3:
        logger.info(f"DMX Sichtbarkeit: NUR RAHMEN GESTÖRT — Innere 8×8 plausibel (schwarz: {inner_ratio:.0%}), "
                    f"Rahmen OK: {frame_ok_count}/4")
    else:
        logger.info(f"DMX Sichtbarkeit: Rahmen scheint OK ({frame_ok_count}/4), "
                    f"aber Dekodierung fehlgeschlagen → versuche Rekonstruktion.")
    return result


def reconstruct_from_inner(inner_8x8: np.ndarray) -> str | None:
    """Setzt einen perfekten Rahmen um die inneren 8x8-Module und dekodiert das synthetische Bild."""
    if inner_8x8 is None or inner_8x8.shape != (8, 8):
        return None

    full_grid = frame_grid()
    full_grid[1:9, 1:9] = inner_8x8
    synthetic = generate_synthetic_dmtx(full_grid)

    decoded = try_decode_dmtx(PILImage.fromarray(synthetic), timeout_ms=400)
    if decoded is not None:
        logger.info(f"DMX Rahmen-Rekonstruktion erfolgreich: '{decoded}'")
        return decoded

    decoded = try_decode_dmtx(PILImage.fromarray(cv2.equalizeHist(synthetic)), timeout_ms=400)
    if decoded is not None:
        logger.info(f"DMX Rahmen-Rekonstruktion erfolgreich (Enhanced): '{decoded}'")
        return decoded

    logger.debug("DMX Rahmen-Rekonstruktion: Dekodierung fehlgeschlagen.")
    return None


def scan_datamatrix_pipeline(frame: np.ndarray) -> dict:
    """
    DataMatrix-Pipeline: zxing-cpp Fast-Path (roh, CLAHE), danach erweiterte Lesekaskade und Rahmen-Rekonstruktion.

    Returns:
        dict mit status ("decoded" | "reconstructed" | "blocked"), text, method_detail, confidence, observed_grid.
    """
    code, method_detail = _decode_basic_zxing(frame)
    if code is not None:
        return {"status": "decoded", "text": code, "method_detail": method_detail,
                "confidence": 1.0, "observed_grid": None}

    visibility = check_dmx_visibility(frame)
    if visibility["status"] == "clear":
        return {"status": "decoded", "text": visibility["decoded_text"], "method_detail": "DataMatrix direkt dekodiert",
                "confidence": 1.0, "observed_grid": visibility.get("observed_grid")}

    if visibility["status"] == "outer_only":
        reconstructed_text = reconstruct_from_inner(visibility["inner_8x8"])
        if reconstructed_text is not None:
            return {"status": "reconstructed", "text": reconstructed_text,
                    "method_detail": "Rahmen rekonstruiert (innere 8×8 OK)",
                    "confidence": 0.90, "observed_grid": visibility.get("observed_grid")}
        logger.info("DMX Pipeline: Rahmen-Rekonstruktion fehlgeschlagen → blocked.")

    return dmx_blocked("DataMatrix nicht lesbar", visibility.get("observed_grid"))
