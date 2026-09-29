"""Klarschrift-Erkennung (EasyOCR, optional PACC) mit Multi-Pass-Vorverarbeitung und Teillesungen."""

import logging
import os

import cv2
import numpy as np

from . import config
from .code_format import (clean_chars, clean_to_4chars, format_partial_display, is_prefix_like, is_valid_horden_code,
                          normalize_partial_3chars)
from .config import ALLOWED_CHARS, REQUIRED_LENGTH
from .image_ops import (clahe, gamma_lut, otsu, preprocess_etch_denoise, preprocess_faded_contrast,
                        preprocess_niblack, preprocess_ridge_enhancement, preprocess_sauvola, rect_kernel, sharpen,
                        to_gray)
from .onnx_models import predict_pacc
from .results import ocr_failed, ocr_ok

logger = logging.getLogger(__name__)

_ocr_reader = None


def _load_ocr():
    """EasyOCR-Reader (Lazy-Loading); nutzt lokale Modelle aus models/easyocr, sonst Standardpfad/Download."""
    global _ocr_reader
    if _ocr_reader is None:
        logger.info("Lade EasyOCR-Modell (Erstladen kann ~10 Sekunden dauern)...")
        import easyocr

        model_dir = os.path.join(config.APP_DIR, 'models', 'easyocr')
        os.makedirs(model_dir, exist_ok=True)
        has_local_models = (
            os.path.exists(os.path.join(model_dir, 'craft_mlt_25k.pth')) and
            os.path.exists(os.path.join(model_dir, 'latin_g2.pth'))
        )
        if has_local_models:
            logger.info(f"Lade lokale OCR-Modelle aus {model_dir}")
            _ocr_reader = easyocr.Reader(['de', 'en'], gpu=False, model_storage_directory=model_dir,
                                         download_enabled=False)
        else:
            logger.info(f"Keine lokalen Modelle in {model_dir} gefunden. Nutze Standard-Pfad und lade ggf. herunter.")
            _ocr_reader = easyocr.Reader(['de', 'en'], gpu=False)
        logger.info("EasyOCR bereit.")
    return _ocr_reader


def _read_text(reader, image: np.ndarray) -> list:
    return reader.readtext(image, detail=1, paragraph=False, beamWidth=1, allowlist=ALLOWED_CHARS)


# --------------------------------------------------------------------------- #
#  Vorverarbeitung                                                             #
# --------------------------------------------------------------------------- #

def _primary_ocr_variants(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Die 2 effektivsten Varianten: CLAHE 3.0 (normal lesbar) und 8.0 (leicht ausgebleicht)."""
    sharpened = sharpen(to_gray(image))
    return [("standard", clahe(sharpened, 3.0)), ("aggressiv", clahe(sharpened, 8.0))]


def _extended_ocr_variants(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Zusätzliche Varianten für schwierige Etiketten (verblasst, geätzt, invertiert, unterbelichtet)."""
    gray = to_gray(image)
    sharpened = sharpen(gray)

    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
    return [
        ("faded_boost", preprocess_faded_contrast(gray)),
        ("etch_denoise", preprocess_etch_denoise(gray)),
        ("ridge_boost", preprocess_ridge_enhancement(gray)),
        ("sauvola_w11", preprocess_sauvola(gray, window_size=11, k=0.2)),
        ("niblack", preprocess_niblack(gray, window_size=21, k=-0.2)),
        ("extrem", clahe(sharpened, 15.0)),
        ("invertiert", clahe(cv2.bitwise_not(sharpened), 8.0)),
        ("binaer_otsu", otsu(sharpened)),
        # Adaptiv + Closing repariert gebrochene/verblasste Buchstabenstriche
        ("morph_close_ocr", cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, rect_kernel(2))),
        # Gamma 0.3 hellt dunkle Etiketten stark auf (liest z. B. W031 auf dunklen Etiketten korrekt)
        ("gamma_03", cv2.LUT(gray, gamma_lut(0.3))),
        ("gamma_05_clahe", clahe(cv2.LUT(gray, gamma_lut(0.5)), 5.0)),
    ]


# --------------------------------------------------------------------------- #
#  Auswertung der EasyOCR-Ergebnisse                                           #
# --------------------------------------------------------------------------- #

def _sorted_confident(ocr_results: list) -> list:
    """Ergebnisse zeilenweise (y, dann x) sortiert, nur mit Konfidenz > 0.2."""
    return [r for r in sorted(ocr_results, key=lambda r: (r[0][0][1], r[0][0][0])) if r[2] > 0.2]


def extract_4char_candidate(ocr_results: list) -> tuple[str | None, float]:
    """Bester formatgültiger 4-Zeichen-Code: zuerst ein einzelnes Ergebnis, sonst räumlich zusammengesetzt."""
    if not ocr_results:
        return None, 0.0

    for _, text, conf in ocr_results:
        candidate = clean_to_4chars(text)
        if candidate is not None and conf > 0.2:
            logger.debug(f"OCR 4-Char Kandidat gefunden (direkt): '{candidate}' (Konfidenz: {conf:.2f})")
            return candidate, conf

    confident = _sorted_confident(ocr_results)
    combined = ''.join(clean_chars(r[1]) for r in confident)
    avg_conf = sum(r[2] for r in confident) / max(1, len(confident))
    if len(combined) == REQUIRED_LENGTH:
        candidate = clean_to_4chars(combined)
        if candidate is not None:
            logger.debug(f"OCR 4-Char Kandidat gefunden (kombiniert): '{candidate}' (Konfidenz: {avg_conf:.2f})")
            return candidate, avg_conf

    return None, 0.0


def extract_partial_candidate(ocr_results: list) -> str | None:
    """3-Zeichen-Teilcode bei teilweiser Verdeckung oder None."""
    if not ocr_results:
        return None
    combined = ''.join(clean_chars(r[1]) for r in _sorted_confident(ocr_results))
    return combined if len(combined) == 3 else None


def build_partial_with_confidence(ocr_results: list,
                                  min_char_conf: float = 0.75) -> tuple[list[str], str, list[int]]:
    """
    Markiert unsichere Zeichen (Konfidenz < min_char_conf) mit '?'.

    Returns:
        (sichere Zeichen, Anzeige-String, Indizes der fehlenden Stellen)
    """
    if not ocr_results:
        return [], "????", [0, 1, 2, 3]

    char_list = []
    for _, text, conf in sorted(ocr_results, key=lambda r: r[0][0][0]):
        char_list.extend((ch, conf) for ch in clean_chars(text))
    char_list = char_list[:4]
    char_list += [('?', 0.0)] * (4 - len(char_list))

    high_conf_chars = []
    display_chars = []
    missing_positions = []
    for i, (ch, conf) in enumerate(char_list):
        if conf >= min_char_conf and ch != '?':
            high_conf_chars.append(ch)
            display_chars.append(ch)
        else:
            display_chars.append('?')
            missing_positions.append(i)

    # 3 sichere Zeichen ohne Präfix-Buchstaben → die Lücke liegt an Stelle 0, nicht am Ende
    if len(high_conf_chars) == 3 and not is_prefix_like(high_conf_chars[0]):
        display_chars = ['?'] + high_conf_chars
        missing_positions = [0]

    return high_conf_chars, ''.join(display_chars), missing_positions


def estimate_missing_position_spatial(ocr_results: list) -> list[int]:
    """Schätzt die fehlende Zeichenposition aus den horizontalen Lücken zwischen den OCR-Boxen (Standard: [3])."""
    if not ocr_results:
        return [3]

    segments = []
    for bbox, text, conf in ocr_results:
        if conf <= 0.2:
            continue
        cleaned = clean_chars(text)
        if not cleaned:
            continue
        xs = [pt[0] for pt in bbox]
        segments.append((min(xs), max(xs), cleaned))

    segments.sort(key=lambda x: x[0])
    if not segments:
        return [3]

    total_chars = sum(len(text) for _, _, text in segments)
    if total_chars == 0:
        return [3]
    char_w = sum(x_max - x_min for x_min, x_max, _ in segments) / total_chars

    # Zeichen den Slots 0..3 zuordnen; Slot 0 beginnt am ersten Segment
    slots = []
    for i, (x_min, _, text) in enumerate(segments):
        if i == 0:
            start_slot = 0
        else:
            gap_slots = int(round((x_min - segments[i - 1][1]) / char_w))
            start_slot = slots[-1] + 1 + gap_slots
        slots.extend(start_slot + offset for offset in range(len(text)))

    missing = [slot for slot in range(4) if slot not in set(slots)]
    if len(missing) == 1:
        logger.info(f"BBox-gap analysis found missing position: {missing[0]} (slots occupied: {slots})")
        return missing
    return [3]


# --------------------------------------------------------------------------- #
#  Multi-Pass OCR                                                              #
# --------------------------------------------------------------------------- #

def _prepare_ocr_zone(frame: np.ndarray, text_crop: bool = False) -> np.ndarray:
    """Untere 65 % des Bildes (Klarschrift; bei text_crop das ganze Bild) mit Rand für die CRAFT-Detektion, max. 800 px breit."""
    h_frame = frame.shape[0]
    ocr_zone = frame if text_crop else frame[int(h_frame * 0.35):, :]
    # value=255 ist nur bei Graustufen weiß; bei BGR entsteht (255, 0, 0) – eine Änderung verschiebt OCR-Ergebnisse
    ocr_zone = cv2.copyMakeBorder(ocr_zone, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)

    # Herunterskalieren spart ~75 % PyTorch-Rechenzeit
    ocr_h, ocr_w = ocr_zone.shape[:2]
    if ocr_w > 800:
        ocr_scale = 800.0 / ocr_w
        ocr_zone = cv2.resize(ocr_zone, (0, 0), fx=ocr_scale, fy=ocr_scale, interpolation=cv2.INTER_AREA)
        logger.debug(f"OCR-Zone herunterskaliert: {ocr_w}x{ocr_h} → {ocr_zone.shape[1]}x{ocr_zone.shape[0]}")
    return ocr_zone


def _partial_from_three_chars(results: list, variant_name: str) -> dict | None:
    """Teillesung aus genau 3 erkannten Zeichen; die fehlende Stelle wird aus den Box-Lücken geschätzt."""
    partial = extract_partial_candidate(results)
    if partial is None:
        return None

    partial_norm, prefix_detected = normalize_partial_3chars(partial)
    missing_pos = estimate_missing_position_spatial(results)
    if prefix_detected and 0 in missing_pos:
        missing_pos = [1]       # mit erkanntem Präfix kann Stelle 0 nicht fehlen
    elif not prefix_detected:
        missing_pos = [0]       # 3 Ziffern ohne Präfix → der Buchstabe fehlt

    partial_display = format_partial_display(partial_norm, missing_pos)
    logger.info(f"OCR Partial [{variant_name}] (3 Zeichen): Display='{partial_display}', Lesbar='{partial_norm}'")
    return {
        "status": "partial",
        "text": None,
        "partial_display": partial_display,
        "readable_chars": partial_norm,
        "confidence": 0.0,
        "readable_count": 3,
        "missing_positions": missing_pos,
        "raw_candidate": None,
    }


def read_ocr_with_status(frame: np.ndarray, text_crop: bool = False) -> dict:
    """
    Liest die Klarschrift mit mehreren Vorverarbeitungen (Multi-Pass) und liefert das beste Ergebnis.
    text_crop: das Bild ist bereits ein Klarschrift-Ausschnitt (nicht auf die untere Bildhälfte beschränken).

    Returns:
        dict mit status ("ok" | "partial" | "failed"), text, partial_display, readable_chars, confidence,
        readable_count, missing_positions, raw_candidate.
    """
    try:
        ocr_zone = _prepare_ocr_zone(frame, text_crop)

        pacc_code, pacc_conf = predict_pacc(ocr_zone)
        if pacc_code is not None and pacc_conf >= 0.70:
            logger.info(f"PACC Fast-Path OCR erfolgreich: '{pacc_code}' (Konfidenz: {pacc_conf:.2f})")
            return ocr_ok(pacc_code, pacc_conf)

        reader = _load_ocr()
        best_ok_result = None
        best_partial_result = None

        for variant_name, preprocessed_img in _primary_ocr_variants(ocr_zone):
            results = _read_text(reader, preprocessed_img)
            if not results:
                logger.debug(f"OCR [{variant_name}]: Keine Zeichen erkannt.")
                continue

            text, confidence = extract_4char_candidate(results)
            if text is not None:
                if confidence >= 0.75:
                    logger.info(f"OCR OK [{variant_name}]: '{text}' (Konfidenz: {confidence:.2f})")
                    return ocr_ok(text, confidence)

                if confidence >= 0.40 and is_valid_horden_code(text):
                    logger.info(f"OCR OK [{variant_name}] (Format-validiert, Early-Exit): '{text}' "
                                f"(Konfidenz: {confidence:.2f})")
                    return ocr_ok(text, confidence)

                # Niedrige Gesamtkonfidenz: Einzelzeichen prüfen
                if confidence >= 0.20:
                    high_conf_chars, partial_display, missing_pos = build_partial_with_confidence(results)
                    if len(high_conf_chars) == 4:
                        logger.info(f"OCR OK [{variant_name}] (Einzelkonfidenz): '{text}' (Gesamt: {confidence:.2f})")
                        if best_ok_result is None or confidence > best_ok_result["confidence"]:
                            best_ok_result = ocr_ok(text, confidence)
                        continue
                    if len(high_conf_chars) >= 3 and best_partial_result is None:
                        readable_str = ''.join(high_conf_chars)
                        logger.info(f"OCR Partial [{variant_name}] (Konfidenz-Filter): "
                                    f"Display='{partial_display}', Lesbar='{readable_str}'")
                        best_partial_result = {
                            "status": "partial",
                            "text": None,
                            "partial_display": partial_display,
                            "readable_chars": readable_str if readable_str else None,
                            "confidence": confidence,
                            "readable_count": len(high_conf_chars),
                            "missing_positions": missing_pos,
                            "raw_candidate": text,
                        }

            if best_partial_result is None:
                best_partial_result = _partial_from_three_chars(results, variant_name)

        # Niedrig-konfidentes OK (< 0.60) verliert gegen eine Teillesung mit Präfix:
        # aggressive CLAHE-Varianten können Ziffern verfälschen (z. B. 2 → 3).
        if best_ok_result is not None and best_partial_result is not None:
            if best_ok_result["confidence"] < 0.60 and best_partial_result.get("readable_chars"):
                _, prefix_detected = normalize_partial_3chars(best_partial_result["readable_chars"])
                if prefix_detected:
                    logger.info(
                        f"OCR: Bevorzuge Partial '{best_partial_result['partial_display']}' "
                        f"über niedrig-konfidentes OK '{best_ok_result['text']}' "
                        f"(Conf={best_ok_result['confidence']:.2f})"
                    )
                    return best_partial_result
        if best_ok_result is not None:
            return best_ok_result

        if best_partial_result is not None:
            logger.info("OCR Fast-Mode ergab nur Partial → Retry mit erweiterten Varianten...")
            for variant_name, preprocessed_img in _extended_ocr_variants(ocr_zone):
                results = _read_text(reader, preprocessed_img)
                if not results:
                    continue
                text, confidence = extract_4char_candidate(results)
                if text is not None and confidence >= 0.40:
                    if is_valid_horden_code(text):
                        logger.info(f"OCR OK [{variant_name}] (Retry, Format-validiert): '{text}' "
                                    f"(Konfidenz: {confidence:.2f})")
                        return ocr_ok(text, confidence)
                    if confidence >= 0.75:
                        logger.info(f"OCR OK [{variant_name}] (Retry): '{text}' (Konfidenz: {confidence:.2f})")
                        return ocr_ok(text, confidence)
            return best_partial_result

        logger.warning("OCR: Keine Variante konnte genügend Zeichen erkennen → failed.")
        return ocr_failed()

    except Exception as e:
        logger.error(f"OCR-Fehler: {e}")
        return ocr_failed()
