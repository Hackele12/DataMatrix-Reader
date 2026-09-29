"""Horden-DB-Abgleich (Schalter config.USE_HORDE_DB_MATCHING): Ganzbild-Matching und OCR-Konfusionskorrektur."""

import logging

import cv2
import numpy as np

from .code_format import is_valid_horden_code
from .image_ops import to_gray
from .results import scan_result

logger = logging.getLogger(__name__)

# Typische OCR-Ziffernverwechslungen je gelesener Ziffer
_OCR_CONFUSION_MAP = {
    '0': ['8', '6', '9'],
    '8': ['0', '3', '6'],
    '3': ['8', '9'],
    '6': ['0', '8'],
    '9': ['0', '3'],
    '4': ['0', '1'],
    '1': ['4', '7'],
    '5': ['3', '6'],
    '7': ['1'],
}


def match_horde_image(frame: np.ndarray) -> dict | None:
    """Ganzbild-Abgleich gegen die gespeicherten Hordenbilder → Scan-Ergebnis oder None."""
    try:
        import horde_db
        horde_match = horde_db.match_horde_image(frame, min_confidence=0.88)
        if horde_match and horde_match.get("success") and horde_match.get("result"):
            code = horde_match["result"]
            conf = horde_match.get("confidence", 0.85)
            logger.info(f"[FAST-PATH HORDEN-DB] Hordenbild-Match direkt erkannt: '{code}' (Conf: {conf:.2%}).")
            return scan_result(True, code, "HordenDB-Match", conf, ocr=code, partial=code)
    except Exception as e:
        logger.warning(f"HordeDB FastPath Fehler: {e}")
    return None


def correct_ocr_confusion(code: str, frame: np.ndarray = None) -> str:
    """
    Korrigiert OCR-Ziffernverwechslungen anhand der Horden-DB: Existiert der Code nicht, aber genau eine
    1-Zeichen-Konfusionsvariante, wird diese übernommen; bei mehreren entscheidet der Bildabgleich.
    """
    if not code or len(code) != 4 or not is_valid_horden_code(code):
        return code

    try:
        import horde_db
        if not horde_db._cache_initialized:
            horde_db.load_horde_db()
        cache = horde_db._horde_cache

        if code.upper() in cache:
            return code

        prefix, digits = code[0], list(code[1:4])
        candidates = []
        for pos in range(3):
            for replacement in _OCR_CONFUSION_MAP.get(digits[pos], []):
                variant_digits = digits.copy()
                variant_digits[pos] = replacement
                variant_code = (prefix + ''.join(variant_digits)).upper()
                if variant_code in cache and variant_code != code.upper():
                    candidates.append(variant_code)

        if not candidates:
            return code
        if len(candidates) == 1:
            logger.info(f"[OCR-POSTPROCESS] Konfusionskorrektur: '{code}' → '{candidates[0]}' "
                        f"(Vorlage in hard_scans_cache/ gefunden)")
            return candidates[0]

        if frame is not None:
            best_candidate = None
            best_corr = -1.0
            resized = cv2.resize(to_gray(frame), (320, 320), interpolation=cv2.INTER_AREA)
            for cand in candidates:
                if cache[cand].get("gray") is not None:
                    res = cv2.matchTemplate(resized, cache[cand]["gray"], cv2.TM_CCOEFF_NORMED)
                    corr = float(res[0][0]) if res is not None else 0.0
                    if corr > best_corr:
                        best_corr = corr
                        best_candidate = cand
            if best_candidate and best_corr > 0.3:
                logger.info(f"[OCR-POSTPROCESS] Konfusionskorrektur (Bildabgleich): '{code}' → '{best_candidate}' "
                            f"(Korrelation={best_corr:.2f}, {len(candidates)} Kandidaten)")
                return best_candidate

        logger.info(f"[OCR-POSTPROCESS] Konfusionskorrektur (Fallback): '{code}' → '{candidates[0]}'")
        return candidates[0]

    except Exception as e:
        logger.debug(f"[OCR-POSTPROCESS] Fehler: {e}")
        return code
