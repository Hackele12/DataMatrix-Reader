"""
scanner.py — Dual-Validation Scanner für Horden-Etiketten

Dieses Modul stellt die Kernfunktionalität für das Einlesen von Barcodes / DataMatrix-Codes
und Klarschrift (OCR) zur Verfügung. Es kombiniert beide Technologien in einer parallelen
Pipeline (Dual-Validation), um maximale Zuverlässigkeit und Robustheit zu gewährleisten.

Features:
- Paralleles Scanning von DataMatrix (pylibdmtx) und Klarschrift (EasyOCR)
- Mathematische DataMatrix-Rekonstruktion (Reed-Solomon-Fehlerkorrektur und Utah-Placement)
- Heuristische Erkennung teilweise verdeckter Codes und deren Gitterrekonstruktion
- Lazy-Loading für schnelle App-Startzeiten und Offline-Betriebseignung
"""

import logging
import time
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# --- Lazy-Loading: Bibliotheken werden erst beim ersten Scan geladen ---
_ocr_reader = None
_dmtx_available = False
_dmtx_loaded = False

# --- Schalter für zxing-cpp Fast-Path Integration ---
USE_ZXING_FASTPATH = True

# --- Erlaubte Zeichen für Horden-Codes ---
ALLOWED_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
REQUIRED_LENGTH = 4


def _load_dmtx():
    """
    Lädt die 'pylibdmtx'-Bibliothek bei Bedarf (Lazy-Loading).
    
    Returns:
        bool: True, wenn die Bibliothek erfolgreich geladen wurde, sonst False.
    """
    global _dmtx_available, _dmtx_loaded
    if not _dmtx_loaded:
        try:
            import setuptools  # noqa: F401 (distutils Kompatibilität für Python 3.12+)
            from pylibdmtx.pylibdmtx import decode  # noqa: F401
            _dmtx_available = True
            logger.info("pylibdmtx erfolgreich geladen.")
        except Exception as e:
            logger.warning(f"pylibdmtx konnte nicht geladen werden: {e}. DataMatrix-Scan deaktiviert.")
            _dmtx_available = False
        _dmtx_loaded = True
    return _dmtx_available


# --- zxing-cpp Hochgeschwindigkeits-Reader (3D-Homographie & Multi-Scale) ---
_zxing_available = None  # None = noch nicht geprüft


def _load_zxing():
    """
    Prüft ob zxing-cpp verfügbar ist (Lazy-Loading).
    
    Returns:
        bool: True wenn zxing-cpp importiert werden kann.
    """
    global _zxing_available
    if _zxing_available is None:
        try:
            import zxingcpp  # noqa: F401
            _zxing_available = True
            logger.info("zxing-cpp (C++20 High-Speed Reader) erfolgreich geladen.")
        except ImportError:
            _zxing_available = False
            logger.info("zxing-cpp nicht verfügbar — verwende pylibdmtx als Fallback.")
    return _zxing_available


def _try_zxing_dmtx(image: np.ndarray) -> str | None:
    """
    Versucht einen DataMatrix-Code per zxing-cpp zu dekodieren (< 5ms).
    Nutzt try_rotate, try_invert und verschiedene Binarisierer.
    
    Args:
        image (np.ndarray): Das Graustufen- oder Farbbild.
        
    Returns:
        str | None: Der erkannte 4-stellige Code oder None.
    """
    if not _load_zxing():
        return None
    
    try:
        import zxingcpp
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        binarizers = [
            zxingcpp.Binarizer.LocalAverage,
            zxingcpp.Binarizer.GlobalHistogram,
            zxingcpp.Binarizer.FixedThreshold,
        ]
        
        for binarizer in binarizers:
            res = zxingcpp.read_barcode(
                gray,
                formats=zxingcpp.BarcodeFormat.DataMatrix,
                try_rotate=True,
                try_downscale=True,
                try_invert=True,
                binarizer=binarizer,
            )
            if res and res.valid and res.text:
                text = res.text.strip()
                candidate = _clean_to_4chars(text)
                if candidate is not None:
                    logger.info(f"zxing-cpp DataMatrix erkannt (Binarisierer {binarizer}): '{candidate}'")
                    return candidate
    except Exception as e:
        logger.debug(f"zxing-cpp Fehler: {e}")
    
    return None


def _load_ocr():
    """
    Lädt das EasyOCR-Modell bei Bedarf (Lazy-Loading) und initialisiert es offline.
    
    Returns:
        easyocr.Reader: Die initialisierte EasyOCR-Instanz.
    """
    global _ocr_reader
    if _ocr_reader is None:
        logger.info("Lade EasyOCR-Modell (Erstladen kann ~10 Sekunden dauern)...")
        import easyocr
        import os
        import sys
        
        # Pfadermittlung für PyInstaller (frozen) oder Skript-Modus
        if getattr(sys, 'frozen', False):
            project_dir = os.path.dirname(sys.executable)
        else:
            project_dir = os.path.dirname(os.path.abspath(__file__))
            
        model_dir = os.path.join(project_dir, 'models', 'easyocr')
        os.makedirs(model_dir, exist_ok=True)
        
        # Prüfung, ob die OCR-Modelle lokal vorhanden sind
        has_local_models = (
            os.path.exists(os.path.join(model_dir, 'craft_mlt_25k.pth')) and
            os.path.exists(os.path.join(model_dir, 'latin_g2.pth'))
        )
        
        if has_local_models:
            logger.info(f"Lade lokale OCR-Modelle aus {model_dir}")
            _ocr_reader = easyocr.Reader(
                ['de', 'en'], gpu=False,
                model_storage_directory=model_dir,
                download_enabled=False
            )
        else:
            logger.info(f"Keine lokalen Modelle in {model_dir} gefunden. Nutze Standard-Pfad und lade ggf. herunter.")
            _ocr_reader = easyocr.Reader(['de', 'en'], gpu=False)
            
        logger.info("EasyOCR bereit.")
    return _ocr_reader


def _sharpen(image: np.ndarray) -> np.ndarray:
    """
    Wendet eine Unsharp-Mask-Schärfung auf das Bild an.
    
    Args:
        image (np.ndarray): Das Eingangsbild.
        
    Returns:
        np.ndarray: Das geschärfte Bild.
    """
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return sharpened


def _preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Bereitet ein Bild für die Texterkennung (OCR) vor (Standard-Variante).
    Konvertiert in Graustufen, schärft das Bild und wendet CLAHE zur Kontrastverbesserung an.
    
    Args:
        image (np.ndarray): Das Originalbild.
        
    Returns:
        np.ndarray: Das vorverarbeitete Bild.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpened = _sharpen(gray)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(sharpened)
    return enhanced


def _preprocess_ocr_variants(image: np.ndarray, fast_mode: bool = True) -> list[tuple[str, np.ndarray]]:
    """
    Erzeugt vorverarbeitete Versionen des Bildes für OCR.
    Im Fast-Mode (Standard) werden nur die 2 effektivsten Varianten erzeugt.
    Im vollständigen Modus werden alle 5 Varianten generiert.
    
    Args:
        image (np.ndarray): Das Originalbild (BGR oder Graustufen).
        fast_mode (bool): True = nur 2 primäre Varianten, False = alle 5.
        
    Returns:
        list[tuple[str, np.ndarray]]: Liste von (Variantenname, vorverarbeitetes Bild).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    sharpened = _sharpen(gray)
    variants = []
    
    # Variante 1: Standard (clipLimit=3.0) – für normal lesbare Codes
    clahe_std = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    variants.append(("standard", clahe_std.apply(sharpened)))
    
    # Variante 2: Aggressiv (clipLimit=8.0) – für leicht ausgebleichte Codes
    clahe_agg = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
    variants.append(("aggressiv", clahe_agg.apply(sharpened)))
    
    # Variante 3: Faded Contrast Boost – Perzentil-Stretching + Morphologie
    faded_boost = _preprocess_faded_contrast(gray)
    variants.append(("faded_boost", faded_boost))
    
    if not fast_mode:
        # Variante 4: Extrem (clipLimit=15.0) – für stark ausgebleichte Codes
        clahe_ext = cv2.createCLAHE(clipLimit=15.0, tileGridSize=(8, 8))
        variants.append(("extrem", clahe_ext.apply(sharpened)))
        
        # Variante 5: Invertiert + aggressives CLAHE – für invertierte Kontraste
        inverted = cv2.bitwise_not(sharpened)
        clahe_inv = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
        variants.append(("invertiert", clahe_inv.apply(inverted)))
        
        # Variante 6: Binär-Otsu – maximaler Schwarz/Weiß-Kontrast
        _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(("binaer_otsu", binary))
        
        # Variante 7: Adaptiv + MorphClose (2x2) – repariert gebrochene/verblasste Buchstabenstriche
        adaptive_ocr = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 3
        )
        kernel_2x2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph_close_ocr = cv2.morphologyEx(adaptive_ocr, cv2.MORPH_CLOSE, kernel_2x2)
        variants.append(("morph_close_ocr", morph_close_ocr))
    
    return variants


def _preprocess_tophat(image: np.ndarray) -> np.ndarray:
    """
    Entfernt Spiegelungen und Glanzstellen auf Metall- und Plastiketiketten per Top-Hat-Filter.
    Isoliert helle Strukturen und neutralisiert großflächigen Glanz.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    enhanced = cv2.addWeighted(gray, 1.0, tophat, 1.5, 0)
    return _sharpen(enhanced)


def _preprocess_faded_contrast(image: np.ndarray) -> np.ndarray:
    """
    Spezial-Preprocessing für verbleichte, extrem kontrastarme Codes/Klarschriften.
    Kombiniert Perzentil-Histogramm-Stretching (2%-98%) mit morphologischem TopHat-BottomHat-Boost.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 1. Perzentil-Stretching (Spreizung des verbleichten Grauwertspektrums auf 0..255)
    p_low, p_high = np.percentile(gray, (2, 98))
    if p_high > p_low:
        stretched = np.clip((gray.astype(np.float32) - p_low) * (255.0 / (p_high - p_low)), 0, 255).astype(np.uint8)
    else:
        stretched = gray

    # 2. Morphologischer Boost: Enhanced = Stretched + TopHat - BottomHat
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    tophat = cv2.morphologyEx(stretched, cv2.MORPH_TOPHAT, kernel)
    bottomhat = cv2.morphologyEx(stretched, cv2.MORPH_BLACKHAT, kernel)
    
    enhanced = cv2.add(stretched, tophat)
    enhanced = cv2.subtract(enhanced, bottomhat)

    return _sharpen(enhanced)


def _preprocess_sauvola(image: np.ndarray, window_size: int = 15, k: float = 0.2) -> np.ndarray:
    """
    Lokale Sauvola-Binarisierung zur Extraktion stark verbleichter Schriften und geätzter Punktraster.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray_f = gray.astype(np.float32)
    mean = cv2.boxFilter(gray_f, cv2.CV_32F, (window_size, window_size))
    sqr_mean = cv2.boxFilter(gray_f**2, cv2.CV_32F, (window_size, window_size))
    std = np.sqrt(np.maximum(0, sqr_mean - mean**2))

    R = 128.0
    thresh = mean * (1.0 + k * (std / R - 1.0))
    binary = np.where(gray_f >= thresh, 255, 0).astype(np.uint8)
    return binary


def _preprocess_for_dmtx(image: np.ndarray) -> np.ndarray:
    """
    Bereitet ein Bild für die DataMatrix-Erkennung vor (Schärfung + TopHat).
    
    Args:
        image (np.ndarray): Das Originalbild.
        
    Returns:
        np.ndarray: Das vorverarbeitete Bild.
    """
    return _preprocess_tophat(image)


# --- OCR-Zeichen-Normalisierung (spezifisch für Horden-Format ^[ABPW][0-9]{3}$) ---
# Erlaubte Anfangsbuchstaben für Horden-Codes
VALID_PREFIXES = frozenset('ABPW')

# Verwechslungstabelle: Ziffer -> Buchstabe (für Position 0)
_DIGIT_TO_LETTER = {'8': 'B', '4': 'A', '9': 'P'}

# Fuzzy-Zuordnung: Visuell ähnliche Buchstaben -> nächster gültiger Präfix (A/B/P/W)
# Wird verwendet wenn OCR einen Buchstaben liest, der keiner der 4 gültigen ist
# (z.B. abgebleichte Codes: W sieht aus wie V, B wie D, etc.)
_FUZZY_PREFIX_MAP = {
    'V': 'W',   # V ↔ W  (sehr ähnliche Form)
    'U': 'W',   # U ↔ W  (offene Unterseite)
    'M': 'W',   # M = umgekehrtes W
    'Y': 'W',   # Y unterer Teil ähnelt V/W
    'N': 'W',   # N Diagonalstrich ähnelt W
    'D': 'B',   # D ↔ B  (runde rechte Seite)
    'R': 'P',   # R = P mit Bein
    'F': 'P',   # F ↔ P  (obere Hälfte ähnlich)
    'T': 'P',   # T oberer Balken ähnelt P
    'H': 'A',   # H ↔ A  (Querbalken zwischen Strichen)
    'K': 'A',   # K Diagonalstriche ähneln A
    'X': 'A',   # X konvergierende Linien wie A
    'L': 'A',   # L = A ohne Spitze (bei Serifen)
    'O': 'B',   # O rund wie B
    'C': 'B',   # C = offenes B/D
    'E': 'B',   # E ≈ B (horizontale Striche)
    'G': 'B',   # G ↔ B/D (runde Form)
    'S': 'B',   # S Kurven ähneln B
    'Q': 'B',   # Q ↔ O ↔ B
    'I': 'B',   # I mit Serifen ↔ B
    'J': 'B',   # J Bogen ähnelt B
    'Z': 'A',   # Z Diagonale ähnelt A
}

# Verwechslungstabelle: Buchstabe -> Ziffer (für Positionen 1-3)
# Vollständiges Mapping aller 26 Buchstaben, damit ungültige Buchstaben an Pos 1-3
# automatisch in die visuell naheliegendste Ziffer umgewandelt werden.
_LETTER_TO_DIGIT = {
    # Mappings aus vorheriger Version
    'O': '0', 'D': '0', 'Q': '0',
    'I': '1', 'L': '1',
    'Z': '2',
    'S': '5',
    'G': '6',
    'B': '8',
    'A': '4',
    'P': '9',
    # Erweiterte Mappings für alle restlichen Buchstaben
    'C': '0',   # C ↔ 0 (offener Kreis)
    'E': '3',   # E ↔ 3
    'F': '7',   # F ↔ 7
    'H': '4',   # H ↔ 4 (Querbalken)
    'J': '0',   # J ↔ 0 (oft eine linksseitig ausgebleichte 0)
    'K': '4',   # K ↔ 4
    'M': '0',   # M ↔ 0 (breiter Kreis)
    'N': '0',   # N ↔ 0 (oder 7)
    'R': '8',   # R ↔ 8 (oder 9)
    'T': '7',   # T ↔ 7 (oder 1)
    'U': '0',   # U ↔ 0 (offener Bogen)
    'V': '0',   # V ↔ 0 (oder 7)
    'W': '0',   # W ↔ 0
    'X': '8',   # X ↔ 8 (oder 4)
    'Y': '9',   # Y ↔ 9
}


def _normalize_ocr_confusions(text: str) -> str:
    """
    Normalisiert einen 4-stelligen Text anhand der Horden-Code-Regeln:
    - Stelle 0: Muss ein Buchstabe aus [A, B, P, W] sein.
      Ziffern werden über _DIGIT_TO_LETTER korrigiert (z.B. 8->B).
      Andere Buchstaben werden über _FUZZY_PREFIX_MAP zum visuell
      ähnlichsten gültigen Präfix zugeordnet (z.B. V->W, D->B).
    - Stellen 1-3: Müssen Ziffern [0-9] sein (z.B. O/D/Q->0, I/L->1, etc.).
    
    Args:
        text (str): Der Eingabetext (4 Zeichen).
        
    Returns:
        str: Der normalisierte Text.
    """
    if not text or len(text) != 4:
        return text.upper()
        
    text = text.upper()
    chars = list(text)
    
    # Erste Stelle: Erwarte Buchstabe (A, B, P, W)
    if chars[0] in VALID_PREFIXES:
        pass  # Bereits gültiger Präfix
    elif chars[0] in _DIGIT_TO_LETTER:
        chars[0] = _DIGIT_TO_LETTER[chars[0]]
    elif chars[0] in _FUZZY_PREFIX_MAP:
        original = chars[0]
        chars[0] = _FUZZY_PREFIX_MAP[chars[0]]
        logger.debug(f"Fuzzy-Präfix: '{original}' → '{chars[0]}' (visuell ähnlichster Buchstabe)")
        
    # Letzte 3 Stellen: Erwarte Ziffern
    for i in range(1, 4):
        if chars[i] in _LETTER_TO_DIGIT:
            chars[i] = _LETTER_TO_DIGIT[chars[i]]
            
    return ''.join(chars)


def _normalize_partial_3chars(text: str) -> tuple[str, bool]:
    """
    Normalisiert einen 3-stelligen Teilcode anhand der Horden-Code-Regeln.
    Erkennt ob die erste Stelle ein gültiger Präfix (A/B/P/W) ist.
    
    - Falls Präfix erkannt: Stellen 1-2 werden als Ziffern normalisiert (O→0, I→1, etc.).
    - Falls kein Präfix: Alle 3 Stellen werden als Ziffern normalisiert.
    
    Args:
        text (str): Der Eingabetext (3 Zeichen).
        
    Returns:
        tuple[str, bool]: (normalisierter Text, True wenn Präfix erkannt wurde)
    """
    if not text or len(text) != 3:
        return (text.upper() if text else text), False
    
    text = text.upper()
    chars = list(text)
    
    # Prüfe ob erste Stelle ein gültiger Präfix ist oder werden kann
    prefix_detected = False
    if chars[0] in VALID_PREFIXES:
        prefix_detected = True
    elif chars[0] in _FUZZY_PREFIX_MAP:
        original = chars[0]
        chars[0] = _FUZZY_PREFIX_MAP[chars[0]]
        prefix_detected = True
        logger.debug(f"Partial Fuzzy-Präfix: '{original}' → '{chars[0]}'")
    elif chars[0] in _DIGIT_TO_LETTER:
        chars[0] = _DIGIT_TO_LETTER[chars[0]]
        prefix_detected = True
    
    if prefix_detected:
        # Restliche Stellen als Ziffern normalisieren
        for i in range(1, 3):
            if chars[i] in _LETTER_TO_DIGIT:
                chars[i] = _LETTER_TO_DIGIT[chars[i]]
    else:
        # Alle als Ziffern normalisieren (Präfix fehlt vermutlich)
        for i in range(3):
            if chars[i] in _LETTER_TO_DIGIT:
                chars[i] = _LETTER_TO_DIGIT[chars[i]]
    
    return ''.join(chars), prefix_detected


import re
_HORDEN_PATTERN = re.compile(r'^[ABPW][0-9]{3}$')


def _clean_to_4chars(text: str) -> str | None:
    """
    Bereinigt den Text, normalisiert OCR-Verwechslungen und prüft,
    ob das Ergebnis dem Horden-Format ^[ABPW][0-9]{3}$ entspricht.
    
    Args:
        text (str): Der zu prüfende Text.
        
    Returns:
        str | None: Der bereinigte, normalisierte Text oder None.
    """
    if not text:
        return None
    clean = ''.join(c for c in text.upper() if c in ALLOWED_CHARS)
    if len(clean) != REQUIRED_LENGTH:
        return None
    normalized = _normalize_ocr_confusions(clean)
    if _HORDEN_PATTERN.match(normalized):
        return normalized
    return None


def _extract_4char_candidate(ocr_results: list) -> tuple[str | None, float]:
    """
    Sucht nach dem besten 4-Zeichen-Kandidaten in den OCR-Ergebnissen.
    
    Zuerst wird nach einem einzelnen Ergebnis mit exakt 4 Zeichen gesucht.
    Falls keines gefunden wird, werden Teilergebnisse räumlich sortiert zusammengefügt.
    
    Args:
        ocr_results (list): Die Ergebnisse von EasyOCR.
        
    Returns:
        tuple[str | None, float]: Der gefundene Code und die Konfidenz (oder None, 0.0).
    """
    if not ocr_results:
        return None, 0.0

    # Strategie 1: Direktes 4-Zeichen-Ergebnis suchen
    for bbox, text, conf in ocr_results:
        candidate = _clean_to_4chars(text)
        if candidate is not None and conf > 0.2:
            logger.debug(f"OCR 4-Char Kandidat gefunden (direkt): '{candidate}' (Konfidenz: {conf:.2f})")
            return candidate, conf

    # Strategie 2: Kombination aller Teilergebnisse
    # Sortierung nach Y-Koordinate (Zeilen) und X-Koordinate (Spalten)
    sorted_results = sorted(ocr_results, key=lambda r: (r[0][0][1], r[0][0][0]))
    combined = ''.join(''.join(c for c in r[1].upper() if c in ALLOWED_CHARS) for r in sorted_results if r[2] > 0.2)
    avg_conf = sum(r[2] for r in sorted_results if r[2] > 0.2) / max(1, sum(1 for r in sorted_results if r[2] > 0.2))

    if len(combined) == REQUIRED_LENGTH:
        candidate = _clean_to_4chars(combined)
        if candidate is not None:
            logger.debug(f"OCR 4-Char Kandidat gefunden (kombiniert): '{candidate}' (Konfidenz: {avg_conf:.2f})")
            return candidate, avg_conf

    return None, 0.0


def _extract_partial_candidate(ocr_results: list) -> str | None:
    """
    Extrahiert einen 3-Zeichen-Teilcode aus OCR-Ergebnissen bei teilweiser Verdeckung.
    
    Args:
        ocr_results (list): Die Ergebnisse von EasyOCR.
        
    Returns:
        str | None: Der 3-Zeichen-Teilcode oder None.
    """
    if not ocr_results:
        return None

    sorted_results = sorted(ocr_results, key=lambda r: (r[0][0][1], r[0][0][0]))
    combined = ''.join(
        ''.join(c for c in r[1].upper() if c in ALLOWED_CHARS)
        for r in sorted_results if r[2] > 0.2
    )

    if len(combined) == 3:
        return combined

    return None


# --- Raster-Konstanten für DataMatrix-Spezifikationen ---
DMTX_GRID_SIZE = 10
DMTX_CELL_PX = 20
DMTX_WARP_SIZE = DMTX_GRID_SIZE * DMTX_CELL_PX


def _orient_corners(gray: np.ndarray, corners: np.ndarray, strict: bool = True) -> np.ndarray | None:
    """
    Bestimmt die korrekte Ausrichtung der 4 Ecken eines DataMatrix-Codes.
    Testet alle 4 Rotationen und bewertet L-Finder und Timing-Muster.
    
    Args:
        gray (np.ndarray): Graustufenbild.
        corners (np.ndarray): Die 4 Ecken des Kandidaten.
        strict (bool): Wenn False, wird der Mindest-Score für Akzeptanz gesenkt.
        
    Returns:
        np.ndarray | None: Die sortierten/ausgerichteten Ecken oder None.
    """
    warp_size = DMTX_WARP_SIZE
    grid = DMTX_GRID_SIZE
    cell = DMTX_CELL_PX
    dst_pts = np.float32([
        [0, 0], [warp_size, 0], [warp_size, warp_size], [0, warp_size]
    ])

    best_score = -1
    best_corners = None

    # Ecken grob sortieren nach Winkel zum Schwerpunkt
    center = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    sorted_idx = np.argsort(angles)
    sorted_corners = corners[sorted_idx]

    # Rotationen bewerten
    for rotation in range(4):
        rotated = np.roll(sorted_corners, rotation, axis=0)
        M = cv2.getPerspectiveTransform(rotated, dst_pts)
        warped = cv2.warpPerspective(gray, M, (warp_size, warp_size),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)

        _, warped_bin = cv2.threshold(warped, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        cells = np.zeros((grid, grid), dtype=np.uint8)
        for row in range(grid):
            for col in range(grid):
                cy = row * cell + cell // 2
                cx = col * cell + cell // 2
                region = warped_bin[cy - 3:cy + 3, cx - 3:cx + 3]
                cells[row, col] = 1 if np.mean(region) > 127 else 0

        score = 0.0

        # L-Finder links (muss schwarz sein)
        left_col = cells[:, 0]
        score += np.sum(left_col == 0)

        # L-Finder unten (muss schwarz sein)
        bottom_row = cells[grid - 1, :]
        score += np.sum(bottom_row == 0)

        # Timing-Pattern oben (alternierend)
        top_row = cells[0, :]
        expected_top = np.array([0 if i % 2 == 0 else 1 for i in range(grid)])
        score += np.sum(top_row == expected_top)

        # Timing-Pattern rechts (alternierend)
        right_col = cells[:, grid - 1]
        expected_right = np.array([0 if i % 2 == 0 else 1 for i in range(grid)])
        expected_right[grid - 1] = 0
        score += np.sum(right_col == expected_right)

        logger.debug(f"Rekonstruktion: Rotation {rotation}, Score {score:.0f}/40")

        if score > best_score:
            best_score = score
            best_corners = rotated.copy()

    # Akzeptanzgrenze bei mindestens 60% Übereinstimmung (entspannt bei strict=False)
    min_score = 24 if strict else 20
    if best_score < min_score:
        logger.debug(f"Rekonstruktion: Bester Score {best_score:.0f}/40 zu niedrig (min {min_score}).")
        return None

    logger.info(f"Rekonstruktion: Orientierung gefunden, Score {best_score:.0f}/40.")
    return best_corners


def _warp_and_sample(gray: np.ndarray, corners: np.ndarray, binarization_method: str = "otsu", strict_l_finder: bool = True) -> np.ndarray | None:
    """
    Entzerrt die Ecken perspektivisch und samplet die Zellen der 10x10 Matrix.
    
    Args:
        gray (np.ndarray): Graustufenbild.
        corners (np.ndarray): Die 4 ausgerichteten Ecken.
        binarization_method (str): Die Binarisierungsmethode ("otsu", "adaptive", "mean").
        strict_l_finder (bool): Ob der L-Finder Plausibilitätscheck erzwungen werden soll.
        
    Returns:
        np.ndarray | None: Die 10x10 Binärmatrix (0=schwarz, 1=weiß) oder None.
    """
    warp_size = DMTX_WARP_SIZE
    grid = DMTX_GRID_SIZE
    cell = DMTX_CELL_PX

    dst_pts = np.float32([
        [0, 0], [warp_size, 0], [warp_size, warp_size], [0, warp_size]
    ])

    M = cv2.getPerspectiveTransform(corners, dst_pts)
    warped = cv2.warpPerspective(gray, M, (warp_size, warp_size),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)

    if binarization_method == "otsu":
        _, warped_bin = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif binarization_method == "adaptive":
        warped_bin = cv2.adaptiveThreshold(
            warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 4
        )
    elif binarization_method == "mean":
        mean_val = np.mean(warped)
        _, warped_bin = cv2.threshold(warped, mean_val, 255, cv2.THRESH_BINARY)
    elif binarization_method == "clahe_otsu":
        clahe_warp = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(4, 4))
        warped_enhanced = clahe_warp.apply(warped)
        _, warped_bin = cv2.threshold(warped_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, warped_bin = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphologisches Schließen (3x3 Kernel) zur Entfernung kleiner Ätzbecken-Löcher in schwarzen Modulen
    kernel_3x3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    warped_bin_clean = cv2.morphologyEx(warped_bin, cv2.MORPH_CLOSE, kernel_3x3)

    cells = np.zeros((grid, grid), dtype=np.uint8)
    for row in range(grid):
        for col in range(grid):
            cy = row * cell + cell // 2
            cx = col * cell + cell // 2
            half = max(2, cell // 3)
            region = warped_bin_clean[cy - half:cy + half, cx - half:cx + half]
            if region.size == 0:
                continue
            cells[row, col] = 1 if np.median(region) > 127 else 0

    # L-Finder Plausibilitätscheck
    if strict_l_finder:
        if np.sum(cells[:, 0] == 0) < 6:
            logger.debug("Rekonstruktion: L-Finder links nicht ausreichend nach Sampling.")
            return None
        if np.sum(cells[grid - 1, :] == 0) < 6:
            logger.debug("Rekonstruktion: L-Finder unten nicht ausreichend nach Sampling.")
            return None

    logger.info("Rekonstruktion: 10x10 Binärmatrix erfolgreich extrahiert.")
    return cells


def _generate_synthetic_dmtx(cells: np.ndarray) -> np.ndarray:
    """
    Generiert ein künstliches, perfektes DataMatrix-Bild aus einer Binärmatrix.
    Fügt eine standardkonforme Quiet-Zone (Rand) von 20 Pixeln hinzu.
    
    Args:
        cells (np.ndarray): 10x10 Binärmatrix.
        
    Returns:
        np.ndarray: Das synthetische Bild.
    """
    grid = cells.shape[0]
    cell_px = DMTX_CELL_PX
    quiet_zone = cell_px

    img_size = grid * cell_px + 2 * quiet_zone
    img = np.full((img_size, img_size), 255, dtype=np.uint8)

    for row in range(grid):
        for col in range(grid):
            x0 = quiet_zone + col * cell_px
            y0 = quiet_zone + row * cell_px
            color = 0 if cells[row, col] == 0 else 255
            img[y0:y0 + cell_px, x0:x0 + cell_px] = color

    logger.debug(f"Synthetisches DataMatrix-Bild generiert: {img_size}x{img_size} Pixel.")
    return img


def _reconstruct_datamatrix(frame: np.ndarray) -> np.ndarray | None:
    """
    Sucht nach Etiketten und rekonstruiert deren DataMatrix-Code.
    
    Args:
        frame (np.ndarray): Das Kamerabild.
        
    Returns:
        np.ndarray | None: Das rekonstruierte Bild oder None.
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    h, w = gray.shape[:2]

    # Etikett-Segmentierung
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    label_crop = gray
    if contours:
        largest_label = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_label) > 10000:
            x_box, y_box, w_box, h_box = cv2.boundingRect(largest_label)
            pad = 10
            x1 = max(0, x_box - pad)
            y1 = max(0, y_box - pad)
            x2 = min(w, x_box + w_box + pad)
            y2 = min(h, y_box + h_box + pad)
            label_crop = gray[y1:y2, x1:x2]

    crop_h, crop_w = label_crop.shape[:2]
    _, crop_bin = cv2.threshold(label_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    crop_inv = cv2.bitwise_not(crop_bin)

    candidates = []
    seen_centers = []
    
    # Verschiedene Kernel-Größen für morphologisches Schließen probieren
    for k_size in [45, 35, 25, 15, 9]:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        closed = cv2.morphologyEx(crop_inv, cv2.MORPH_CLOSE, kernel, iterations=2)
        cnts, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 400 or area > (crop_h * crop_w) * 0.70:
                continue
                
            rect = cv2.minAreaRect(c)
            rect_w, rect_h = rect[1]
            if rect_w == 0 or rect_h == 0:
                continue
            aspect = max(rect_w, rect_h) / min(rect_w, rect_h)
            if aspect > 1.6:
                continue
                
            center = rect[0]
            duplicate = False
            for sc in seen_centers:
                dist = np.sqrt((center[0] - sc[0])**2 + (center[1] - sc[1])**2)
                if dist < 20:
                    duplicate = True
                    break
            if duplicate:
                continue
                
            seen_centers.append(center)
            candidates.append((c, area, rect))

    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)

    from PIL import Image as PILImage
    for contour, area, rect in candidates:
        box = cv2.boxPoints(rect)
        corners = np.float32(box)
        oriented = _orient_corners(label_crop, corners)
        if oriented is not None:
            warp_size = 200
            dst_pts = np.float32([
                [0, 0], [warp_size, 0], [warp_size, warp_size], [0, warp_size]
            ])
            M = cv2.getPerspectiveTransform(oriented, dst_pts)
            warped = cv2.warpPerspective(label_crop, M, (warp_size, warp_size),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REPLICATE)
            pad = 20
            padded = cv2.copyMakeBorder(warped, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
            
            # Dekodierungs-Versuch
            if _try_decode_dmtx(PILImage.fromarray(padded), timeout_ms=250) is not None:
                return padded
                
            # Grid-Sampling Fallback
            cells = _warp_and_sample(label_crop, oriented)
            if cells is not None:
                synthetic = _generate_synthetic_dmtx(cells)
                if _try_decode_dmtx(PILImage.fromarray(synthetic), timeout_ms=250) is not None:
                    return synthetic

    return None


# --- Reed-Solomon Fehlerkorrektur & GF(256) Arithmetik ---
_GF_POLY = 0x12D
_gf_exp = [0] * 512
_gf_log = [0] * 256
_gf_initialized = False


def _init_gf_tables():
    """Initialisiert die mathematischen Galois-Feld Tabellen für Reed-Solomon."""
    global _gf_initialized
    if _gf_initialized:
        return
    x = 1
    _gf_exp[0] = 1
    for i in range(1, 255):
        x <<= 1
        if x & 0x100:
            x ^= _GF_POLY
        _gf_exp[i] = x
        _gf_log[x] = i
    for i in range(255, 512):
        _gf_exp[i] = _gf_exp[i - 255]
    _gf_initialized = True


def _gf_mul(a: int, b: int) -> int:
    """Multipliziert zwei Elemente im GF(256)-Feld."""
    if a == 0 or b == 0:
        return 0
    return _gf_exp[_gf_log[a] + _gf_log[b]]


def _compute_rs_ecc(data: list[int]) -> list[int]:
    """
    Berechnet die Reed-Solomon-Fehlerkorrektur (5 ECC-Bytes für 3 Datenbytes).
    
    Args:
        data (list[int]): Die 3 Daten-Codewörter.
        
    Returns:
        list[int]: Die 5 berechneten Fehlerkorrektur-Codewörter.
    """
    _init_gf_tables()
    g = [1]
    for i in range(1, 6):
        alpha_i = _gf_exp[i]
        next_g = [0] * (len(g) + 1)
        for j in range(len(g)):
            next_g[j] ^= g[j]
            next_g[j + 1] ^= _gf_mul(g[j], alpha_i)
        g = next_g

    ecc = [0] * 5
    for byte in data:
        feedback = byte ^ ecc[0]
        ecc = ecc[1:] + [0]
        if feedback != 0:
            for i in range(5):
                ecc[i] ^= _gf_mul(g[i + 1], feedback)
    return ecc


def _encode_ascii_codewords(text: str) -> list[int] | None:
    """
    Enkodiert den Text in ASCII-Codewörter für DataMatrix ECC200.
    
    Args:
        text (str): Der 4-stellige Code.
        
    Returns:
        list[int] | None: Die 3 enkodierten Codewörter oder None.
    """
    if not text or len(text) != REQUIRED_LENGTH:
        return None
    text = text.upper()
    codewords = []
    i = 0
    while i < len(text):
        c = text[i]
        if c not in ALLOWED_CHARS:
            return None
        # Ziffernpaare kompakt enkodieren (Base 100)
        if c.isdigit() and i + 1 < len(text) and text[i + 1].isdigit():
            codewords.append(130 + int(c) * 10 + int(text[i + 1]))
            i += 2
        else:
            codewords.append(ord(c) + 1)
            i += 1
    # Mit Padding-Bytes auffüllen
    while len(codewords) < 3:
        codewords.append(129)
    if len(codewords) != 3:
        return None
    return codewords


# --- Utah-Placement für 8x8 Datenbereich (ISO/IEC 16022) ---
_PLACEMENT_MAP = None


def _get_placement_map() -> list[list]:
    """
    Generiert die Utah-Placement-Map für den 8x8 Datenbereich (ISO/IEC 16022).
    
    Returns:
        list[list]: Die Placement-Map mit Koordinatenzuordnungen.
    """
    global _PLACEMENT_MAP
    if _PLACEMENT_MAP is not None:
        return _PLACEMENT_MAP

    nrow, ncol = 8, 8

    def place_bit(grid, r, c, cw_idx, bit_idx):
        if r < 0:
            r += nrow
            c += 4 - ((nrow + 4) % 8)
        if c < 0:
            c += ncol
            r += 4 - ((ncol + 4) % 8)
        if 0 <= r < nrow and 0 <= c < ncol and grid[r][c] is None:
            grid[r][c] = (cw_idx, bit_idx)

    def place_utah(grid, r, c, cw_idx):
        place_bit(grid, r - 2, c - 2, cw_idx, 1)
        place_bit(grid, r - 2, c - 1, cw_idx, 2)
        place_bit(grid, r - 1, c - 2, cw_idx, 3)
        place_bit(grid, r - 1, c - 1, cw_idx, 4)
        place_bit(grid, r - 1, c,     cw_idx, 5)
        place_bit(grid, r,     c - 2, cw_idx, 6)
        place_bit(grid, r,     c - 1, cw_idx, 7)
        place_bit(grid, r,     c,     cw_idx, 8)

    def place_corner1(grid, cw_idx):
        place_bit(grid, nrow - 1, 0, cw_idx, 1)
        place_bit(grid, nrow - 1, 1, cw_idx, 2)
        place_bit(grid, nrow - 1, 2, cw_idx, 3)
        place_bit(grid, 0, ncol - 2, cw_idx, 4)
        place_bit(grid, 0, ncol - 1, cw_idx, 5)
        place_bit(grid, 1, ncol - 1, cw_idx, 6)
        place_bit(grid, 2, ncol - 1, cw_idx, 7)
        place_bit(grid, 3, ncol - 1, cw_idx, 8)

    grid = [[None] * ncol for _ in range(nrow)]
    r, c = 4, 0
    idx = 0

    while True:
        if r == nrow and c == 0:
            place_corner1(grid, idx)
            idx += 1
            r -= 2
            c += 2
        else:
            # Sweeps diagonal nach oben links
            while True:
                if r < nrow and c >= 0 and grid[r][c] is None:
                    place_utah(grid, r, c, idx)
                    idx += 1
                r -= 2
                c += 2
                if not (r >= 0 and c < ncol):
                    break
            r += 1
            c += 3
            # Sweeps diagonal nach unten rechts
            while True:
                if r >= 0 and c < ncol and grid[r][c] is None:
                    place_utah(grid, r, c, idx)
                    idx += 1
                r += 2
                c -= 2
                if not (r < nrow and c >= 0):
                    break
            r += 3
            c += 1

        if not (r < nrow or c < ncol):
            break

    _PLACEMENT_MAP = grid
    return grid


def _generate_reference_grid(text: str) -> np.ndarray | None:
    """
    Generiert das fehlerfreie 10x10 Referenzraster für einen gegebenen Code.
    
    Args:
        text (str): Der 4-stellige Code.
        
    Returns:
        np.ndarray | None: Das perfekte 10x10 Raster (0=schwarz, 1=weiß) oder None.
    """
    data_cw = _encode_ascii_codewords(text)
    if data_cw is None:
        return None

    ecc_cw = _compute_rs_ecc(data_cw)
    all_cw = data_cw + ecc_cw

    placement = _get_placement_map()
    data_grid = np.ones((8, 8), dtype=np.uint8)
    for r in range(8):
        for c in range(8):
            entry = placement[r][c]
            if entry is not None:
                cw_idx, bit_idx = entry
                if cw_idx < len(all_cw):
                    bit_val = (all_cw[cw_idx] >> (8 - bit_idx)) & 1
                    data_grid[r, c] = 0 if bit_val else 1

    full = np.ones((10, 10), dtype=np.uint8)

    # L-Finder setzen
    full[:, 0] = 0
    full[9, :] = 0
    
    # Timing-Muster setzen
    for c in range(10):
        full[0, c] = 0 if c % 2 == 0 else 1
    for r in range(10):
        full[r, 9] = 1 if r % 2 == 0 else 0
    full[9, 9] = 0

    # Datenbereich einbetten
    full[1:9, 1:9] = data_grid

    return full


# --- Mathematische 4.000-Code Vektor-Datenbank (0.5ms Lookup) ---
_ALL_CODES_LIST = None
_ALL_CODES_MATRIX = None


def _get_precomputed_4000_grid_matrix():
    """
    Vorberechnung aller 4.000 gültigen 10x10 DataMatrix-Binärmatrizen (A000-W999).
    Ermöglicht mathematischen 0.5ms Vektor-Lookup gegen alle Horden-Codes.
    
    Returns:
        tuple[list[str], np.ndarray]: Liste aller 4.000 Codes und NumPy Matrix (4000, 100).
    """
    global _ALL_CODES_LIST, _ALL_CODES_MATRIX
    if _ALL_CODES_MATRIX is None:
        codes = [f"{prefix}{num:03d}" for prefix in "ABPW" for num in range(1000)]
        matrix_rows = []
        for code in codes:
            grid = _generate_reference_grid(code)
            if grid is not None:
                matrix_rows.append(grid.flatten())
            else:
                matrix_rows.append(np.ones(100, dtype=np.uint8))
        _ALL_CODES_LIST = codes
        _ALL_CODES_MATRIX = np.array(matrix_rows, dtype=np.uint8)
        logger.info(f"4.000-Code DataMatrix Vektor-Datenbank erfolgreich vorberechnet ({_ALL_CODES_MATRIX.shape}).")
    return _ALL_CODES_LIST, _ALL_CODES_MATRIX


def _extract_observed_grid(frame: np.ndarray, binarization_method: str = "otsu", strict_l_finder: bool = True) -> np.ndarray | None:
    """
    Sucht nach dem Etikett im Bild und extrahiert das beobachtete 10x10 Grid.
    
    Args:
        frame (np.ndarray): Das Eingangsbild.
        binarization_method (str): Die Binarisierungsmethode.
        strict_l_finder (bool): Ob der L-Finder Plausibilitätscheck erzwungen werden soll.
        
    Returns:
        np.ndarray | None: Das extrahierte 10x10 Grid oder None.
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    h, w = gray.shape[:2]

    # Groben Etikettausschnitt ermitteln
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    label_crop = gray
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 10000:
            xb, yb, wb, hb = cv2.boundingRect(largest)
            pad = 10
            label_crop = gray[max(0, yb - pad):min(h, yb + hb + pad),
                              max(0, xb - pad):min(w, xb + wb + pad)]

    crop_h, crop_w = label_crop.shape[:2]
    
    # Mehrere CLAHE-Stufen für verblasste Etiketten probieren
    candidates = []
    seen_centers = []
    
    for clip_limit in [3.0, 8.0, 15.0]:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        label_enhanced = clahe.apply(label_crop)
        
        _, crop_bin_otsu = cv2.threshold(label_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        crop_bin_adapt = cv2.adaptiveThreshold(
            label_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 4
        )

        for b_img in [crop_bin_otsu, crop_bin_adapt]:
            crop_inv = cv2.bitwise_not(b_img)
            for k_size in [45, 35, 25, 15, 9]:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
                closed = cv2.morphologyEx(crop_inv, cv2.MORPH_CLOSE, kernel, iterations=2)
                cnts, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = cv2.contourArea(c)
                if area < 400 or area > (crop_h * crop_w) * 0.70:
                    continue
                rect = cv2.minAreaRect(c)
                rw, rh = rect[1]
                if rw == 0 or rh == 0:
                    continue
                if max(rw, rh) / min(rw, rh) > 2.0:
                    continue
                center = rect[0]
                dup = any(np.sqrt((center[0] - s[0])**2 + (center[1] - s[1])**2) < 20
                          for s in seen_centers)
                if dup:
                    continue
                seen_centers.append(center)
                candidates.append((c, area, rect))

    candidates.sort(key=lambda x: x[1], reverse=True)

    for contour, area, rect in candidates:
        box = cv2.boxPoints(rect)
        corners = np.float32(box)
        oriented = _orient_corners(label_enhanced, corners, strict=strict_l_finder)
        if oriented is not None:
            # Eckpunkt-Feinabstimmung für präzises DataMatrix-Grid-Sampling
            best_cells = None
            best_l_score = -1
            center = oriented.mean(axis=0)
            
            for scale in [0.96, 1.00, 1.04]:
                scaled = center + (oriented - center) * scale
                for dx in [-4, 0, 4]:
                    for dy in [-4, 0, 4]:
                        shifted = (scaled + np.array([dx, dy])).astype(np.float32)
                        cells = _warp_and_sample(label_enhanced, shifted, binarization_method=binarization_method, strict_l_finder=strict_l_finder)
                        if cells is not None:
                            l_score = np.sum(cells[:, 0] == 0) + np.sum(cells[9, :] == 0)
                            if l_score > best_l_score:
                                best_l_score = l_score
                                best_cells = cells
            if best_cells is not None:
                return best_cells

    return None


# --- Horden-Code Validierung (Format: ^[ABPW][0-9]{3}$) ---

def _is_valid_horden_code(code: str) -> bool:
    """
    Prüft, ob ein Code dem Horden-Format entspricht: [A|B|P|W] gefolgt von 3 Ziffern.
    
    Args:
        code (str): Der zu prüfende Code.
        
    Returns:
        bool: True wenn der Code dem Format entspricht.
    """
    return bool(_HORDEN_PATTERN.match(code))


_ref_grid_cache: dict[str, np.ndarray | None] = {}


def _get_cached_reference_grid(text: str) -> np.ndarray | None:
    """
    Gibt ein berechnetes Referenzraster zurück (mit LRU-Cache).
    
    Args:
        text (str): Der 4-stellige Code.
        
    Returns:
        np.ndarray | None: Das Gitter-Referenzraster.
    """
    if text not in _ref_grid_cache:
        _ref_grid_cache[text] = _generate_reference_grid(text)
        if len(_ref_grid_cache) > 500:
            keys = list(_ref_grid_cache.keys())
            for k in keys[:250]:
                del _ref_grid_cache[k]
    return _ref_grid_cache.get(text)


def _try_reconstruct(frame: np.ndarray, ocr_text: str | None,
                      ocr_conf: float, ocr_partial: str | None,
                      missing_positions: list[int] | None = None) -> dict | None:
    """
    Versucht den Code durch Abgleich der Gitterzellen mit Referenzrastern zu rekonstruieren.
    Kandidaten werden formatbasiert generiert (^[ABPW][0-9]{3}$).
    
    Args:
        frame (np.ndarray): Das Bild.
        ocr_text (str | None): Eventuelles OCR-Ergebnis.
        ocr_conf (float): OCR-Konfidenz.
        ocr_partial (str | None): Eventueller 3-stelliger OCR-Teilcode.
        
    Returns:
        dict | None: Ergebnis-Dictionary oder None bei Fehlschlag.
    """
    candidates = set()

    # Kandidatengenerierung (formatbasiert: ^[ABPW][0-9]{3}$)
    if ocr_text and len(ocr_text) == REQUIRED_LENGTH:
        # Normalisiere den OCR-Text und prüfe ob er gültig ist
        normalized = _normalize_ocr_confusions(ocr_text)
        if _is_valid_horden_code(normalized):
            candidates.add(normalized)
        # Erzeuge Nachbarkandidaten: Variiere jede Stelle einzeln
        for pos in range(REQUIRED_LENGTH):
            if pos == 0:
                # Position 0: Nur erlaubte Buchstaben
                for c in VALID_PREFIXES:
                    candidate = c + normalized[1:]
                    if _is_valid_horden_code(candidate):
                        candidates.add(candidate)
            else:
                # Positionen 1-3: Nur Ziffern
                for d in '0123456789':
                    candidate = normalized[:pos] + d + normalized[pos + 1:]
                    if _is_valid_horden_code(candidate):
                        candidates.add(candidate)
    elif ocr_partial and len(ocr_partial) == 3:
        # Normalisiere den Teilcode und prüfe ob Präfix vorhanden
        partial_norm, prefix_detected = _normalize_partial_3chars(ocr_partial)
        
        if prefix_detected:
            # Präfix bekannt → nur fehlende Ziffer an Position 1, 2 oder 3 einfügen
            prefix = partial_norm[0]
            digits = partial_norm[1:]  # Die 2 erkannten Ziffern
            
            allowed_insert_positions = range(3)
            if missing_positions:
                # missing_positions sind 0-basierte Indizes des 4-stelligen Strings.
                # Da prefix_detected True ist, ist das Präfix an Index 0 (immer vorhanden).
                # Die fehlende Ziffer liegt an Index 1, 2 oder 3.
                # In der insert_pos Schleife entspricht insert_pos = Index - 1.
                allowed_insert_positions = [pos - 1 for pos in missing_positions if 1 <= pos <= 3]
                if not allowed_insert_positions:
                    allowed_insert_positions = range(3)
            
            for insert_pos in allowed_insert_positions:
                for d in '0123456789':
                    code_digits = digits[:insert_pos] + d + digits[insert_pos:]
                    candidate = prefix + code_digits
                    if _is_valid_horden_code(candidate):
                        candidates.add(candidate)
            logger.info(
                f"Rekonstruktion: Partial '{ocr_partial}' → normalisiert '{partial_norm}' "
                f"(Präfix '{prefix}' erkannt). {len(candidates)} Kandidaten erzeugt."
            )
        else:
            # Kein Präfix → alle 4 Präfixe durchprobieren, 3 Ziffern sind bekannt
            for prefix in VALID_PREFIXES:
                candidate = prefix + partial_norm
                if _is_valid_horden_code(candidate):
                    candidates.add(candidate)
    else:
        # Kein OCR-Hinweis vorhanden → alle 4000 gültigen Codes als Kandidaten erzeugen
        # (4 Präfixe × 1000 Nummern = 4000 Codes)
        logger.info("Rekonstruktion: Keine OCR-Daten. Erzeuge alle 4000 formatgültigen Kandidaten...")
        for prefix in VALID_PREFIXES:
            for num in range(1000):
                candidates.add(f"{prefix}{num:03d}")

    if not candidates:
        return {"success": False, "grid_detected": False}

    # Gitter-Extraktion mit verschiedenen Binarisierungsmethoden (inkl. CLAHE+Otsu)
    observed_variants = {}
    for method in ["otsu", "adaptive", "mean", "clahe_otsu"]:
        grid_obs = _extract_observed_grid(frame, binarization_method=method, strict_l_finder=False)
        if grid_obs is not None:
            observed_variants[method] = grid_obs

    if not observed_variants:
        logger.debug("Rekonstruktion: Kein 10x10 Grid im Bild unter irgendeiner Binarisierung gefunden.")
        return {"success": False, "grid_detected": False}

    logger.info(f"Rekonstruktion: Teste {len(candidates)} Kandidaten gegen {len(observed_variants)} Grid-Varianten...")

    best_candidate = None
    best_overall_score = -1.0
    best_overall_margin = -1.0
    best_method = None
    all_codes, all_matrix = _get_precomputed_4000_grid_matrix()

    # Vektorisierter Übereinstimmungs-Vergleich über alle 4.000 Horden-Codes in < 1ms
    for method, observed in observed_variants.items():
        obs_flat = observed.flatten()
        # Matrix-Vergleich: Summe gleicher Bits für alle 4.000 Vorlagen in einem C/NumPy-Schritt
        matching_bits = np.sum(all_matrix == obs_flat, axis=1)

        if len(candidates) < 4000:
            # Falls gezielte Kandidaten vorhanden sind: Unterauswahl filtern
            indices = [all_codes.index(c) for c in candidates if c in all_codes]
            if not indices:
                continue
            cand_indices = np.array(indices)
            sub_matches = matching_bits[cand_indices]
            sorted_indices = np.argsort(sub_matches)[::-1]
            
            best_idx = cand_indices[sorted_indices[0]]
            best_cand_method = all_codes[best_idx]
            best_score_method = sub_matches[sorted_indices[0]] / 100.0
            
            second_score = (sub_matches[sorted_indices[1]] / 100.0) if len(sorted_indices) > 1 else 0.0
            margin_method = best_score_method - second_score
        else:
            # Alle 4.000 Vorlagen vergleichen
            top2_indices = np.argpartition(matching_bits, -2)[-2:]
            top2_indices = top2_indices[np.argsort(matching_bits[top2_indices])[::-1]]
            
            best_idx = top2_indices[0]
            second_idx = top2_indices[1]
            
            best_cand_method = all_codes[best_idx]
            best_score_method = matching_bits[best_idx] / 100.0
            margin_method = (matching_bits[best_idx] - matching_bits[second_idx]) / 100.0

        logger.debug(f"Rekonstruktion ({method}): Bester='{best_cand_method}' Score={best_score_method:.1%}, Abstand={margin_method:.1%}")
        
        if best_score_method > best_overall_score:
            best_overall_score = best_score_method
            best_overall_margin = margin_method
            best_candidate = best_cand_method
            best_method = method

    if best_candidate is None:
        return None

    logger.info(
        f"Rekonstruktion: Bester='{best_candidate}' Score={best_overall_score:.1%} (Methode: {best_method}), "
        f"Abstand={best_overall_margin:.1%}"
    )

    # Schwellenwerte zur Qualitätssicherung
    MIN_SCORE_STRICT = 0.80
    MIN_MARGIN_STRICT = 0.04
    MIN_SCORE_RELAXED = 0.65
    MIN_MARGIN_RELAXED = 0.10

    is_valid = False
    if best_overall_score >= MIN_SCORE_STRICT and best_overall_margin >= MIN_MARGIN_STRICT:
        is_valid = True
    elif best_overall_score >= MIN_SCORE_RELAXED and best_overall_margin >= MIN_MARGIN_RELAXED:
        logger.info("Rekonstruktion: Akzeptiert über Stufe 2 (Teilverdeckung).")
        is_valid = True
    elif ocr_text:
        normalized_ocr = _normalize_ocr_confusions(ocr_text)
        if best_candidate == normalized_ocr and ocr_conf >= 0.40 and best_overall_score >= 0.60:
            logger.info(
                f"Rekonstruktion: Akzeptiert über Stufe 3 (OCR-Übereinstimmung mit '{best_candidate}', "
                f"Score={best_overall_score:.1%}, OCR-Konfidenz={ocr_conf:.2f})."
            )
            is_valid = True
    
    # Stufe 4: Einziger Kandidat aus OCR und plausibles Grid
    if not is_valid and ocr_text and len(candidates) <= 35 and best_overall_score >= 0.60:
        normalized_ocr = _normalize_ocr_confusions(ocr_text)
        if best_candidate == normalized_ocr:
            logger.info(
                f"Rekonstruktion: Akzeptiert über Stufe 4 (wenige Kandidaten, "
                f"OCR-Match '{best_candidate}', Score={best_overall_score:.1%})."
            )
            is_valid = True

    # Stufe 5: OCR-Partial (3-stelliger Teilcode wie W03?) & Ziel-Matching des 10x10 Gitters
    # Erfordert ausreichenden Abstand zum zweitbesten Kandidaten (≥ 5%)
    if not is_valid and ocr_partial and best_overall_score >= 0.55 and best_overall_margin >= 0.05:
        clean_partial = ocr_partial.replace("?", "").strip().upper()
        if len(clean_partial) >= 2 and best_candidate.startswith(clean_partial):
            logger.info(
                f"Rekonstruktion: Akzeptiert über Stufe 5 (Partial-Match '{ocr_partial}' -> '{best_candidate}', "
                f"Score={best_overall_score:.1%}, Margin={best_overall_margin:.1%})."
            )
            is_valid = True

    if is_valid:
        logger.info(f"[OK] REKONSTRUIERT: '{best_candidate}' (Score={best_overall_score:.1%}, Abstand={best_overall_margin:.1%})")
        return {
            "success": True,
            "result": best_candidate,
            "method": "Rekonstruiert",
            "confidence": min(1.0, max(0.98, best_overall_score)),
            "dmtx_result": None,
            "ocr_result": ocr_text or ocr_partial,
            "verified": False,
            "ocr_partial_display": ocr_text or ocr_partial,
        }

    logger.info(f"Rekonstruktion ABGELEHNT: Score={best_overall_score:.1%}, Abstand={best_overall_margin:.1%}")
    return {"success": False, "grid_detected": True}


def _try_decode_dmtx(pil_img, timeout_ms: int = 250) -> str | None:
    """
    Versucht ein Bild per pylibdmtx als DataMatrix-Code zu dekodieren.
    
    Args:
        pil_img (PIL.Image): Das zu scannende Bild.
        timeout_ms (int): Maximales Timeout für den Scan.
        
    Returns:
        str | None: Der erkannte 4-stellige Code oder None.
    """
    try:
        from pylibdmtx.pylibdmtx import decode
        decoded = decode(pil_img, timeout=timeout_ms)
        if decoded:
            result_text = decoded[0].data.decode("utf-8", errors="ignore").strip()
            candidate = _clean_to_4chars(result_text)
            if candidate is not None:
                return candidate
    except Exception:
        pass
    return None


def _read_datamatrix(frame: np.ndarray) -> str | None:
    """
    Dekodiert den DataMatrix-Code unter Verwendung verschiedener Optimierungen und Fallbacks.
    
    Args:
        frame (np.ndarray): Das Graustufen- oder Farbbild.
        
    Returns:
        str | None: Der 4-stellige Code oder None bei Fehlschlag.
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    h, w = gray.shape[:2]

    # ===== STUFE 0: zxing-cpp Fast-Path (< 5ms, 3D-Homographie) =====
    if USE_ZXING_FASTPATH:
        zxing_result = _try_zxing_dmtx(gray)
        if zxing_result is not None:
            logger.info(f"DataMatrix gefunden (zxing-cpp Fast-Path): {zxing_result}")
            return zxing_result

        # zxing-cpp mit Kontrastverstärkung & Top-Hat Entspiegelung versuchen
        for clip_limit in [4.0, 10.0]:
            clahe_zx = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            enhanced_zx = clahe_zx.apply(gray)
            zxing_result = _try_zxing_dmtx(enhanced_zx)
            if zxing_result is not None:
                logger.info(f"DataMatrix gefunden (zxing-cpp + CLAHE {clip_limit}): {zxing_result}")
                return zxing_result

        tophat_img = _preprocess_tophat(gray)
        zxing_result = _try_zxing_dmtx(tophat_img)
        if zxing_result is not None:
            logger.info(f"DataMatrix gefunden (zxing-cpp + TopHat): {zxing_result}")
            return zxing_result

        faded_img = _preprocess_faded_contrast(gray)
        zxing_result = _try_zxing_dmtx(faded_img)
        if zxing_result is not None:
            logger.info(f"DataMatrix gefunden (zxing-cpp + FadedBoost): {zxing_result}")
            return zxing_result

        sauvola_img = _preprocess_sauvola(gray)
        zxing_result = _try_zxing_dmtx(sauvola_img)
        if zxing_result is not None:
            logger.info(f"DataMatrix gefunden (zxing-cpp + Sauvola): {zxing_result}")
            return zxing_result

    # ===== STUFE 1+: pylibdmtx Fallback (nur wenn zxing-cpp fehlschlägt) =====
    if not _load_dmtx():
        return None

    try:
        from PIL import Image as PILImage

        # Bild für pylibdmtx-Fallback auf max. 1200px Breite herunterskalieren
        if w > 1200:
            scale = 1200.0 / w
            gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            h, w = gray.shape[:2]
            logger.debug(f"DataMatrix-Fallback: Bild auf {w}x{h} herunterskaliert für pylibdmtx.")

        # 1. Schneller Direktscan auf Graustufenbild
        res_raw = _try_decode_dmtx(PILImage.fromarray(gray), timeout_ms=300)
        if res_raw is not None:
            logger.info(f"DataMatrix gefunden (Direkt-Scan): {res_raw}")
            return res_raw

        # 1b. Upscale bei kleinen Bildern (< 600px Breite)
        if w < 600:
            scale = 2
            gray_up = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
            gray_up = _sharpen(gray_up)
            res_up = _try_decode_dmtx(PILImage.fromarray(gray_up), timeout_ms=400)
            if res_up is not None:
                logger.info(f"DataMatrix gefunden (Upscale {scale}x): {res_up}")
                return res_up

        # 1c. CLAHE-verstärkter Direktscan
        for clip_limit in [4.0, 8.0, 15.0]:
            clahe_dmx = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            gray_enhanced = clahe_dmx.apply(gray)
            res_clahe = _try_decode_dmtx(PILImage.fromarray(gray_enhanced), timeout_ms=300)
            if res_clahe is not None:
                logger.info(f"DataMatrix gefunden (CLAHE clipLimit={clip_limit}): {res_clahe}")
                return res_clahe

        # 2. Ausschnitt des Etiketts ermitteln
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        label_crop = gray
        if contours:
            largest_label = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_label) > 10000:
                x_box, y_box, w_box, h_box = cv2.boundingRect(largest_label)
                pad = 10
                x1 = max(0, x_box - pad)
                y1 = max(0, y_box - pad)
                x2 = min(w, x_box + w_box + pad)
                y2 = min(h, y_box + h_box + pad)
                label_crop = gray[y1:y2, x1:x2]
                logger.debug(f"Label-Ausschnitt ermittelt: x={x1}, y={y1}, w={x2-x1}, h={y2-y1}")

        crop_h, crop_w = label_crop.shape[:2]
        
        # CLAHE-Verstärkung vor Binarisierung für bessere Konturerkennung bei verblassten Etiketten
        clahe_label = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        label_enhanced = clahe_label.apply(label_crop)
        _, crop_bin = cv2.threshold(label_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        crop_inv = cv2.bitwise_not(crop_bin)

        # 3. Konturkandidaten suchen
        candidates = []
        seen_centers = []
        
        for k_size in [35, 15]:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
            closed = cv2.morphologyEx(crop_inv, cv2.MORPH_CLOSE, kernel, iterations=2)
            cnts, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            for c in cnts:
                area = cv2.contourArea(c)
                if area < 400 or area > (crop_h * crop_w) * 0.70:
                    continue
                    
                rect = cv2.minAreaRect(c)
                rect_w, rect_h = rect[1]
                if rect_w == 0 or rect_h == 0:
                    continue
                aspect = max(rect_w, rect_h) / min(rect_w, rect_h)
                if aspect > 1.6:
                    continue
                    
                center = rect[0]
                duplicate = False
                for sc in seen_centers:
                    dist = np.sqrt((center[0] - sc[0])**2 + (center[1] - sc[1])**2)
                    if dist < 20:
                        duplicate = True
                        break
                if duplicate:
                    continue
                    
                seen_centers.append(center)
                candidates.append((c, area, rect, k_size))

        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        logger.debug(f"DataMatrix-Kandidaten gefunden: {len(candidates)}")

        # 4. Kandidaten prüfen
        for idx, (contour, area, rect, k_size) in enumerate(candidates):
            x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
            pad_c = 20
            x1_c = max(0, x_c - pad_c)
            y1_c = max(0, y_c - pad_c)
            x2_c = min(crop_w, x_c + w_c + pad_c)
            y2_c = min(crop_h, y_c + h_c + pad_c)
            
            dmx_crop = label_crop[y1_c:y2_c, x1_c:x2_c]
            
            # Methode A: Roh-Ausschnitt
            res = _try_decode_dmtx(PILImage.fromarray(dmx_crop), timeout_ms=350)
            if res is not None:
                logger.info(f"DataMatrix gefunden (Kandidat {idx} direkt): {res}")
                return res
                
            # Methode B: Lokale Binarisierung
            _, dmx_crop_bin = cv2.threshold(dmx_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            res = _try_decode_dmtx(PILImage.fromarray(dmx_crop_bin), timeout_ms=350)
            if res is not None:
                logger.info(f"DataMatrix gefunden (Kandidat {idx} Otsu): {res}")
                return res
                
            # Methode C: Gitter-Rekonstruktion
            box = cv2.boxPoints(rect)
            corners = np.float32(box)
            oriented = _orient_corners(label_crop, corners)
            if oriented is not None:
                cells = _warp_and_sample(label_crop, oriented)
                if cells is not None:
                    synthetic = _generate_synthetic_dmtx(cells)
                    res = _try_decode_dmtx(PILImage.fromarray(synthetic), timeout_ms=400)
                    if res is not None:
                        logger.info(f"DataMatrix gefunden (Kandidat {idx} Rekonstruktion): {res}")
                        return res

        # 5. Paralleler Filter-Fallback auf dem Etikett
        logger.debug("Starte parallele Filter-Pipeline auf dem Label-Ausschnitt.")

        def _make_variant_sharpen():
            return PILImage.fromarray(_sharpen(label_crop))

        def _make_variant_otsu_erode():
            _, bin_img = cv2.threshold(label_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            eroded = cv2.erode(bin_img, kernel, iterations=1)
            return PILImage.fromarray(eroded)

        def _make_variant_close():
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            closed_img = cv2.morphologyEx(label_crop, cv2.MORPH_CLOSE, kernel_close)
            return PILImage.fromarray(closed_img)

        def _make_variant_adaptive():
            adaptive = cv2.adaptiveThreshold(
                label_crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            kernel_strong = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            eroded_adaptive = cv2.erode(adaptive, kernel_strong, iterations=1)
            return PILImage.fromarray(eroded_adaptive)

        def _make_variant_upscale():
            h_c, w_c = label_crop.shape[:2]
            upscaled = cv2.resize(label_crop, (w_c * 2, h_c * 2), interpolation=cv2.INTER_CUBIC)
            sharpened = _sharpen(upscaled)
            return PILImage.fromarray(sharpened)

        def _make_variant_clahe(clip_limit):
            def _inner():
                clahe_v = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
                enhanced = clahe_v.apply(label_crop)
                return PILImage.fromarray(enhanced)
            return _inner

        def _make_variant_clahe_upscale(clip_limit):
            def _inner():
                clahe_v = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
                enhanced = clahe_v.apply(label_crop)
                h_c, w_c = enhanced.shape[:2]
                upscaled = cv2.resize(enhanced, (w_c * 2, h_c * 2), interpolation=cv2.INTER_CUBIC)
                sharpened = _sharpen(upscaled)
                return PILImage.fromarray(sharpened)
            return _inner

        variants = [
            ("Schärfen", _make_variant_sharpen),
            ("Otsu+Erosion", _make_variant_otsu_erode),
            ("MorphClose", _make_variant_close),
            ("Adaptiv+Erosion", _make_variant_adaptive),
            ("2x-Upscale", _make_variant_upscale),
            ("CLAHE-4.0", _make_variant_clahe(4.0)),
            ("CLAHE-8.0", _make_variant_clahe(8.0)),
            ("CLAHE-15.0", _make_variant_clahe(15.0)),
            ("CLAHE-8.0+Upscale", _make_variant_clahe_upscale(8.0)),
            ("CLAHE-15.0+Upscale", _make_variant_clahe_upscale(15.0)),
        ]

        with ThreadPoolExecutor(max_workers=len(variants)) as executor:
            futures = {}
            for name, make_fn in variants:
                try:
                    pil_img = make_fn()
                    future = executor.submit(_try_decode_dmtx, pil_img, 400)
                    futures[future] = name
                except Exception as e:
                    logger.debug(f"DMTX Variante '{name}' Preprocessing-Fehler: {e}")

            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        logger.info(f"DataMatrix gefunden (Label-Fallback: {name}): {result}")
                        for f in futures:
                            f.cancel()
                        return result
                except Exception as e:
                    logger.debug(f"DMTX-Variante '{name}' Fehler: {e}")

        logger.info("Kein DataMatrix-Code gefunden (Kandidaten + alle Fallbacks fehlgeschlagen).")
        return None
    except Exception as e:
        logger.warning(f"DataMatrix-Scan Fehler: {e}")
        return None


def _read_ocr(frame: np.ndarray) -> tuple[str | None, float, str | None]:
    """
    Legacy-Wrapper für OCR-Abfragen.
    
    Args:
        frame (np.ndarray): Das Eingangsbild.
        
    Returns:
        tuple[str | None, float, str | None]: 4-stelliger Code, Konfidenz und 3-stelliger Teilcode.
    """
    result = _read_ocr_with_status(frame)
    if result["status"] == "ok":
        return result["text"], result["confidence"], None
    elif result["status"] == "partial":
        return None, 0.0, result.get("readable_chars")
    return None, 0.0, None


def _read_ocr_with_status(frame: np.ndarray) -> dict:
    """
    Liest Text per OCR mit mehreren Preprocessing-Varianten (Multi-Pass).
    Iteriert über verschiedene Kontraststufen und gibt das beste Ergebnis zurück.
    Bei ausgebleichten Codes erhöht dies die Erkennungsrate erheblich.
    
    Args:
        frame (np.ndarray): Das Eingangsbild.
        
    Returns:
        dict: Statusinformationen der OCR-Lesung.
    """
    default_result = {
        "status": "failed", "text": None, "partial_display": None,
        "readable_chars": None, "confidence": 0.0,
        "readable_count": 0, "missing_positions": [],
        "raw_candidate": None,
    }

    try:
        reader = _load_ocr()
        h_frame, w_frame = frame.shape[:2]
        # Untere 65% des Bildes scannen (Klarschrift liegt typischerweise unten)
        ocr_zone = frame[int(h_frame * 0.35):, :]
        # Ausreichend weißer Rand (Quiet Zone) für EasyOCR CRAFT-Detektion
        ocr_zone = cv2.copyMakeBorder(ocr_zone, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        
        # OCR-Zone auf max. 800px Breite herunterskalieren (spart ~75% PyTorch-Rechenzeit)
        ocr_h, ocr_w = ocr_zone.shape[:2]
        if ocr_w > 800:
            ocr_scale = 800.0 / ocr_w
            ocr_zone = cv2.resize(ocr_zone, (0, 0), fx=ocr_scale, fy=ocr_scale, interpolation=cv2.INTER_AREA)
            logger.debug(f"OCR-Zone herunterskaliert: {ocr_w}x{ocr_h} → {ocr_zone.shape[1]}x{ocr_zone.shape[0]}")
        
        # Primäre Preprocessing-Varianten erzeugen (Fast-Mode: nur 2 statt 5)
        variants = _preprocess_ocr_variants(ocr_zone.copy(), fast_mode=True)
        
        # Bestes Ergebnis über alle Varianten sammeln
        best_ok_result = None       # Bestes "ok" Ergebnis (4 Zeichen, gültig)
        best_partial_result = None  # Bestes "partial" Ergebnis (3 Zeichen)
        
        for variant_name, preprocessed_img in variants:
            results = reader.readtext(
                preprocessed_img,
                detail=1,
                paragraph=False,
                beamWidth=1,
                allowlist=ALLOWED_CHARS,
            )
            
            if not results:
                logger.debug(f"OCR [{variant_name}]: Keine Zeichen erkannt.")
                continue
            
            text, confidence = _extract_4char_candidate(results)
            
            # ---- 4-Zeichen-Ergebnis gefunden ----
            if text is not None:
                # Hohe Konfidenz (≥ 0.75): sofort zurückgeben
                if confidence >= 0.75:
                    logger.info(f"OCR OK [{variant_name}]: '{text}' (Konfidenz: {confidence:.2f})")
                    return {
                        "status": "ok",
                        "text": text,
                        "partial_display": text,
                        "readable_chars": text,
                        "confidence": confidence,
                        "readable_count": 4,
                        "missing_positions": [],
                        "raw_candidate": text,
                    }
                
                # Mittlere Konfidenz (≥ 0.40) UND gültiges Horden-Format:
                # Format-Validierung gibt zusätzliche Sicherheit
                if confidence >= 0.40 and _is_valid_horden_code(text):
                    logger.info(
                        f"OCR OK [{variant_name}] (Format-validiert, Early-Exit): '{text}' "
                        f"(Konfidenz: {confidence:.2f})"
                    )
                    return {
                        "status": "ok",
                        "text": text,
                        "partial_display": text,
                        "readable_chars": text,
                        "confidence": confidence,
                        "readable_count": 4,
                        "missing_positions": [],
                        "raw_candidate": text,
                    }
                
                # Niedrige Konfidenz (≥ 0.20): Einzelzeichen prüfen
                if confidence >= 0.20:
                    high_conf_chars, partial_display, missing_pos = _build_partial_with_confidence(
                        results, min_char_conf=0.75
                    )
                    
                    if len(high_conf_chars) == 4:
                        logger.info(
                            f"OCR OK [{variant_name}] (Einzelkonfidenz): '{text}' "
                            f"(Gesamt: {confidence:.2f})"
                        )
                        candidate_result = {
                            "status": "ok",
                            "text": text,
                            "partial_display": text,
                            "readable_chars": text,
                            "confidence": confidence,
                            "readable_count": 4,
                            "missing_positions": [],
                            "raw_candidate": text,
                        }
                        if best_ok_result is None or confidence > best_ok_result["confidence"]:
                            best_ok_result = candidate_result
                        continue
                    elif len(high_conf_chars) >= 3 and best_partial_result is None:
                        readable_str = ''.join(high_conf_chars)
                        logger.info(
                            f"OCR Partial [{variant_name}] (Konfidenz-Filter): "
                            f"Display='{partial_display}', Lesbar='{readable_str}'"
                        )
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
            
            # ---- 3-Zeichen-Teillesung prüfen ----
            if best_partial_result is None:
                partial = _extract_partial_candidate(results)
                if partial is not None:
                    # Normalisiere den Teilcode (z.B. O→0, J→0)
                    partial_norm, prefix_detected = _normalize_partial_3chars(partial)
                    
                    # Pixel-basierte Lückenanalyse: sucht visuell nach der fehlenden Position
                    missing_pos = _estimate_missing_position_spatial(preprocessed_img, results)
                    # Sicherheitscheck: Wenn Präfix erkannt, darf Position 0 nicht fehlen
                    if prefix_detected and 0 in missing_pos:
                        missing_pos = [1]
                    
                    partial_display = _format_partial_display(partial_norm, missing_pos)
                    logger.info(
                        f"OCR Partial [{variant_name}] (3 Zeichen): "
                        f"Display='{partial_display}', Lesbar='{partial_norm}'"
                    )
                    best_partial_result = {
                        "status": "partial",
                        "text": None,
                        "partial_display": partial_display,
                        "readable_chars": partial_norm,
                        "confidence": 0.0,
                        "readable_count": 3,
                        "missing_positions": missing_pos,
                        "raw_candidate": None,
                    }
        
        # Bestes Ergebnis zurückgeben
        # Wenn OK-Ergebnis niedrige Konfidenz hat (<0.60) und ein Partial mit bekanntem Präfix existiert,
        # bevorzuge das Partial — aggressive CLAHE-Varianten können Ziffern verfälschen (z.B. 2→3)
        if best_ok_result is not None and best_partial_result is not None:
            if best_ok_result["confidence"] < 0.60 and best_partial_result.get("readable_chars"):
                partial_norm, prefix_detected = _normalize_partial_3chars(best_partial_result["readable_chars"])
                if prefix_detected:
                    logger.info(
                        f"OCR: Bevorzuge Partial '{best_partial_result['partial_display']}' "
                        f"über niedrig-konfidentes OK '{best_ok_result['text']}' "
                        f"(Conf={best_ok_result['confidence']:.2f})"
                    )
                    return best_partial_result
        if best_ok_result is not None:
            return best_ok_result
        
        # Retry mit restlichen Varianten wenn Fast-Mode nur Partial lieferte
        if best_partial_result is not None and best_ok_result is None:
            logger.info("OCR Fast-Mode ergab nur Partial → Retry mit erweiterten Varianten...")
            extra_variants = _preprocess_ocr_variants(ocr_zone.copy(), fast_mode=False)
            # Nur die zusätzlichen Varianten (ab Index 2) verarbeiten
            for variant_name, preprocessed_img in extra_variants[2:]:
                results = reader.readtext(
                    preprocessed_img,
                    detail=1,
                    paragraph=False,
                    beamWidth=1,
                    allowlist=ALLOWED_CHARS,
                )
                if not results:
                    continue
                
                text, confidence = _extract_4char_candidate(results)
                if text is not None and confidence >= 0.40:
                    if _is_valid_horden_code(text):
                        logger.info(
                            f"OCR OK [{variant_name}] (Retry, Format-validiert): '{text}' "
                            f"(Konfidenz: {confidence:.2f})"
                        )
                        return {
                            "status": "ok",
                            "text": text,
                            "partial_display": text,
                            "readable_chars": text,
                            "confidence": confidence,
                            "readable_count": 4,
                            "missing_positions": [],
                            "raw_candidate": text,
                        }
                    elif confidence >= 0.75:
                        logger.info(
                            f"OCR OK [{variant_name}] (Retry): '{text}' "
                            f"(Konfidenz: {confidence:.2f})"
                        )
                        return {
                            "status": "ok",
                            "text": text,
                            "partial_display": text,
                            "readable_chars": text,
                            "confidence": confidence,
                            "readable_count": 4,
                            "missing_positions": [],
                            "raw_candidate": text,
                        }
        
        if best_partial_result is not None:
            return best_partial_result
        
        logger.warning("OCR: Keine Variante konnte genügend Zeichen erkennen → failed.")
        return default_result

    except Exception as e:
        logger.error(f"OCR-Fehler: {e}")
        return default_result


def _build_partial_with_confidence(
    ocr_results: list, min_char_conf: float = 0.75
) -> tuple[list[str], str, list[int]]:
    """
    Überprüft die Einzelkonfidenz der Zeichen und markiert unsichere Zeichen mit '?'.
    
    Args:
        ocr_results (list): Die Ergebnisse von EasyOCR.
        min_char_conf (float): Minimale Konfidenz pro Zeichen.
        
    Returns:
        tuple[list[str], str, list[int]]: Sichere Zeichen, Anzeige-String und Lücken-Indizes.
    """
    if not ocr_results:
        return [], "????", [0, 1, 2, 3]

    sorted_results = sorted(ocr_results, key=lambda r: r[0][0][0])

    char_list = []
    for bbox, text, conf in sorted_results:
        cleaned = ''.join(c for c in text.upper() if c in ALLOWED_CHARS)
        for ch in cleaned:
            char_list.append((ch, conf))

    char_list = char_list[:4]

    while len(char_list) < 4:
        char_list.append(('?', 0.0))

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

    partial_display = ''.join(display_chars)
    return high_conf_chars, partial_display, missing_positions


def _estimate_missing_position_spatial(preprocessed_img: np.ndarray, ocr_results: list) -> list[int]:
    """
    Schätzt die Position des fehlenden Zeichens durch Analyse der horizontalen Abstände (Gaps)
    zwischen den Bounding Boxes der von EasyOCR erkannten Zeichen.
    
    Args:
        preprocessed_img (np.ndarray): Unbenutzt (nur zur Abwärtskompatibilität).
        ocr_results (list): Die Ergebnisse von EasyOCR.
        
    Returns:
        list[int]: Liste mit dem Index des vermuteten fehlenden Zeichens.
    """
    if not ocr_results:
        return [3]

    # Filter and sort OCR segments from left to right
    segments = []
    for bbox, text, conf in ocr_results:
        if conf <= 0.2:
            continue
        cleaned = ''.join(c for c in text.upper() if c in ALLOWED_CHARS)
        if not cleaned:
            continue
        # Get left and right x coordinates
        xs = [pt[0] for pt in bbox]
        x_min, x_max = min(xs), max(xs)
        segments.append((x_min, x_max, cleaned))
    
    segments.sort(key=lambda x: x[0])
    if not segments:
        return [3]
    
    # Calculate average character width
    total_w = 0
    total_chars = 0
    for x_min, x_max, text in segments:
        total_w += (x_max - x_min)
        total_chars += len(text)
    
    if total_chars == 0:
        return [3]
    char_w = total_w / total_chars
    
    # Let's map characters to slots 0, 1, 2, 3
    # We start with slot 0 at the start of the first segment
    slots = []
    for i, (x_min, x_max, text) in enumerate(segments):
        if i == 0:
            start_slot = 0
        else:
            prev_x_max = segments[i-1][1]
            gap = x_min - prev_x_max
            gap_slots = int(round(gap / char_w))
            start_slot = slots[-1] + 1 + gap_slots
            
        for offset in range(len(text)):
            slots.append(start_slot + offset)
            
    # Find the missing slots in range(4)
    all_slots = set(slots)
    missing = [slot for slot in range(4) if slot not in all_slots]
    
    if len(missing) == 1:
        logger.info(f"BBox-gap analysis found missing position: {missing[0]} (slots occupied: {slots})")
        return missing
        
    return [3]  # Default: letztes Zeichen


def _format_partial_display(readable_chars: str, missing_positions: list[int]) -> str:
    """
    Formatiert die Anzeige der Teillesung mit '?' an den fehlenden Stellen.
    
    Args:
        readable_chars (str): Lesbare Zeichen.
        missing_positions (list[int]): Fehlende Indizes.
        
    Returns:
        str: Der formatierte 4-stellige String.
    """
    if not missing_positions:
        return readable_chars[:4] if len(readable_chars) >= 4 else readable_chars

    result = list("????")
    char_idx = 0
    for pos in range(4):
        if pos in missing_positions:
            result[pos] = '?'
        else:
            if char_idx < len(readable_chars):
                result[pos] = readable_chars[char_idx]
                char_idx += 1

    return ''.join(result)


def _check_dmx_visibility(frame: np.ndarray) -> dict:
    """
    Analysiert die Sichtbarkeit des DataMatrix-Codes auf Beschädigung des Rahmens oder der Daten.
    
    Args:
        frame (np.ndarray): Das Graustufenbild.
        
    Returns:
        dict: Statusinformationen der Sichtbarkeitsanalyse.
    """
    result = {
        "status": "blocked",
        "decoded_text": None,
        "observed_grid": None,
        "inner_8x8": None,
    }

    # 1. Direkter Dekodierungsversuch
    direct_text = _read_datamatrix(frame)
    if direct_text is not None:
        result["status"] = "clear"
        result["decoded_text"] = direct_text
        logger.info(f"DMX Sichtbarkeit: KLAR — direkt dekodiert: '{direct_text}'")
        return result

    # 2. Grid-Extraktion
    observed = None
    for method in ["otsu", "adaptive", "mean"]:
        observed = _extract_observed_grid(frame, binarization_method=method)
        if observed is not None:
            result["observed_grid"] = observed
            logger.debug(f"DMX Sichtbarkeit: Grid extrahiert mit Methode '{method}'")
            break

    if observed is None:
        logger.info("DMX Sichtbarkeit: BLOCKED — kein Grid extrahierbar.")
        return result

    inner = observed[1:9, 1:9].copy()
    result["inner_8x8"] = inner

    # Plausibilitätsprüfung des Datenbereichs (Schwarz-Weiß-Verteilung)
    black_count = np.sum(inner == 0)
    white_count = np.sum(inner == 1)
    total = inner.size
    inner_ratio = black_count / total
    is_inner_plausible = 0.15 <= inner_ratio <= 0.85

    if is_inner_plausible:
        # Rahmen-Zustand prüfen
        left_col_ok = np.sum(observed[:, 0] == 0) >= 8
        bottom_row_ok = np.sum(observed[9, :] == 0) >= 8

        expected_top = np.array([0 if i % 2 == 0 else 1 for i in range(10)])
        top_row_ok = np.sum(observed[0, :] == expected_top) >= 7

        expected_right = np.array([0 if i % 2 == 0 else 1 for i in range(10)])
        expected_right[9] = 0
        right_col_ok = np.sum(observed[:, 9] == expected_right) >= 7

        frame_ok_count = sum([left_col_ok, bottom_row_ok, top_row_ok, right_col_ok])

        if frame_ok_count < 3:
            result["status"] = "outer_only"
            logger.info(
                f"DMX Sichtbarkeit: NUR RAHMEN GESTÖRT — "
                f"Innere 8×8 plausibel (schwarz: {inner_ratio:.0%}), "
                f"Rahmen OK: {frame_ok_count}/4"
            )
            return result
        else:
            result["status"] = "outer_only"
            logger.info(
                f"DMX Sichtbarkeit: Rahmen scheint OK ({frame_ok_count}/4), "
                f"aber Dekodierung fehlgeschlagen → versuche Rekonstruktion."
            )
            return result

    logger.info(
        f"DMX Sichtbarkeit: BLOCKED — Innere 8×8 nicht plausibel "
        f"(Schwarz-Anteil: {inner_ratio:.0%})"
    )
    return result


def _reconstruct_from_inner(inner_8x8: np.ndarray) -> str | None:
    """
    Baut einen perfekten Rahmen um die inneren 8x8 Module und dekodiert sie.
    
    Args:
        inner_8x8 (np.ndarray): Die inneren Datenmodule.
        
    Returns:
        str | None: Der dekodierte Code oder None.
    """
    from PIL import Image as PILImage

    if inner_8x8 is None or inner_8x8.shape != (8, 8):
        return None

    full_grid = np.ones((10, 10), dtype=np.uint8)

    full_grid[:, 0] = 0
    full_grid[9, :] = 0

    for c in range(10):
        full_grid[0, c] = 0 if c % 2 == 0 else 1

    for r in range(10):
        full_grid[r, 9] = 1 if r % 2 == 0 else 0
    full_grid[9, 9] = 0

    full_grid[1:9, 1:9] = inner_8x8

    synthetic = _generate_synthetic_dmtx(full_grid)
    decoded = _try_decode_dmtx(PILImage.fromarray(synthetic), timeout_ms=400)

    if decoded is not None:
        logger.info(f"DMX Rahmen-Rekonstruktion erfolgreich: '{decoded}'")
        return decoded

    # Fallback mit Histogramm-Ausgleich
    enhanced = cv2.equalizeHist(synthetic)
    decoded = _try_decode_dmtx(PILImage.fromarray(enhanced), timeout_ms=400)

    if decoded is not None:
        logger.info(f"DMX Rahmen-Rekonstruktion erfolgreich (Enhanced): '{decoded}'")
        return decoded

    logger.debug("DMX Rahmen-Rekonstruktion: Dekodierung fehlgeschlagen.")
    return None


def _scan_datamatrix_pipeline(frame: np.ndarray) -> dict:
    """
    Führt die DataMatrix-Erkennungs- und Rekonstruktions-Pipeline aus.
    zxing-cpp wird als blitzschneller Fast-Path vorangestellt.
    
    Args:
        frame (np.ndarray): Das Graustufenbild.
        
    Returns:
        dict: Das Scan-Ergebnis der DataMatrix-Pipeline.
    """
    # ===== Fast-Path: zxing-cpp Direkterkennung (< 5ms) =====
    zxing_result = _try_zxing_dmtx(frame)
    if zxing_result is not None:
        logger.info(f"DMX Pipeline: zxing-cpp Fast-Path erfolgreich: '{zxing_result}'")
        return {
            "status": "decoded",
            "text": zxing_result,
            "method_detail": "DataMatrix direkt dekodiert (zxing-cpp)",
            "confidence": 1.0,
            "observed_grid": None,
        }

    # zxing-cpp mit CLAHE-Kontrastverstärkung
    if len(frame.shape) == 3:
        gray_for_zx = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_for_zx = frame
    for clip_limit in [4.0, 10.0]:
        clahe_zx = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        enhanced_zx = clahe_zx.apply(gray_for_zx)
        zxing_result = _try_zxing_dmtx(enhanced_zx)
        if zxing_result is not None:
            logger.info(f"DMX Pipeline: zxing-cpp + CLAHE {clip_limit} erfolgreich: '{zxing_result}'")
            return {
                "status": "decoded",
                "text": zxing_result,
                "method_detail": f"DataMatrix dekodiert (zxing-cpp + CLAHE {clip_limit})",
                "confidence": 1.0,
                "observed_grid": None,
            }

    # ===== Fallback: pylibdmtx + Rekonstruktion =====
    visibility = _check_dmx_visibility(frame)

    if visibility["status"] == "clear":
        return {
            "status": "decoded",
            "text": visibility["decoded_text"],
            "method_detail": "DataMatrix direkt dekodiert",
            "confidence": 1.0,
            "observed_grid": visibility.get("observed_grid"),
        }

    if visibility["status"] == "outer_only":
        inner = visibility["inner_8x8"]
        reconstructed_text = _reconstruct_from_inner(inner)

        if reconstructed_text is not None:
            return {
                "status": "reconstructed",
                "text": reconstructed_text,
                "method_detail": "Rahmen rekonstruiert (innere 8×8 OK)",
                "confidence": 0.90,
                "observed_grid": visibility.get("observed_grid"),
            }
        else:
            logger.info("DMX Pipeline: Rahmen-Rekonstruktion fehlgeschlagen → blocked.")

    return {
        "status": "blocked",
        "text": None,
        "method_detail": "DataMatrix nicht lesbar",
        "confidence": 0.0,
        "observed_grid": visibility.get("observed_grid"),
    }


def _is_dmx_consistent_with_ocr(dmx_text: str, ocr_text: str | None, ocr_readable: str | None) -> bool:
    """
    Prüft, ob ein rekonstruierter DataMatrix-Text mit der OCR-Lesung kompatibel ist.
    Dies verhindert Falscherkennungen durch DataMatrix-Rekonstruktionen auf Rauschen/falschen Konturen.
    """
    if not dmx_text:
        return False
        
    dmx_norm = _normalize_ocr_confusions(dmx_text)
    
    # Falls wir eine vollständige OCR-Lesung haben
    if ocr_text:
        ocr_norm = _normalize_ocr_confusions(ocr_text)
        # Identisch oder maximal 1 Zeichen Unterschied erlaubt (Tippfehler)
        diffs = sum(1 for c1, c2 in zip(dmx_norm, ocr_norm) if c1 != c2)
        if diffs <= 1:
            return True
        return False
        
    # Falls wir nur eine OCR-Teillesung (3 Zeichen) haben
    if ocr_readable and len(ocr_readable) == 3:
        partial_norm, prefix_detected = _normalize_partial_3chars(ocr_readable)
        
        # dmx_norm muss die Teillesungs-Zeichen in der richtigen Reihenfolge enthalten
        # Wir testen alle 4 Subsequenzen der Länge 3 von dmx_norm
        for skip_idx in range(4):
            sub_seq = dmx_norm[:skip_idx] + dmx_norm[skip_idx+1:]
            if sub_seq == partial_norm:
                return True
        return False
        
    # Wenn keine OCR-Information existiert, können wir es nicht prüfen (standardmäßig konsistent)
    return True


def _merge_results(ocr_result: dict, dmx_result: dict, frame: np.ndarray) -> dict:
    """
    Führt die Ergebnisse von OCR und DataMatrix zusammen.
    
    Args:
        ocr_result (dict): Das OCR-Ergebnis.
        dmx_result (dict): Das DataMatrix-Ergebnis.
        frame (np.ndarray): Das Graustufenbild (für Rekonstruktion).
        
    Returns:
        dict: Das endgültige verifizierte oder rekonstruierte Scan-Ergebnis.
    """
    ocr_status = ocr_result["status"]
    dmx_status = dmx_result["status"]

    ocr_text = ocr_result.get("text")
    ocr_partial_display = ocr_result.get("partial_display")
    ocr_readable = ocr_result.get("readable_chars")
    ocr_conf = ocr_result.get("confidence", 0.0)
    ocr_missing_pos = ocr_result.get("missing_positions", [])
    ocr_raw_candidate = ocr_result.get("raw_candidate")

    dmx_text = dmx_result.get("text")
    dmx_conf = dmx_result.get("confidence", 0.0)

    logger.info(
        f"Merge: OCR={ocr_status}('{ocr_text or ocr_partial_display}') "
        f"+ DMX={dmx_status}('{dmx_text}')"
    )

    # DMX-Konsistenzprüfung für rekonstruierte Codes
    if dmx_status == "reconstructed":
        if not _is_dmx_consistent_with_ocr(dmx_text, ocr_text, ocr_readable or ocr_raw_candidate):
            logger.warning(
                f"[WARN] DMX-Rekonstruktion '{dmx_text}' verworfen, "
                f"da unvereinbar mit OCR '{ocr_text or ocr_readable or ocr_raw_candidate}'."
            )
            dmx_status = "blocked"
            dmx_text = None

    # 1. Fall: DataMatrix erfolgreich (direkt oder Rahmen-rekonstruiert)
    if dmx_status in ("decoded", "reconstructed"):
        method = "Verifiziert" if dmx_status == "decoded" else "Rekonstruiert"
        confidence = dmx_conf

        is_verified = False
        ocr_check_text = ocr_text or ocr_raw_candidate
        if ocr_check_text and dmx_text:
            dmtx_norm = _normalize_ocr_confusions(dmx_text.strip())
            ocr_norm = _normalize_ocr_confusions(ocr_check_text.strip())

            if dmtx_norm == ocr_norm or dmtx_norm in ocr_norm or ocr_norm in dmtx_norm:
                logger.info(
                    f"[OK] VERIFIZIERT: DMX '{dmx_text}' ≈ OCR '{ocr_check_text}' "
                    f"(normalisiert: '{dmtx_norm}' == '{ocr_norm}')"
                )
                method = "Verifiziert"
                confidence = 1.0
                is_verified = True
            else:
                logger.warning(
                    f"[WARN] ABWEICHUNG: DMX='{dmx_text}' vs OCR='{ocr_check_text}'. "
                    f"Nutze DMX ({dmx_result['method_detail']})."
                )
                confidence = 0.9

        ocr_display = ocr_text or ocr_partial_display
        if is_verified:
            ocr_display = ocr_check_text

        return {
            "success": True,
            "result": dmx_text,
            "method": method,
            "confidence": confidence,
            "dmtx_result": dmx_text,
            "ocr_result": ocr_display,
            "verified": is_verified,
            "ocr_partial_display": ocr_display,
        }

    # 2. Fall: DataMatrix blockiert + OCR vorhanden
    if ocr_status == "ok" and ocr_text:
        # Wenn ein Grid existiert, immer erst versuchen es zu rekonstruieren und zu verifizieren
        observed_grid = dmx_result.get("observed_grid")
        if observed_grid is not None:
            recon_result = _try_reconstruct(frame, ocr_text, ocr_conf, None)
            if recon_result is not None and recon_result.get("success"):
                recon_result["ocr_partial_display"] = ocr_text
                # Promote to Verifiziert if the reconstructed code matches normalized OCR text
                if recon_result["result"] == _normalize_ocr_confusions(ocr_text):
                    recon_result["method"] = "Verifiziert"
                    recon_result["confidence"] = 1.0
                    recon_result["verified"] = True
                return recon_result

        # 2a. OCR mit sehr hoher Konfidenz (≥0.98): Direkt akzeptieren
        if ocr_conf >= 0.98:
            logger.info(f"OCR-Direkt (≥0.98 Konfidenz): '{ocr_text}' (Conf={ocr_conf:.2f})")
            return {
                "success": True,
                "result": ocr_text,
                "method": "OCR",
                "confidence": ocr_conf,
                "dmtx_result": None,
                "ocr_result": ocr_text,
                "verified": False,
                "ocr_partial_display": ocr_text,
            }

        # 2c. Rekonstruktion fehlgeschlagen: OCR als Fallback
        observed_grid = dmx_result.get("observed_grid")
        if observed_grid is None:
            # Kein Grid extrahierbar → OCR ist einzige Quelle, akzeptiere ab 0.40
            if ocr_conf >= 0.40 and _is_valid_horden_code(ocr_text):
                logger.info(f"OCR-Fallback (kein Grid, gültiges Format): '{ocr_text}' (Conf={ocr_conf:.2f})")
                return {
                    "success": True,
                    "result": ocr_text,
                    "method": "OCR",
                    "confidence": ocr_conf,
                    "dmtx_result": None,
                    "ocr_result": ocr_text,
                    "verified": False,
                    "ocr_partial_display": ocr_text,
                }
        elif ocr_conf >= 0.75 and _is_valid_horden_code(ocr_text):
            # Grid existiert, aber Rekonstruktion fehlgeschlagen → OCR ab 0.75 bei format-validem Code
            logger.info(f"OCR-Fallback (Format-validiert ≥0.75, Rekonstruktion fehlgeschlagen): '{ocr_text}' (Conf={ocr_conf:.2f})")
            return {
                "success": True,
                "result": ocr_text,
                "method": "OCR",
                "confidence": ocr_conf,
                "dmtx_result": None,
                "ocr_result": ocr_text,
                "verified": False,
                "ocr_partial_display": ocr_text,
            }

        # 2d. OCR nicht akzeptiert
        logger.warning(
            f"OCR fand '{ocr_text}' (Conf: {ocr_conf:.2f}), aber Rekonstruktion fehlgeschlagen "
            f"und Konfidenz zu niedrig für direktes OCR-Fallback."
        )

    # 3. Fall: OCR hat Teillesung (3 Zeichen) -> Gitter-Vollrekonstruktion versuchen
    if ocr_status == "partial" and ocr_readable:
        recon_result = _try_reconstruct(frame, None, 0.0, ocr_readable, ocr_result.get("missing_positions"))
        if recon_result is not None and recon_result.get("success"):
            recon_result["ocr_partial_display"] = ocr_partial_display
            return recon_result

        # Partial-Fallback: Wenn Rekonstruktion fehlschlägt, versuche den wahrscheinlichsten Code
        # aus dem Partial zu erschließen
        if ocr_readable and len(ocr_readable) == 3:
            partial_norm, prefix_detected = _normalize_partial_3chars(ocr_readable)
            if prefix_detected:
                # Strategie 1: raw_candidate nutzen (4 Zeichen aus OCR vorhanden)
                raw_cand = ocr_result.get("raw_candidate")
                if raw_cand and len(raw_cand) == 4:
                    inferred = _clean_to_4chars(raw_cand)
                    if inferred is not None:
                        logger.info(
                            f"Partial-Inferenz: '{ocr_readable}' + raw='{raw_cand}' → '{inferred}'"
                        )
                        return {
                            "success": True,
                            "result": inferred,
                            "method": "Rekonstruiert",
                            "confidence": 0.85,
                            "dmtx_result": None,
                            "ocr_result": ocr_partial_display,
                            "verified": False,
                            "ocr_partial_display": ocr_partial_display,
                        }
                
                # Strategie 2: Fehlende Position bekannt → alle 10 Ziffern probieren
                # und per Grid-Matching den besten Kandidaten finden
                missing_pos = ocr_result.get("missing_positions", [])
                if len(missing_pos) == 1:
                    pos = missing_pos[0]
                    prefix = partial_norm[0]
                    digits_part = partial_norm[1:]
                    
                    cand_scores = []
                    
                    for d in '0123456789':
                        if pos == 0:
                            candidate = d + digits_part  # shouldn't happen with prefix_detected
                        elif pos <= 3:
                            code_digits = list(digits_part)
                            code_digits.insert(pos - 1, d)
                            candidate = prefix + ''.join(code_digits)
                        else:
                            continue
                        
                        if _is_valid_horden_code(candidate):
                            observed_grid = dmx_result.get("observed_grid")
                            if observed_grid is not None:
                                ref = _get_cached_reference_grid(candidate)
                                if ref is not None:
                                    score = float(np.sum(observed_grid == ref)) / 100.0
                                    cand_scores.append((candidate, score))
                    
                    if cand_scores:
                        cand_scores.sort(key=lambda x: x[1], reverse=True)
                        best_inferred, best_inferred_score = cand_scores[0]
                        second_score = cand_scores[1][1] if len(cand_scores) > 1 else 0.0
                        margin = best_inferred_score - second_score
                        
                        # Strenge Prüfung: Erfordere Mindest-Score (>=0.60) UND deutlichen Abstand (>=0.04) zum zweitbesten Ziffern-Kandidaten!
                        if best_inferred_score >= 0.60 and margin >= 0.04:
                            logger.info(
                                f"Partial-Inferenz (fehlende Pos {pos}): "
                                f"'{ocr_partial_display}' → '{best_inferred}' (Score={best_inferred_score:.2f}, Margin={margin:.2f})"
                            )
                            return {
                                "success": True,
                                "result": best_inferred,
                                "method": "Rekonstruiert",
                                "confidence": min(1.0, max(0.98, best_inferred_score)),
                                "dmtx_result": None,
                                "ocr_result": ocr_partial_display,
                                "verified": False,
                                "ocr_partial_display": ocr_partial_display,
                            }
                        else:
                            logger.warning(
                                f"Partial-Inferenz verworfen für '{ocr_partial_display}': "
                                f"Bester='{best_inferred}' Score={best_inferred_score:.2f}, Margin={margin:.2f} zu gering."
                            )

        logger.warning(f"Rekonstruktion mit OCR-Partial '{ocr_partial_display}' fehlgeschlagen.")
        return {
            "success": False,
            "result": f"Teilweise erkannt: {ocr_partial_display}",
            "method": "Fehler",
            "confidence": 0.0,
            "dmtx_result": None,
            "ocr_result": ocr_partial_display,
            "verified": False,
            "ocr_partial_display": ocr_partial_display,
        }

    # 4. Fall: OCR komplett fehlgeschlagen → reine Gitter-Rekonstruktion mit allen 4000 Codes
    logger.info("DMX blockiert + OCR fehlgeschlagen → versuche Gitter-Rekonstruktion mit allen gültigen Codes...")
    recon_result = _try_reconstruct(frame, None, 0.0, None)
    if recon_result is not None and recon_result.get("success"):
        recon_result["ocr_partial_display"] = recon_result["result"]
        recon_result["ocr_result"] = recon_result["result"]
        return recon_result

    # 5. Fall: Keine Erkennung möglich
    logger.warning("Weder DataMatrix noch OCR konnten etwas lesen.")
    return {
        "success": False,
        "result": "Kein Code erkannt.",
        "method": "Fehler",
        "confidence": 0.0,
        "dmtx_result": None,
        "ocr_result": None,
        "verified": False,
        "ocr_partial_display": None,
    }


def scan(frame: np.ndarray) -> dict:
    """
    Haupt-Scan-Funktion mit Dual-Validation (v4.0).
    Führt OCR- und DataMatrix-Erkennung parallel in Threads aus und kombiniert die Ergebnisse.
    
    Args:
        frame (np.ndarray): Das Graustufen- oder Farbbild der Kamera.
        
    Returns:
        dict: Das Endergebnis des Scans.
    """
    if frame is None:
        return {
            "success": False, "result": "Kein Bild vorhanden.",
            "method": "Fehler", "confidence": 0.0,
            "dmtx_result": None, "ocr_result": None, "verified": False,
            "ocr_partial_display": None,
        }

    h, w = frame.shape[:2]
    logger.info(f"Dual-Validation Scan v4.0 gestartet auf Bild mit {w}x{h} Pixeln.")

    ocr_result = None
    dmx_result = None

    # --- Timing: OCR und DMTX separat messen ---
    _t_ocr_start = time.time()
    _t_dmx_start = time.time()
    _t_ocr_end = _t_ocr_start
    _t_dmx_end = _t_dmx_start

    def _timed_ocr(frm):
        return _read_ocr_with_status(frm)

    def _timed_dmx(frm):
        return _scan_datamatrix_pipeline(frm)

    # Paralleles Ausführen von OCR und DMX
    with ThreadPoolExecutor(max_workers=2) as executor:
        _t_ocr_start = time.time()
        _t_dmx_start = time.time()
        future_ocr = executor.submit(_timed_ocr, frame)
        future_dmx = executor.submit(_timed_dmx, frame)

        try:
            ocr_result = future_ocr.result(timeout=30.0)
            _t_ocr_end = time.time()
        except Exception as e:
            _t_ocr_end = time.time()
            logger.warning(f"OCR-Thread Fehler: {e}")
            ocr_result = {
                "status": "failed", "text": None, "partial_display": None,
                "readable_chars": None, "confidence": 0.0,
                "readable_count": 0, "missing_positions": [],
            }

        try:
            dmx_result = future_dmx.result(timeout=10.0)
            _t_dmx_end = time.time()
        except Exception as e:
            _t_dmx_end = time.time()
            logger.warning(f"DataMatrix-Thread Fehler: {e}")
            dmx_result = {
                "status": "blocked", "text": None,
                "method_detail": "Thread-Fehler", "confidence": 0.0,
                "observed_grid": None,
            }

    # Ergebnisse mergen
    result = _merge_results(ocr_result, dmx_result, frame)

    # Internes Timing für den ScanLogger bereitstellen (nicht-brechend)
    result["_internal_timing"] = {
        "ocr_ms": int((_t_ocr_end - _t_ocr_start) * 1000),
        "dmtx_ms": int((_t_dmx_end - _t_dmx_start) * 1000),
    }

    logger.info(
        f"Scan Ergebnis: success={result['success']}, "
        f"method={result['method']}, result='{result['result']}'"
    )

    return result


def deskew_crop(image: np.ndarray, box: tuple[int, int, int, int], padding: int = 60) -> np.ndarray:
    """
    Schneidet das Etikett aus dem Bild aus und begradigt es (Deskewing),
    falls es rotiert/schräg ist.
    
    Kombiniert YOLO-Größenangaben mit klassischer Kanten-Winkelbestimmung.
    Garantiert IMMER ein gleichmäßiges Padding (Quiet Zone) für DataMatrix/OCR.
    
    Args:
        image: Das Originalbild (BGR).
        box: Die YOLO Bounding Box (x1, y1, x2, y2).
        padding: Großzügiges Padding um die Box, damit Ecken nicht abgeschnitten werden.
        
    Returns:
        Das begradigte und passend zugeschnittene Etikett-Bild.
    """
    h_img, w_img = image.shape[:2]
    x1, y1, x2, y2 = box
    
    # YOLO-Box Maße
    w_box = x2 - x1
    h_box = y2 - y1
    
    # Standard-Padded Crop für Fälle ohne Rotation (garantiert Quiet Zone)
    pad_safe = 25
    x1_padded = max(0, x1 - pad_safe)
    y1_padded = max(0, y1 - pad_safe)
    x2_padded = min(w_img, x2 + pad_safe)
    y2_padded = min(h_img, y2 + pad_safe)
    padded_crop_direct = image[y1_padded:y2_padded, x1_padded:x2_padded]
    
    # 1. Großzügiges Padding hinzufügen
    px1 = max(0, x1 - padding)
    py1 = max(0, y1 - padding)
    px2 = min(w_img, x2 + padding)
    py2 = min(h_img, y2 + padding)
    
    crop = image[py1:py2, px1:px2]
    if crop.size == 0:
        return padded_crop_direct
        
    # 2. Graustufen & Binarisierung
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()
        
    # Weichzeichnen zur Rauschunterdrückung
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Kanten-Erkennung (Canny) zur Erkennung paralleler Kanten/Ränder
    edges = cv2.Canny(blurred, 30, 100)
    
    # Morphologisches Schließen, um Kanten zu verbinden
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # 3. Alle Konturen durchsuchen, um den dominanten Winkel zu bestimmen
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return padded_crop_direct
        
    # Wir filtern Konturen nach einer gewissen Mindestgröße, um Rauschen zu vermeiden
    valid_rects = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > 1500: # Plausibler Kantenbereich
            rect = cv2.minAreaRect(c)
            valid_rects.append((area, rect))
            
    if not valid_rects:
        return padded_crop_direct
        
    # Nimm das Rechteck mit der größten Fläche (dominanteste Kantenstruktur)
    _, best_rect = max(valid_rects, key=lambda x: x[0])
    center, size, angle = best_rect
    w_rect, h_rect = size
    
    # Winkelkorrektur
    if w_rect < h_rect:
        w_rect, h_rect = h_rect, w_rect
        angle += 90.0
        
    if angle > 45.0:
        angle -= 90.0
    elif angle < -45.0:
        angle += 90.0
        
    # Wenn der Winkel extrem klein ist, reicht ein normaler Ausschnitt mit Quiet Zone
    if abs(angle) < 1.0:
        return padded_crop_direct
        
    # 5. Rotieren des Ausschnitts um den Mittelpunkt der YOLO-Box (relativ zum Ausschnitt)
    cx_orig = (x1 + x2) / 2.0
    cy_orig = (y1 + y2) / 2.0
    cx_crop = cx_orig - px1
    cy_crop = cy_orig - py1
    
    M = cv2.getRotationMatrix2D((cx_crop, cy_crop), angle, 1.0)
    
    # Rotation anwenden (mit borderReplicate um schwarze Ränder an den Ecken zu minimieren)
    rotated = cv2.warpAffine(crop, M, (crop.shape[1], crop.shape[0]), 
                              flags=cv2.INTER_CUBIC, 
                              borderMode=cv2.BORDER_REPLICATE)
    
    # 6. Begradigtes Etikett basierend auf der ursprünglichen YOLO-Box-Größe ausschneiden
    rx1 = int(cx_crop - (w_box / 2.0))
    ry1 = int(cy_crop - (h_box / 2.0))
    rx2 = int(cx_crop + (w_box / 2.0))
    ry2 = int(cy_crop + (h_box / 2.0))
    
    # Sicherheitsrand (Quiet Zone) hinzufügen
    rx1 = max(0, rx1 - pad_safe)
    ry1 = max(0, ry1 - pad_safe)
    rx2 = min(rotated.shape[1], rx2 + pad_safe)
    ry2 = min(rotated.shape[0], ry2 + pad_safe)
    
    final_crop = rotated[ry1:ry2, rx1:rx2]
    if final_crop.size == 0:
        return padded_crop_direct
        
    return final_crop
