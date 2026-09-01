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
import os
import sys
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

# --- Pipeline 3: Referenzbild-Datenbank (Template-Matching gegen generated_codes) ---
_REF_IMG_MATRIX = None    # np.ndarray (N, 10000) — Flattened binarisierte Referenzbilder
_REF_IMG_CODES = None     # list[str] — Code-Namen in gleicher Reihenfolge
_REF_IMG_SIZE = (100, 100)  # Normgröße für den Vergleich
_REF_IMG_DIR = None       # Pfad zum generated_codes Ordner
_REF_IMG_PACKED = None    # np.ndarray (N, 1250) — Gepackte Referenzbilder (packbits)


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
            gray = image.copy()
            
        gray = np.ascontiguousarray(gray)
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


# --- Trained AI Models (ONNX Runtime) & DataMatrix Generator ---
_pacc_session = None
_pacc_loaded = False

_unet_session = None
_unet_loaded = False

_PACC_PREFIX_CLASSES = ['A', 'B', 'P', 'W']


def _load_pacc():
    """
    Lädt das Position-Aware Char Classifier (PACC) ONNX Modell.
    """
    global _pacc_session, _pacc_loaded
    if not _pacc_loaded:
        try:
            import onnxruntime as ort
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'char_classifier.onnx')
            if os.path.exists(model_path):
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                opts.inter_op_num_threads = 1
                _pacc_session = ort.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])
                logger.info("PACC Char-Classifier (ONNX) erfolgreich geladen.")
            else:
                logger.warning(f"PACC ONNX Modell nicht gefunden: {model_path}")
        except Exception as e:
            logger.warning(f"Fehler beim Laden von PACC ONNX: {e}")
        _pacc_loaded = True
    return _pacc_session


def _load_unet_binarizer():
    """
    Lädt das MicroUNet DataMatrix-Binarisierer ONNX Modell.
    """
    global _unet_session, _unet_loaded
    if not _unet_loaded:
        try:
            import onnxruntime as ort
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'unet_binarizer.onnx')
            if os.path.exists(model_path):
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                opts.inter_op_num_threads = 1
                _unet_session = ort.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])
                logger.info("MicroUNet DataMatrix-Binarisierer (ONNX) erfolgreich geladen.")
            else:
                logger.warning(f"MicroUNet ONNX Modell nicht gefunden: {model_path}")
        except Exception as e:
            logger.warning(f"Fehler beim Laden von MicroUNet ONNX: {e}")
        _unet_loaded = True
    return _unet_session


def _predict_pacc(image: np.ndarray) -> tuple[str | None, float]:
    """
    Führt ultra-schnelle Klassifikation mit dem Position-Aware Char Classifier (PACC) durch (< 3ms).
    
    Args:
        image (np.ndarray): Text-Crop oder Bildbereich des Etiketts.
        
    Returns:
        tuple[str | None, float]: (erkoannter Code, Konfidenz) oder (None, 0.0)
    """
    session = _load_pacc()
    if session is None or image is None or image.size == 0:
        return None, 0.0

    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        resized = cv2.resize(gray, (128, 32), interpolation=cv2.INTER_AREA)
        inp = (resized.astype(np.float32) / 255.0)[np.newaxis, np.newaxis, :, :]

        outputs = session.run(None, {"input": inp})
        p0, p1, p2, p3 = outputs

        def softmax(x):
            e = np.exp(x - np.max(x))
            return e / e.sum()

        probs0 = softmax(p0[0])
        probs1 = softmax(p1[0])
        probs2 = softmax(p2[0])
        probs3 = softmax(p3[0])

        idx0 = int(np.argmax(probs0))
        idx1 = int(np.argmax(probs1))
        idx2 = int(np.argmax(probs2))
        idx3 = int(np.argmax(probs3))

        code = _PACC_PREFIX_CLASSES[idx0] + str(idx1) + str(idx2) + str(idx3)
        confs = [probs0[idx0], probs1[idx1], probs2[idx2], probs3[idx3]]
        avg_conf = float(np.mean(confs))

        if _HORDEN_PATTERN.match(code):
            return code, avg_conf
    except Exception as e:
        logger.debug(f"PACC Inferenz Fehler: {e}")

    return None, 0.0


def _preprocess_unet_binarize(image: np.ndarray) -> np.ndarray | None:
    """
    Wendet das MicroUNet Binarisierungsmodell auf ein Graustufenbild an (~8-10ms).
    Erzeugt ein sauberes binäres Schwarz-Weiß-Bild der DataMatrix.
    
    Args:
        image (np.ndarray): Graustufen- oder BGR-Bild.
        
    Returns:
        np.ndarray | None: Binarisiertes Bild (uint8 0/255) oder None.
    """
    session = _load_unet_binarizer()
    if session is None or image is None or image.size == 0:
        return None

    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape[:2]
        resized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
        inp = (resized.astype(np.float32) / 255.0)[np.newaxis, np.newaxis, :, :]

        outputs = session.run(None, {"input": inp})
        out_map = outputs[0][0, 0]

        sig_map = 1.0 / (1.0 + np.exp(-out_map))
        bin_128 = np.where(sig_map >= 0.5, 255, 0).astype(np.uint8)

        binary_out = cv2.resize(bin_128, (w, h), interpolation=cv2.INTER_NEAREST)
        return binary_out
    except Exception as e:
        logger.debug(f"MicroUNet Binarizer Fehler: {e}")
        return None


def _generate_datamatrix_image(data: str, output_path: str = None, cellsize: int = 10) -> np.ndarray | None:
    """
    Generiert einen synthetischen DataMatrix-Code (Schwarz/Weiß) mittels generate_datamatrix / pystrich.
    
    Args:
        data (str): Zu kodierender Text.
        output_path (str, optional): Zielpfad (.png oder .svg).
        cellsize (int): Modulgröße in Pixeln.
        
    Returns:
        np.ndarray | None: BGR-Bild als Numpy-Array oder None.
    """
    try:
        from generate_datamatrix import generate_datamatrix as gen_dm
        if output_path:
            gen_dm(data, output_path, cellsize=cellsize)
        
        from pystrich.datamatrix import DataMatrixEncoder
        encoder = DataMatrixEncoder(data)
        pil_img = encoder.get_pilimage(cellsize=cellsize).convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.debug(f"Fehler bei _generate_datamatrix_image: {e}")
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
    
    if not fast_mode:
        # Variante 3: Faded Contrast Boost – Perzentil-Stretching + Morphologie
        faded_boost = _preprocess_faded_contrast(gray)
        variants.append(("faded_boost", faded_boost))
        
        # Variante: Etch Denoise – Spezialfilter für verätzte/beschädigte Oberflächen
        etch_denoise = _preprocess_etch_denoise(gray)
        variants.append(("etch_denoise", etch_denoise))
        
        # Variante: Ridge Boost – Sobel & LoG Kanten-/Strukturverstärkung
        ridge_boost = _preprocess_ridge_enhancement(gray)
        variants.append(("ridge_boost", ridge_boost))
        
        # Variante: Sauvola W11 – feines lokales Schwellenwert-Fenster
        sauvola_w11 = _preprocess_sauvola(gray, window_size=11, k=0.2)
        variants.append(("sauvola_w11", sauvola_w11))
        
        # Variante: Niblack – Lokaler Niblack-Schwellenwert
        niblack_img = _preprocess_niblack(gray, window_size=21, k=-0.2)
        variants.append(("niblack", niblack_img))
        
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
        
        # Variante 8: Gamma 0.3 – für dunkle/unterbelichtete Bilder (starke Aufhellung)
        # Diagnose zeigt: Gamma 0.3 liest W031 auf dunklen Etiketten korrekt.
        lut_gamma03 = np.array([((i / 255.0) ** 0.3) * 255 for i in range(256)]).astype("uint8")
        gamma03 = cv2.LUT(gray, lut_gamma03)
        variants.append(("gamma_03", gamma03))
        
        # Variante 9: Gamma 0.5 + CLAHE – moderate Aufhellung mit Kontrastboost
        lut_gamma05 = np.array([((i / 255.0) ** 0.5) * 255 for i in range(256)]).astype("uint8")
        gamma05 = cv2.LUT(gray, lut_gamma05)
        clahe_gamma = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        variants.append(("gamma_05_clahe", clahe_gamma.apply(gamma05)))
    
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


def _preprocess_etch_denoise(image: np.ndarray) -> np.ndarray:
    """
    Spezial-Vorverarbeitung zur Entrauschung und Neutralisierung von Ätzspuren,
    Säureflecken, Glanzstellen und Lochfraß-Beschädigungen auf metallischen/geätzten Horden.
    
    Kombiniert Perzentil-Stretching, TopHat/BlackHat-Morphologie, Morphologisches Closing (3x3)
    und Unsharp-Masking.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 1. Perzentil-Histogramm-Stretching (1%-99%)
    p_low, p_high = np.percentile(gray, (1, 99))
    if p_high > p_low:
        stretched = np.clip((gray.astype(np.float32) - p_low) * (255.0 / (p_high - p_low)), 0, 255).astype(np.uint8)
    else:
        stretched = gray

    # 2. Kombinierte Morphologie: TopHat + BlackHat Boost für verätzte Oberflächen
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    tophat = cv2.morphologyEx(stretched, cv2.MORPH_TOPHAT, kernel_large)
    blackhat = cv2.morphologyEx(stretched, cv2.MORPH_BLACKHAT, kernel_large)

    enhanced = cv2.addWeighted(stretched, 1.0, tophat, 1.2, 0)
    enhanced = cv2.subtract(enhanced, (blackhat * 1.2).astype(np.uint8))

    # 3. Morphologisches Closing (3x3 Kernel) zur Schließung von Modullöchern/Ätzspuren
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel_close)

    # 4. Unsharp-Masking
    return _sharpen(closed)


def _preprocess_ridge_enhancement(image: np.ndarray) -> np.ndarray:
    """
    Schnelle Kanten- und Ridge-Strukturverstärkung für stark verblasste Ätzpunkte
    und schwache OCR-Buchstabenstriche (optimierte 8-Bit Sobel-Variante).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 1. Perzentil-Stretching (2%-98%)
    p_low, p_high = np.percentile(gray, (2, 98))
    if p_high > p_low:
        stretched = np.clip((gray.astype(np.float32) - p_low) * (255.0 / (p_high - p_low)), 0, 255).astype(np.uint8)
    else:
        stretched = gray

    # 2. Schnelle 8-Bit Sobel-Kantenextraktion
    grad_x = cv2.Sobel(stretched, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(stretched, cv2.CV_16S, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad_mag = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    # 3. Kanten-Boost
    enhanced = cv2.addWeighted(stretched, 0.7, grad_mag, 0.5, 0)
    return _sharpen(enhanced)




def _preprocess_sauvola(image: np.ndarray, window_size: int = 15, k: float = 0.2) -> np.ndarray:
    """
    Lokale Sauvola-Binarisierung zur Extraktion stark verbleichter Schriften und geätzter Punktraster.
    T = mean * (1 + k * (std / R - 1))
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


def _preprocess_niblack(image: np.ndarray, window_size: int = 21, k: float = -0.2) -> np.ndarray:
    """
    Lokale Niblack-Binarisierung zur Extraktion feiner Ätzstrukturen bei schwachem Kontrast.
    T = mean + k * std
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray_f = gray.astype(np.float32)
    mean = cv2.boxFilter(gray_f, cv2.CV_32F, (window_size, window_size))
    sqr_mean = cv2.boxFilter(gray_f**2, cv2.CV_32F, (window_size, window_size))
    std = np.sqrt(np.maximum(0, sqr_mean - mean**2))

    thresh = mean + k * std
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
            
    # Stelle 1 (erste Ziffer): 6 -> 0 Normalisierung
    # Auf dunklen/kontrastarmen Bildern verwechselt EasyOCR die 0 an 1. Ziffernstelle systematisch mit 6 (z.B. W631 -> W031).
    if chars[1] == '6':
        chars[1] = '0'
        
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


def _repair_l_finder(binary_image: np.ndarray) -> np.ndarray:
    """
    Repariert unterbrochene oder durch Ätzung angefressene L-Finder-Schenkel.
    Verwendet richtungsgebundene morphologische Schließung (horizontal: Kernel (1, 7), vertikal: Kernel (7, 1)).
    """
    if binary_image is None or binary_image.size == 0:
        return binary_image

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))

    closed_h = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel_h)
    closed_v = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel_v)

    repaired = cv2.bitwise_and(closed_h, closed_v)
    return repaired


def _intersection_of_lines(line1: tuple, line2: tuple) -> np.ndarray | None:
    """Berechnet den Schnittpunkt zweier Geraden im R2 (Punkt + Richtungsvektor)."""
    (vx1, vy1, x1, y1) = line1
    (vx2, vy2, x2, y2) = line2

    denom = vx1 * vy2 - vy1 * vx2
    if abs(denom) < 1e-6:
        return None

    t = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / denom
    intersection_x = x1 + t * vx1
    intersection_y = y1 + t * vy1
    return np.array([intersection_x, intersection_y], dtype=np.float32)


def _refine_corners_ransac(image: np.ndarray, initial_corners: np.ndarray) -> np.ndarray:
    """
    Refiniert grobe Ecken (von minAreaRect) durch RANSAC-Linienanpassung
    an die 4 Außenkanten des DataMatrix-Kandidaten.
    """
    if initial_corners is None or len(initial_corners) != 4:
        return initial_corners

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    edges = cv2.Canny(gray, 50, 150)
    
    center = initial_corners.mean(axis=0)
    angles = np.arctan2(initial_corners[:, 1] - center[1], initial_corners[:, 0] - center[0])
    sorted_idx = np.argsort(angles)
    sorted_pts = initial_corners[sorted_idx]

    lines = []
    num_pts = len(sorted_pts)
    for i in range(num_pts):
        pt1 = sorted_pts[i]
        pt2 = sorted_pts[(i + 1) % num_pts]

        mask = np.zeros_like(edges)
        cv2.line(mask, tuple(pt1.astype(int)), tuple(pt2.astype(int)), 255, 15)
        edge_pts = np.column_stack(np.where((edges > 0) & (mask > 0)))

        if len(edge_pts) >= 10:
            pts_xy = np.float32(edge_pts[:, [1, 0]])
            fit = cv2.fitLine(pts_xy, cv2.DIST_HUBER, 0, 0.01, 0.01)
            vx, vy, x, y = float(fit[0][0]), float(fit[1][0]), float(fit[2][0]), float(fit[3][0])
            lines.append((vx, vy, x, y))
        else:
            vx = float(pt2[0] - pt1[0])
            vy = float(pt2[1] - pt1[1])
            norm = float(np.hypot(vx, vy) + 1e-6)
            lines.append((vx / norm, vy / norm, float(pt1[0]), float(pt1[1])))

    refined_corners = []
    for i in range(4):
        line1 = lines[i]
        line2 = lines[(i + 1) % 4]
        intersect = _intersection_of_lines(line1, line2)
        if intersect is not None:
            refined_corners.append(intersect)
        else:
            refined_corners.append(sorted_pts[(i + 1) % 4])

    refined = np.array(refined_corners, dtype=np.float32)

    if np.isnan(refined).any() or np.isinf(refined).any():
        return initial_corners

    if np.max(np.abs(refined - sorted_pts)) > 25.0:
        return initial_corners

    return refined


def _orient_corners(gray: np.ndarray, corners: np.ndarray, strict: bool = True) -> np.ndarray | None:
    """
    Bestimmt die korrekte Ausrichtung der 4 Ecken eines DataMatrix-Codes.
    Testet alle 4 Rotationen und bewertet L-Finder und Timing-Muster mit RANSAC-Refinement und L-Reparatur.
    
    Args:
        gray (np.ndarray): Graustufenbild.
        corners (np.ndarray): Die 4 Ecken des Kandidaten.
        strict (bool): Wenn False, wird der Mindest-Score für Akzeptanz gesenkt.
        
    Returns:
        np.ndarray | None: Die sortierten/ausgerichteten Ecken oder None.
    """
    corners = _refine_corners_ransac(gray, corners)

    if corners is None or not isinstance(corners, np.ndarray) or corners.shape != (4, 2):
        return None
    corners = np.float32(corners)

    warp_size = DMTX_WARP_SIZE
    grid = DMTX_GRID_SIZE
    cell = DMTX_CELL_PX
    dst_pts = np.float32([
        [0, 0], [warp_size, 0], [warp_size, warp_size], [0, warp_size]
    ])

    best_score = -1
    best_corners = None

    center = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    sorted_idx = np.argsort(angles)
    sorted_corners = corners[sorted_idx]

    for rotation in range(4):
        rotated = np.float32(np.roll(sorted_corners, rotation, axis=0))
        M = cv2.getPerspectiveTransform(rotated, dst_pts)
        warped = cv2.warpPerspective(gray, M, (warp_size, warp_size),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)

        _, warped_bin = cv2.threshold(warped, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphologische L-Finder Reparatur für verätzte Kanten
        repaired_bin = _repair_l_finder(warped_bin)

        cells = np.zeros((grid, grid), dtype=np.uint8)
        for row in range(grid):
            for col in range(grid):
                cy = row * cell + cell // 2
                cx = col * cell + cell // 2
                region = repaired_bin[cy - 3:cy + 3, cx - 3:cx + 3]
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

    # Akzeptanzgrenze (entspannt für verätzte L-Finder)
    min_score = 20 if strict else 16
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
    if gray is None or gray.size == 0 or corners is None or not isinstance(corners, np.ndarray) or corners.shape != (4, 2):
        return None
    corners = np.float32(corners)

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
    elif binarization_method == "etch_denoise":
        etch_warp = _preprocess_etch_denoise(warped)
        _, warped_bin = cv2.threshold(etch_warp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif binarization_method == "ridge_boost":
        ridge_warp = _preprocess_ridge_enhancement(warped)
        _, warped_bin = cv2.threshold(ridge_warp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif binarization_method == "sauvola_w11":
        warped_bin = _preprocess_sauvola(warped, window_size=11, k=0.2)
    elif binarization_method == "sauvola_w21":
        warped_bin = _preprocess_sauvola(warped, window_size=21, k=0.2)
    elif binarization_method == "niblack":
        warped_bin = _preprocess_niblack(warped, window_size=21, k=-0.2)
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

    # L-Finder Plausibilitätscheck (entspannt für verätzte L-Linien)
    min_l = 5 if strict_l_finder else 4
    if np.sum(cells[:, 0] == 0) < min_l:
        logger.debug(f"Rekonstruktion: L-Finder links nicht ausreichend nach Sampling ({np.sum(cells[:, 0] == 0)}/10).")
        return None
    if np.sum(cells[grid - 1, :] == 0) < min_l:
        logger.debug(f"Rekonstruktion: L-Finder unten nicht ausreichend nach Sampling ({np.sum(cells[grid - 1, :] == 0)}/10).")
        return None

    logger.info("Rekonstruktion: 10x10 Binärmatrix erfolgreich extrahiert.")
    return cells


def _warp_and_sample_soft(gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    Entzerrt die Ecken perspektivisch und extrahiert eine kontinuierliche 10x10 Float-Matrix
    P in [0.0, 1.0]^{10 x 10} von Helligkeitsintensitäten (0.0 = schwarz, 1.0 = weiß).
    
    Args:
        gray (np.ndarray): Graustufenbild.
        corners (np.ndarray): Die 4 ausgerichteten Ecken.
        
    Returns:
        np.ndarray: Die 10x10 Float-Matrix mit kontinuierlichen Zellederivaten in [0.0, 1.0].
    """
    if gray is None or gray.size == 0 or corners is None or not isinstance(corners, np.ndarray) or corners.shape != (4, 2):
        return np.zeros((DMTX_GRID_SIZE, DMTX_GRID_SIZE), dtype=np.float32)
    corners = np.float32(corners)

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

    p_low, p_high = np.percentile(warped, (2, 98))
    if p_high > p_low:
        warped_norm = np.clip((warped.astype(np.float32) - p_low) / (p_high - p_low), 0.0, 1.0)
    else:
        warped_norm = warped.astype(np.float32) / 255.0

    soft_matrix = np.zeros((grid, grid), dtype=np.float32)
    for row in range(grid):
        for col in range(grid):
            cy = row * cell + cell // 2
            cx = col * cell + cell // 2
            half = max(2, cell // 3)
            region = warped_norm[cy - half:cy + half, cx - half:cx + half]
            if region.size == 0:
                soft_matrix[row, col] = 0.5
            else:
                soft_matrix[row, col] = float(np.mean(region))

    return soft_matrix


def _generate_synthetic_dmtx(cells: np.ndarray) -> np.ndarray:
    if cells is None or not isinstance(cells, np.ndarray) or cells.ndim != 2:
        return np.full((120, 120), 255, dtype=np.uint8)
    grid = cells.shape[0]
    cell_px = DMTX_CELL_PX
    quiet_zone = cell_px

    img_size = grid * cell_px + 2 * quiet_zone
    img = np.full((img_size, img_size), 255, dtype=np.uint8)

    for row in range(grid):
        for col in range(grid):
            x0 = quiet_zone + col * cell_px
            y0 = quiet_zone + row * cell_px
            val = cells[row, col]
            if isinstance(val, (float, np.floating)):
                color = int(np.clip(val * 255.0, 0, 255))
            else:
                color = 0 if val == 0 else 255
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
            M = cv2.getPerspectiveTransform(np.float32(oriented), dst_pts)
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


def _match_soft_grid_matrix(soft_matrix: np.ndarray, candidate_codes: set[str] | list[str] = None) -> tuple[str | None, float, float]:
    """
    Vektorisiertes Soft-Matching einer 10x10 Grauwert-Wahrscheinlichkeitsmatrix P in [0.0, 1.0]
    gegen die 4.000 vorberechneten DataMatrix-Referenzgitter R.
    
    Score(C) = 1.0 - mean(|P - R^(C)|)
    
    Returns:
        tuple[str | None, float, float]: (bester_code, bester_score, abstand_zum_zweitbesten)
    """
    all_codes, all_matrix = _get_precomputed_4000_grid_matrix()

    if soft_matrix is None or soft_matrix.shape != (10, 10):
        return None, 0.0, 0.0

    soft_flat = soft_matrix.flatten().astype(np.float32)
    ref_float = all_matrix.astype(np.float32)

    diffs = np.abs(ref_float - soft_flat)
    scores = 1.0 - np.mean(diffs, axis=1)

    if candidate_codes and len(candidate_codes) < 4000:
        cand_list = list(candidate_codes)
        indices = [all_codes.index(c) for c in cand_list if c in all_codes]
        if not indices:
            return None, 0.0, 0.0
        cand_indices = np.array(indices)
        sub_scores = scores[cand_indices]
        sorted_arg = np.argsort(sub_scores)[::-1]

        best_idx = cand_indices[sorted_arg[0]]
        best_cand = all_codes[best_idx]
        best_score = float(sub_scores[sorted_arg[0]])

        second_score = float(sub_scores[sorted_arg[1]]) if len(sorted_arg) > 1 else 0.0
        margin = best_score - second_score
    else:
        top2 = np.argpartition(scores, -2)[-2:]
        top2_sorted = top2[np.argsort(scores[top2])[::-1]]

        best_idx = top2_sorted[0]
        second_idx = top2_sorted[1]

        best_cand = all_codes[best_idx]
        best_score = float(scores[best_idx])
        margin = float(scores[best_idx] - scores[second_idx])

    return best_cand, best_score, margin


def _template_match_candidates(frame: np.ndarray, candidates: list[str]) -> tuple[str | None, float, float]:
    """
    Multi-Scale Template-Matching: Generiert für jeden Kandidaten ein synthetisches
    DataMatrix-Bild und vergleicht es per normalisierter Kreuzkorrelation (TM_CCOEFF_NORMED)
    mit dem Originalbild. Dies ist die letzte Verteidigungslinie für extrem beschädigte
    Bilder, bei denen keine Grid-Extraktion möglich ist.
    
    Args:
        frame: Das Eingangsbild (Graustufen oder BGR).
        candidates: Liste der infrage kommenden Horden-Codes.
        
    Returns:
        tuple[str | None, float, float]: (bester_kandidat, score, margin_zum_zweitbesten)
    """
    if not candidates or frame is None:
        return None, 0.0, 0.0

    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    h, w = gray.shape[:2]

    # Kontrastverstärkung für beschädigte Bilder
    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Vorberechnung: Synthetische Templates für alle Kandidaten
    templates = {}
    for cand in candidates:
        ref_grid = _generate_reference_grid(cand)
        if ref_grid is not None:
            synth = _generate_synthetic_dmtx(ref_grid)
            templates[cand] = synth

    if not templates:
        return None, 0.0, 0.0

    # Multi-Scale Template-Matching
    # Die DataMatrix könnte verschiedene Größen im Bild haben
    best_scores = {}
    
    # Schätze die ungefähre DataMatrix-Größe aus dem Bildausschnitt
    # Typisch: DMX nimmt ca. 20-60% der Bildbreite ein
    min_tmpl_size = max(30, int(min(w, h) * 0.15))
    max_tmpl_size = min(int(min(w, h) * 0.8), max(w, h))
    
    # 8-10 Skalierungsstufen, gleichmäßig logarithmisch verteilt
    n_scales = 8
    scales = np.linspace(min_tmpl_size, max_tmpl_size, n_scales).astype(int)

    for cand, synth in templates.items():
        cand_best_score = -1.0
        synth_h, synth_w = synth.shape[:2]
        
        for target_size in scales:
            scale_factor = target_size / synth_w
            new_w = int(synth_w * scale_factor)
            new_h = int(synth_h * scale_factor)
            
            if new_w >= w or new_h >= h or new_w < 20 or new_h < 20:
                continue
            
            resized = cv2.resize(synth, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Template-Matching mit normalisierter Kreuzkorrelation
            for img_variant in [enhanced, gray]:
                result = cv2.matchTemplate(img_variant, resized, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                if max_val > cand_best_score:
                    cand_best_score = max_val

        best_scores[cand] = cand_best_score

    if not best_scores:
        return None, 0.0, 0.0

    # Sortiere nach Score
    sorted_cands = sorted(best_scores.items(), key=lambda x: x[1], reverse=True)
    best_cand, best_score = sorted_cands[0]
    second_score = sorted_cands[1][1] if len(sorted_cands) > 1 else 0.0
    margin = best_score - second_score

    logger.info(
        f"Template-Matching: Bester='{best_cand}' Score={best_score:.3f}, "
        f"Zweitbester='{sorted_cands[1][0] if len(sorted_cands) > 1 else '-'}' "
        f"Score={second_score:.3f}, Margin={margin:.3f}"
    )

    return best_cand, float(best_score), float(margin)



def _extract_observed_grid_soft(frame: np.ndarray) -> np.ndarray | None:
    """
    Sucht nach dem Etikett und extrahiert die kontinuierliche 10x10 Soft-Intensity-Matrix.
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    h, w = gray.shape[:2]
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
    candidates = []
    seen_centers = []

    for clip_limit in [3.0, 8.0, 15.0]:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        label_enhanced = clahe.apply(label_crop)

        _, crop_bin_otsu = cv2.threshold(label_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        crop_inv = cv2.bitwise_not(crop_bin_otsu)

        for k_size in [35, 25, 15, 9]:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
            closed = cv2.morphologyEx(crop_inv, cv2.MORPH_CLOSE, kernel, iterations=2)
            cnts, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            for c in cnts:
                area = cv2.contourArea(c)
                if area < 400 or area > (crop_h * crop_w) * 0.70:
                    continue
                rect = cv2.minAreaRect(c)
                rw, rh = rect[1]
                if rw == 0 or rh == 0 or max(rw, rh) / min(rw, rh) > 2.0:
                    continue
                center = rect[0]
                if any(np.sqrt((center[0] - s[0])**2 + (center[1] - s[1])**2) < 20 for s in seen_centers):
                    continue
                seen_centers.append(center)
                candidates.append((c, area, rect))

    candidates.sort(key=lambda x: x[1], reverse=True)

    for contour, area, rect in candidates:
        box = cv2.boxPoints(rect)
        corners = np.float32(box)
        oriented = _orient_corners(label_crop, corners, strict=False)
        if oriented is not None:
            return _warp_and_sample_soft(label_crop, oriented)

    return None


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


def _generate_10_candidates_from_partial(ocr_partial: str) -> list[str]:
    """
    Erzeugt die 10 infrage kommenden 4-Zeichen Horden-Codes bei 3/4 erkannten OCR-Zeichen.
    """
    if not ocr_partial:
        return []

    cands = []
    if len(ocr_partial) == 4 and '?' in ocr_partial:
        q_pos = ocr_partial.find('?')
        if q_pos == 0:
            for prefix in VALID_PREFIXES:
                cand = prefix + ocr_partial[1:]
                if _is_valid_horden_code(cand):
                    cands.append(cand)
        else:
            for d in '0123456789':
                cand = ocr_partial[:q_pos] + d + ocr_partial[q_pos + 1:]
                if _is_valid_horden_code(cand):
                    cands.append(cand)
        return cands

    partial_norm, prefix_detected = _normalize_partial_3chars(ocr_partial)
    if prefix_detected:
        prefix = partial_norm[0]
        digits = partial_norm[1:]
        for insert_pos in range(3):
            for d in '0123456789':
                code_digits = digits[:insert_pos] + d + digits[insert_pos:]
                cand = prefix + code_digits
                if _is_valid_horden_code(cand) and cand not in cands:
                    cands.append(cand)
    else:
        for prefix in VALID_PREFIXES:
            cand = prefix + partial_norm
            if _is_valid_horden_code(cand) and cand not in cands:
                cands.append(cand)

    return cands


def _cross_validate_ocr_dmtx(ocr_partial: str, observed_grid: np.ndarray) -> tuple[str | None, float]:
    """
    Bidirektionale Cross-Validation zwischen OCR-Teilergebnis und DataMatrix-Grid.
    """
    if not ocr_partial or observed_grid is None or observed_grid.shape != (10, 10):
        return None, 0.0

    cands = _generate_10_candidates_from_partial(ocr_partial)
    if not cands:
        return None, 0.0

    all_codes, all_matrix = _get_precomputed_4000_grid_matrix()
    obs_flat = observed_grid.flatten()

    best_cand = None
    best_score = -1.0

    for cand in cands:
        if cand in all_codes:
            idx = all_codes.index(cand)
            ref_flat = all_matrix[idx]
            match_score = np.mean(ref_flat == obs_flat)
            if match_score > best_score:
                best_score = match_score
                best_cand = cand

    if best_score >= 0.70:
        logger.info(f"Cross-Validation ERFOLGREICH: Teilcode '{ocr_partial}' -> DataMatrix Match '{best_cand}' (Score: {best_score:.1%})")
        return best_cand, float(best_score)

    return None, 0.0


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

    # Gitter-Extraktion mit verschiedenen Binarisierungsmethoden (inkl. CLAHE+Otsu, EtchDenoise, RidgeBoost, Sauvola, Niblack)
    observed_variants = {}
    for method in ["otsu", "adaptive", "mean", "clahe_otsu", "etch_denoise", "ridge_boost", "sauvola_w11", "sauvola_w21", "niblack"]:
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

    # Soft-Probability Vector Matching über kontinuierliche Grauwert-Zellintensitäten
    soft_grid = _extract_observed_grid_soft(frame)
    if soft_grid is not None:
        soft_cand, soft_score, soft_margin = _match_soft_grid_matrix(soft_grid, candidates)
        if soft_cand is not None:
            logger.info(f"Rekonstruktion (Soft-Matching): Bester='{soft_cand}' Score={soft_score:.1%}, Abstand={soft_margin:.1%}")
            if soft_score > best_overall_score:
                best_overall_score = soft_score
                best_overall_margin = soft_margin
                best_candidate = soft_cand
                best_method = "Soft-Matching"

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
    if not is_valid and ocr_partial and best_overall_score >= 0.65 and best_overall_margin >= 0.05:
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
    Versucht ein Bild per zxing-cpp und pylibdmtx als DataMatrix-Code zu dekodieren.
    zxing-cpp wird bevorzugt; pylibdmtx wird auf binarisierten Bildern übersprungen,
    um C-Segmentation-Faults der C-Bibliothek libdmtx zu verhindern.
    """
    if pil_img is None:
        return None

    try:
        img_arr = np.array(pil_img)
        if img_arr.size == 0:
            return None

        # 1. zxing-cpp bevorzugen (blitzschnell, C++20 memory-safe)
        zx_res = _try_zxing_dmtx(img_arr)
        if zx_res is not None:
            return zx_res

        # 2. pylibdmtx-Schutz: Binarisierte Bilder (≤10 eindeutige Grauwerte) überspringen,
        # da libdmtx.dll auf Stufenkanten binärer Erosionen C-Heap/Stack-Faults verursacht.
        unique_vals = len(np.unique(img_arr))
        if unique_vals <= 10:
            return None

        if not _load_dmtx():
            return None

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

        # MicroUNet KI-Binarisierer versuchen (speziell für geätzte/verblassende Codes)
        unet_img = _preprocess_unet_binarize(gray)
        if unet_img is not None:
            zxing_result = _try_zxing_dmtx(unet_img)
            if zxing_result is not None:
                logger.info(f"DataMatrix gefunden (zxing-cpp + MicroUNet Binarisierer): {zxing_result}")
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

        etch_img = _preprocess_etch_denoise(gray)
        zxing_result = _try_zxing_dmtx(etch_img)
        if zxing_result is not None:
            logger.info(f"DataMatrix gefunden (zxing-cpp + EtchDenoise): {zxing_result}")
            return zxing_result

        ridge_img = _preprocess_ridge_enhancement(gray)
        zxing_result = _try_zxing_dmtx(ridge_img)
        if zxing_result is not None:
            logger.info(f"DataMatrix gefunden (zxing-cpp + RidgeBoost): {zxing_result}")
            return zxing_result

        sauvola_w11_img = _preprocess_sauvola(gray, window_size=11, k=0.2)
        zxing_result = _try_zxing_dmtx(sauvola_w11_img)
        if zxing_result is not None:
            logger.info(f"DataMatrix gefunden (zxing-cpp + Sauvola W11): {zxing_result}")
            return zxing_result

        niblack_img = _preprocess_niblack(gray, window_size=21, k=-0.2)
        zxing_result = _try_zxing_dmtx(niblack_img)
        if zxing_result is not None:
            logger.info(f"DataMatrix gefunden (zxing-cpp + Niblack): {zxing_result}")
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
            if dmx_crop is None or dmx_crop.size == 0 or dmx_crop.shape[0] < 5 or dmx_crop.shape[1] < 5:
                continue
            if len(dmx_crop.shape) == 3:
                dmx_crop = cv2.cvtColor(dmx_crop, cv2.COLOR_BGR2GRAY)
            dmx_crop = np.ascontiguousarray(dmx_crop)
            
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

        print("[DEBUG] _read_datamatrix: step 5 filter variants...")
        for name, make_fn in variants:
            print(f"[DEBUG] filter variant '{name}' testing...")
            try:
                pil_img = make_fn()
                result = _try_decode_dmtx(pil_img, timeout_ms=300)
                if result is not None:
                    logger.info(f"DataMatrix gefunden (Label-Fallback: {name}): {result}")
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
        
        # ===== STUFE 0: PACC Neural Char-Classifier Fast-Path (< 3ms) =====
        pacc_code, pacc_conf = _predict_pacc(ocr_zone)
        if pacc_code is not None and pacc_conf >= 0.70:
            logger.info(f"PACC Fast-Path OCR erfolgreich: '{pacc_code}' (Konfidenz: {pacc_conf:.2f})")
            return {
                "status": "ok",
                "text": pacc_code,
                "partial_display": pacc_code,
                "readable_chars": pacc_code,
                "confidence": pacc_conf,
                "readable_count": 4,
                "missing_positions": [],
                "raw_candidate": pacc_code,
            }
        
        reader = _load_ocr()
        
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
                    # Sicherheitscheck 1: Wenn Präfix erkannt, darf Position 0 nicht fehlen
                    if prefix_detected and 0 in missing_pos:
                        missing_pos = [1]
                    # Sicherheitscheck 2: Wenn KEIN Präfix erkannt (3 Ziffern), MUSS Position 0 (Buchstabe) fehlen!
                    elif not prefix_detected:
                        missing_pos = [0]
                    
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

    # Horden-Code Plausibilität: Wenn 3 Zeichen gelesen wurden, aber kein Präfix-Buchstabe vorhanden ist,
    # liegt die Lücke garantiert an Position 0 (Buchstabe fehlt), nicht am Ende!
    if len(high_conf_chars) == 3:
        first_ch = high_conf_chars[0]
        has_prefix = (first_ch in VALID_PREFIXES or first_ch in _FUZZY_PREFIX_MAP or first_ch in _DIGIT_TO_LETTER)
        if not has_prefix:
            display_chars = ['?'] + [c for c in high_conf_chars[:3]]
            missing_positions = [0]

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
    Stellt sicher, dass bei 3 Ziffern ohne Buchstabe das '?' an Pos 0 steht.
    """
    if not readable_chars:
        return "????"

    clean_chars = ''.join(c for c in readable_chars if c in ALLOWED_CHARS)

    # Spezialfall: 3 Ziffern ohne Präfix-Buchstabe (z.B. "032" oder "103")
    # Der Buchstabe an Pos 0 fehlt zwingend -> Format ist "?032" bzw. "?103"
    if len(clean_chars) == 3 and not (clean_chars[0] in VALID_PREFIXES or clean_chars[0] in _FUZZY_PREFIX_MAP or clean_chars[0] in _DIGIT_TO_LETTER):
        return "?" + clean_chars

    if not missing_positions:
        return clean_chars[:4] if len(clean_chars) >= 4 else clean_chars

    result = list("????")
    char_idx = 0
    for pos in range(4):
        if pos in missing_positions:
            result[pos] = '?'
        else:
            if char_idx < len(clean_chars):
                result[pos] = clean_chars[char_idx]
                char_idx += 1

    res_str = ''.join(result)
    if res_str[0] in '0123456789' and len(clean_chars) == 3:
        return "?" + clean_chars

    return res_str


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


# =============================================================================
# Pipeline 3: Referenzbild-Abgleich (Hamming-Distanz gegen generated_codes)
# =============================================================================

def _get_ref_img_dir() -> str:
    """
    Ermittelt den Pfad zum Ordner 'generated_codes' relativ zum Skript-/EXE-Verzeichnis.
    """
    global _REF_IMG_DIR
    if _REF_IMG_DIR is None:
        if getattr(sys, 'frozen', False):
            base = os.path.dirname(sys.executable)
        else:
            base = os.path.dirname(os.path.abspath(__file__))
        _REF_IMG_DIR = os.path.join(base, 'generated_codes')
    return _REF_IMG_DIR


def _load_reference_images() -> tuple[list[str], np.ndarray]:
    """
    Lazy-Loading: Liest alle PNG-Referenzbilder aus 'generated_codes/' und
    speichert sie als binarisierte, flattened Vektoren in einer NumPy-Matrix.
    
    Returns:
        tuple[list[str], np.ndarray]: (Liste der Code-Namen, Matrix (N, 10000) uint8)
    """
    global _REF_IMG_MATRIX, _REF_IMG_CODES
    if _REF_IMG_MATRIX is not None:
        return _REF_IMG_CODES, _REF_IMG_MATRIX

    ref_dir = _get_ref_img_dir()
    if not os.path.isdir(ref_dir):
        logger.warning(f"Referenzbild-Ordner '{ref_dir}' nicht gefunden. Pipeline 3 deaktiviert.")
        _REF_IMG_CODES = []
        _REF_IMG_MATRIX = np.empty((0, _REF_IMG_SIZE[0] * _REF_IMG_SIZE[1]), dtype=np.uint8)
        return _REF_IMG_CODES, _REF_IMG_MATRIX

    t0 = time.time()
    codes = []
    rows = []
    norm_w, norm_h = _REF_IMG_SIZE

    # Sortierte Liste aller PNG-Dateien
    png_files = sorted([f for f in os.listdir(ref_dir) if f.lower().endswith('.png')])

    for fname in png_files:
        code = os.path.splitext(fname)[0]  # z.B. "W002"
        fpath = os.path.join(ref_dir, fname)
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # Auf Normgröße skalieren
        resized = cv2.resize(img, (norm_w, norm_h), interpolation=cv2.INTER_AREA)
        # Otsu-Binarisierung (0 oder 255 → 0 oder 1)
        _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_01 = (binary // 255).astype(np.uint8)  # 0=schwarz, 1=weiß
        rows.append(binary_01.flatten())
        codes.append(code)

    if rows:
        _REF_IMG_MATRIX = np.array(rows, dtype=np.uint8)
    else:
        _REF_IMG_MATRIX = np.empty((0, norm_w * norm_h), dtype=np.uint8)
    _REF_IMG_CODES = codes

    dt = time.time() - t0
    logger.info(
        f"Pipeline 3: {len(codes)} Referenzbilder aus '{ref_dir}' geladen "
        f"(Matrix: {_REF_IMG_MATRIX.shape}, {dt:.2f}s)"
    )
    return _REF_IMG_CODES, _REF_IMG_MATRIX


def _extract_dmx_region(frame: np.ndarray) -> list[np.ndarray]:
    """
    Extrahiert den quadratischen DataMatrix-Bereich aus dem Kamerabild.
    Sucht nach dem größten annähernd quadratischen Konturobjekt und liefert
    alle 4 Rotationsvarianten zurück (da die Ecksortierung nicht eindeutig ist).
    Falls keine Kontur gefunden wird, wird das gesamte Bild als Fallback skaliert.
    
    Returns:
        list[np.ndarray]: Liste von binarisierten 100×100-Varianten (0/1), oder leere Liste.
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    h, w = gray.shape[:2]
    norm_w, norm_h = _REF_IMG_SIZE
    variants = []

    # Binarisieren und Konturen finden
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = cv2.bitwise_not(binary)

    # Morphologisches Closing für fragmentierte DMX-Module
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best_contour = None
    best_area = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area < 400 or area > (h * w) * 0.70:
            continue

        rect = cv2.minAreaRect(c)
        rect_w, rect_h = rect[1]
        if rect_w == 0 or rect_h == 0:
            continue
        aspect = max(rect_w, rect_h) / min(rect_w, rect_h)
        if aspect > 1.6:
            continue

        if area > best_area:
            best_area = area
            best_contour = c

    if best_contour is not None:
        rect = cv2.minAreaRect(best_contour)
        box = cv2.boxPoints(rect)
        box = np.float32(box)
        # Sortiere Punkte nach Winkel
        center = box.mean(axis=0)
        angles = np.arctan2(box[:, 1] - center[1], box[:, 0] - center[0])
        sorted_idx = np.argsort(angles)
        sorted_box = box[sorted_idx]

        dst = np.float32([[0, 0], [norm_w, 0], [norm_w, norm_h], [0, norm_h]])
        M = cv2.getPerspectiveTransform(sorted_box, dst)
        warped = cv2.warpPerspective(gray, M, (norm_w, norm_h))
        _, warped_bin = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        base = (warped_bin // 255).astype(np.uint8)

        # Alle 4 Rotationsvarianten erzeugen (0°, 90°, 180°, 270°)
        for k in range(4):
            rotated = np.rot90(base, k)
            variants.append(rotated)

    # Fallback: Gesamtes Bild skalieren (z.B. wenn Bild bereits ein sauberer DMX-Ausschnitt ist)
    resized = cv2.resize(gray, (norm_w, norm_h), interpolation=cv2.INTER_AREA)
    _, resized_bin = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fallback = (resized_bin // 255).astype(np.uint8)
    # Auch hier alle 4 Rotationen
    for k in range(4):
        rotated = np.rot90(fallback, k)
        variants.append(rotated)

    return variants


def _scan_reference_image_pipeline(frame: np.ndarray) -> dict:
    """
    Pipeline 3: Vergleicht den DataMatrix-Bereich im Kamerabild
    gegen alle vorgenerierten Referenzbilder per vektorisierter Hamming-Distanz.
    Testet automatisch alle Rotationsvarianten und wählt den besten Match.
    
    Optimierung: np.packbits komprimiert 10.000 Pixel → 1.250 Bytes pro Vektor,
    XOR + popcount auf gepackten Bits → 8× schneller als naive Variante.
    
    Args:
        frame (np.ndarray): Das Graustufen- oder Farbbild.
        
    Returns:
        dict: Pipeline-Ergebnis mit status, text, confidence, method_detail.
    """
    blocked_result = {
        "status": "blocked",
        "text": None,
        "confidence": 0.0,
        "method_detail": "RefImg: nicht erkannt",
    }

    try:
        codes, ref_matrix = _load_reference_images()
        if len(codes) == 0:
            return blocked_result

        # DMX-Region aus dem Kamerabild extrahieren (alle Rotationsvarianten)
        variants = _extract_dmx_region(frame)
        if not variants:
            logger.debug("Pipeline 3: Keine DMX-Region im Bild gefunden.")
            return blocked_result

        n_pixels = _REF_IMG_SIZE[0] * _REF_IMG_SIZE[1]

        # Referenz-Matrix einmalig packen (gecacht nach erstem Aufruf)
        global _REF_IMG_PACKED
        if '_REF_IMG_PACKED' not in globals() or _REF_IMG_PACKED is None:
            _REF_IMG_PACKED = np.packbits(ref_matrix, axis=1)  # (26000, 1250)

        # Alle Varianten in eine Matrix stapeln und packen
        variant_matrix = np.array([v.flatten() for v in variants], dtype=np.uint8)  # (V, 10000)
        variant_packed = np.packbits(variant_matrix, axis=1)  # (V, 1250)

        # Popcount-Lookup-Table (256 Einträge)
        _popcount_lut = np.array([bin(i).count('1') for i in range(256)], dtype=np.int32)

        # Batch-XOR: Für jede Variante gegen alle Referenzen
        overall_best_score = -1.0
        overall_best_code = None
        overall_second_score = 0.0
        overall_second_code = None

        for i in range(len(variant_packed)):
            # XOR: (N, 1250) — eine Variante gegen alle Referenzen
            xor_packed = np.bitwise_xor(_REF_IMG_PACKED, variant_packed[i])  # (N, 1250)
            # Popcount per Byte via LUT, dann Summe → Hamming-Distanz
            hamming = np.sum(_popcount_lut[xor_packed], axis=1)  # (N,)
            scores = 1.0 - (hamming.astype(np.float32) / n_pixels)

            # Top-2 für Score + Margin
            top2_idx = np.argpartition(scores, -2)[-2:]
            top2_sorted = top2_idx[np.argsort(scores[top2_idx])[::-1]]
            best_idx = top2_sorted[0]
            second_idx = top2_sorted[1] if len(top2_sorted) > 1 else best_idx

            if float(scores[best_idx]) > overall_best_score:
                overall_best_score = float(scores[best_idx])
                overall_best_code = codes[best_idx]
                overall_second_score = float(scores[second_idx])
                overall_second_code = codes[second_idx]

            # Early-Termination: Bei Score >= 0.95 sofort aufhören
            if overall_best_score >= 0.95:
                break

        margin = overall_best_score - overall_second_score

        logger.info(
            f"Pipeline 3 RefImg: Bester='{overall_best_code}' Score={overall_best_score:.3f}, "
            f"Zweiter='{overall_second_code}' Score={overall_second_score:.3f}, "
            f"Margin={margin:.3f}"
        )

        # Mindest-Score und Mindest-Margin für Akzeptanz
        if overall_best_score >= 0.70 and margin >= 0.02:
            return {
                "status": "matched",
                "text": overall_best_code,
                "confidence": min(1.0, max(0.60, overall_best_score)),
                "method_detail": f"RefImg Hamming-Match (Score={overall_best_score:.3f}, Margin={margin:.3f})",
            }

        return blocked_result

    except Exception as e:
        logger.warning(f"Pipeline 3 RefImg Fehler: {e}")
        return blocked_result


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


def _compute_joint_bayes_confidence(ocr_code: str | None, ocr_conf: float,
                                    dmtx_code: str | None, dmtx_score: float) -> tuple[str | None, float, str]:
    """
    Kombiniert OCR- und DataMatrix-Ergebnisse nach dem Theorem von Bayes:
    P(c | OCR, DMTX) ~ P(OCR | c) * P(DMTX | c) * P(c)
    
    Returns:
        tuple[str | None, float, str]: (gewählter_code, konfidenz_score, methode_name)
    """
    if not ocr_code and not dmtx_code:
        return None, 0.0, "Fehler"

    if ocr_code and dmtx_code and ocr_code == dmtx_code:
        p_ocr = max(0.85, ocr_conf)
        p_dmtx = max(0.90, dmtx_score)
        bayes_conf = 1.0 - (1.0 - p_ocr) * (1.0 - p_dmtx)
        return ocr_code, min(1.0, max(0.98, bayes_conf)), "Verifiziert"

    if dmtx_code and not ocr_code:
        return dmtx_code, min(0.96, max(0.70, dmtx_score)), "Rekonstruiert"

    if ocr_code and not dmtx_code:
        return ocr_code, min(0.95, max(0.60, ocr_conf)), "OCR"

    l_ocr = ocr_conf * 0.90
    l_dmtx = dmtx_score * 0.85

    if l_dmtx >= l_ocr:
        logger.info(f"Bayes-Fusion: Wähle DMTX '{dmtx_code}' ({dmtx_score:.1%}) über OCR '{ocr_code}' ({ocr_conf:.1%})")
        return dmtx_code, float(l_dmtx), "Bayes-Fusion (DMTX)"
    else:
        logger.info(f"Bayes-Fusion: Wähle OCR '{ocr_code}' ({ocr_conf:.1%}) über DMTX '{dmtx_code}' ({dmtx_score:.1%})")
        return ocr_code, float(l_ocr), "Bayes-Fusion (OCR)"


def _merge_results(ocr_result: dict, dmx_result: dict, frame: np.ndarray,
                   ref_img_result: dict = None) -> dict:
    """
    Führt die Ergebnisse von OCR, DataMatrix und Referenzbild-Pipeline zusammen.
    
    Args:
        ocr_result (dict): Das OCR-Ergebnis.
        dmx_result (dict): Das DataMatrix-Ergebnis.
        frame (np.ndarray): Das Graustufenbild (für Rekonstruktion).
        ref_img_result (dict): Das Referenzbild-Pipeline-Ergebnis (Pipeline 3).
        
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

    # Pipeline 3: Referenzbild-Ergebnis extrahieren
    ref_text = None
    ref_conf = 0.0
    ref_status = "blocked"
    if ref_img_result:
        ref_text = ref_img_result.get("text")
        ref_conf = ref_img_result.get("confidence", 0.0)
        ref_status = ref_img_result.get("status", "blocked")

    logger.info(
        f"Merge: OCR={ocr_status}('{ocr_text or ocr_partial_display}') "
        f"+ DMX={dmx_status}('{dmx_text}') "
        f"+ RefImg={ref_status}('{ref_text}')"
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
                # RefImg als Tie-Breaker bei DMX ≠ OCR Konflikt
                if ref_text and ref_text == dmtx_norm:
                    logger.info(
                        f"[REFIMG-TIEBREAK] RefImg bestätigt DMX '{dmx_text}' gegen OCR '{ocr_check_text}'"
                    )
                    confidence = 0.98
                elif ref_text and ref_text == ocr_norm:
                    logger.info(
                        f"[REFIMG-TIEBREAK] RefImg bestätigt OCR '{ocr_check_text}' gegen DMX '{dmx_text}'"
                    )
                    # OCR + RefImg überstimmen DMX-Rekonstruktion (als unverifiziert markieren!)
                    return {
                        "success": True,
                        "result": ocr_check_text,
                        "method": "OCR+RefImg-Tiebreak",
                        "confidence": 0.85,
                        "dmtx_result": dmx_text,
                        "ocr_result": ocr_check_text,
                        "verified": False,
                        "ocr_partial_display": ocr_check_text,
                    }
                else:
                    logger.warning(
                        f"[WARN] ABWEICHUNG: DMX='{dmx_text}' vs OCR='{ocr_check_text}'. "
                        f"Nutze DMX ({dmx_result['method_detail']})."
                    )
                    confidence = 0.9

        # Dreifach-Verifikation: DMX + OCR + RefImg stimmen überein
        if is_verified and ref_text and ref_text == dmx_text:
            logger.info(f"[TRIPLE-MATCH] Alle 3 Pipelines bestätigen: '{dmx_text}'")
            confidence = 1.0

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

        # --- NEU: Erweiterte Cross-Validation für Partial-Codes ---
        # Generiere die Kandidaten aus dem Partial-Display (z.B. 'W03?' → W030..W039)
        partial_for_cv = ocr_partial_display or _format_partial_display(
            ocr_readable, ocr_result.get("missing_positions", [])
        )
        cv_candidates = _generate_10_candidates_from_partial(partial_for_cv)

        if cv_candidates:
            # Strategie A: Cross-Validation mit dem DMX-beobachteten Grid (binary)
            observed_grid = dmx_result.get("observed_grid")
            if observed_grid is not None and observed_grid.shape == (10, 10):
                cv_code, cv_score = _cross_validate_ocr_dmtx(partial_for_cv, observed_grid)
                if cv_code is not None:
                    logger.info(
                        f"[CROSS-VAL] Partial '{partial_for_cv}' → Cross-Validation Match '{cv_code}' "
                        f"(Score: {cv_score:.1%})"
                    )
                    return {
                        "success": True,
                        "result": cv_code,
                        "method": "Rekonstruiert",
                        "confidence": min(1.0, max(0.90, cv_score)),
                        "dmtx_result": cv_code,
                        "ocr_result": ocr_partial_display,
                        "verified": False,
                        "ocr_partial_display": ocr_partial_display,
                    }

            # Strategie B: Soft-Grid-Extraktion + Soft-Matching gegen die Kandidaten
            soft_grid = _extract_observed_grid_soft(frame)
            if soft_grid is not None:
                soft_cand, soft_score, soft_margin = _match_soft_grid_matrix(
                    soft_grid, cv_candidates
                )
                if soft_cand is not None and soft_score >= 0.55:
                    logger.info(
                        f"[SOFT-CV] Partial '{partial_for_cv}' → Soft-Grid Match '{soft_cand}' "
                        f"(Score: {soft_score:.1%}, Margin: {soft_margin:.1%})"
                    )
                    return {
                        "success": True,
                        "result": soft_cand,
                        "method": "Rekonstruiert",
                        "confidence": min(1.0, max(0.85, soft_score)),
                        "dmtx_result": soft_cand,
                        "ocr_result": ocr_partial_display,
                        "verified": False,
                        "ocr_partial_display": ocr_partial_display,
                    }

            # Strategie C: Alle 9 Binarisierungs-Varianten für Grid-Extraktion durchprobieren
            # und Cross-Validation gegen die Kandidaten
            for method in ["otsu", "adaptive", "clahe_otsu", "etch_denoise", "ridge_boost",
                           "sauvola_w11", "sauvola_w21", "niblack"]:
                grid_obs = _extract_observed_grid(frame, binarization_method=method, strict_l_finder=False)
                if grid_obs is not None:
                    cv_code2, cv_score2 = _cross_validate_ocr_dmtx(partial_for_cv, grid_obs)
                    if cv_code2 is not None:
                        logger.info(
                            f"[CROSS-VAL-{method}] Partial '{partial_for_cv}' → Match '{cv_code2}' "
                            f"(Score: {cv_score2:.1%})"
                        )
                        return {
                            "success": True,
                            "result": cv_code2,
                            "method": "Rekonstruiert",
                            "confidence": min(1.0, max(0.88, cv_score2)),
                            "dmtx_result": cv_code2,
                            "ocr_result": ocr_partial_display,
                            "verified": False,
                            "ocr_partial_display": ocr_partial_display,
                        }

            # Strategie D: Multi-Scale Template-Matching als letzter Versuch
            # Wenn kein Grid extrahierbar ist, vergleiche synthetische DMX-Bilder
            # der Kandidaten direkt mit dem Originalbild
            if cv_candidates and len(cv_candidates) <= 10:
                tmpl_cand, tmpl_score, tmpl_margin = _template_match_candidates(
                    frame, cv_candidates
                )
                if tmpl_cand is not None and tmpl_score >= 0.25 and tmpl_margin >= 0.01:
                    logger.info(
                        f"[TEMPLATE-MATCH] Partial '{partial_for_cv}' → Match '{tmpl_cand}' "
                        f"(Score: {tmpl_score:.3f}, Margin: {tmpl_margin:.3f})"
                    )
                    return {
                        "success": True,
                        "result": tmpl_cand,
                        "method": "Rekonstruiert",
                        "confidence": min(0.92, max(0.70, tmpl_score)),
                        "dmtx_result": tmpl_cand,
                        "ocr_result": ocr_partial_display,
                        "verified": False,
                        "ocr_partial_display": ocr_partial_display,
                    }

        # Partial-Fallback: Wenn Rekonstruktion und Cross-Validation fehlschlagen,
        # versuche den wahrscheinlichsten Code aus dem Partial zu erschließen
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

    # 4b. Fall: Pipeline 3 (RefImg) als letzte Rettung, wenn DMX + OCR komplett fehlgeschlagen
    if ref_text and ref_status == "matched" and ref_conf >= 0.75:
        logger.info(
            f"[REFIMG-RESCUE] DMX + OCR fehlgeschlagen, RefImg liefert '{ref_text}' "
            f"(Conf={ref_conf:.2f})"
        )
        return {
            "success": True,
            "result": ref_text,
            "method": "RefImg",
            "confidence": min(0.90, ref_conf),
            "dmtx_result": None,
            "ocr_result": None,
            "verified": False,
            "ocr_partial_display": ref_text,
        }

    # 5. Fall: Keine Erkennung möglich
    logger.warning("Weder DataMatrix noch OCR noch RefImg konnten etwas lesen.")
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


def scan_datamatrix(frame: np.ndarray) -> dict:
    """
    Reine DataMatrix-Auswertung auf einem bereits zugeschnittenen DataMatrix-Crop.
    Führt nur DataMatrix-Pipelines aus (zxing, pylibdmtx, Rekonstruktion, RefImg).
    Kein OCR — spart ~200ms pro Scan.
    
    Args:
        frame (np.ndarray): Der zugeschnittene DataMatrix-Bereich (deskew'd).
        
    Returns:
        dict: Ergebnis mit keys: status, text, confidence, method_detail, observed_grid.
    """
    if frame is None or frame.size == 0:
        return {
            "status": "blocked", "text": None,
            "confidence": 0.0, "method_detail": "Kein Bild",
            "observed_grid": None,
        }

    t0 = time.time()

    # 1. Fast-Path: zxing-cpp direkt auf dem Crop
    dmx_result = _scan_datamatrix_pipeline(frame)
    if dmx_result and dmx_result.get("status") == "decoded" and dmx_result.get("text"):
        code = dmx_result["text"]
        if _is_valid_horden_code(code):
            logger.info(f"[2CLASS-DMX] DataMatrix direkt erkannt: '{code}' ({int((time.time()-t0)*1000)}ms)")
            dmx_result["confidence"] = 1.0
            return dmx_result

    # 2. Referenzbild-Pipeline (Pipeline 3)
    ref_result = _scan_reference_image_pipeline(frame)
    if ref_result and ref_result.get("status") == "matched" and ref_result.get("text"):
        ref_text = ref_result["text"]
        ref_conf = ref_result.get("confidence", 0.0)
        if ref_conf >= 0.75 and _is_valid_horden_code(ref_text):
            logger.info(f"[2CLASS-DMX] RefImg Match: '{ref_text}' (Conf={ref_conf:.2f}, {int((time.time()-t0)*1000)}ms)")
            # Wenn DMX auch ein Ergebnis hat, bevorzuge DMX
            if dmx_result and dmx_result.get("text"):
                dmx_result["_ref_img_text"] = ref_text
                dmx_result["_ref_img_conf"] = ref_conf
                return dmx_result
            return {
                "status": "matched_refimg", "text": ref_text,
                "confidence": ref_conf, "method_detail": "RefImg",
                "observed_grid": None,
            }

    # 3. Rekonstruktions-Versuch (ohne OCR-Hinweis)
    # ACHTUNG: Auf Crops ist blinde 4000-Code-Rekonstruktion unzuverlässig.
    # Konfidenz-Cap auf 0.85 setzen und strengere Schwellen anwenden.
    recon_result = _try_reconstruct(frame, None, 0.0, None)
    if recon_result is not None and recon_result.get("success"):
        recon_conf = min(0.85, recon_result.get("confidence", 0.85))
        logger.info(f"[2CLASS-DMX] Rekonstruktion: '{recon_result['result']}' (Conf={recon_conf:.2f}, {int((time.time()-t0)*1000)}ms)")
        return {
            "status": "reconstructed", "text": recon_result["result"],
            "confidence": recon_conf,
            "method_detail": recon_result.get("method", "Rekonstruiert"),
            "observed_grid": None,
        }

    # 4. Kein Ergebnis
    logger.info(f"[2CLASS-DMX] Keine DataMatrix erkannt ({int((time.time()-t0)*1000)}ms)")
    return dmx_result if dmx_result else {
        "status": "blocked", "text": None,
        "confidence": 0.0, "method_detail": "Kein DMX erkannt",
        "observed_grid": None,
    }


def scan_ocr(frame: np.ndarray) -> dict:
    """
    Reine OCR-Auswertung auf einem bereits zugeschnittenen Text-Crop.
    Führt nur OCR-Pipelines aus (EasyOCR, PACC Char-Classifier).
    Keine DataMatrix-Verarbeitung.
    
    Args:
        frame (np.ndarray): Der zugeschnittene Text-Bereich (deskew'd).
        
    Returns:
        dict: Ergebnis mit keys: status, text, confidence, partial_display,
              readable_chars, missing_positions, raw_candidate.
    """
    if frame is None or frame.size == 0:
        return {
            "status": "failed", "text": None, "confidence": 0.0,
            "partial_display": None, "readable_chars": None,
            "missing_positions": [], "raw_candidate": None,
        }

    t0 = time.time()

    # 1. EasyOCR-basierte Erkennung mit Status
    ocr_result = _read_ocr_with_status(frame)
    ocr_text = ocr_result.get("text")
    ocr_conf = ocr_result.get("confidence", 0.0)

    # 2. PACC Char-Classifier als Ergänzung/Verifikation
    pacc_text, pacc_conf = _predict_pacc(frame)
    if pacc_text and pacc_conf > 0.0:
        logger.info(f"[2CLASS-OCR] PACC: '{pacc_text}' (Conf={pacc_conf:.2f})")
        # PACC als Fallback wenn EasyOCR nichts findet
        if not ocr_text and pacc_conf >= 0.70:
            ocr_result["text"] = pacc_text
            ocr_result["confidence"] = pacc_conf
            ocr_result["status"] = "ok"
            ocr_result["partial_display"] = pacc_text
        # PACC als Verifikation wenn beide übereinstimmen
        elif ocr_text and pacc_text == ocr_text:
            ocr_result["confidence"] = max(ocr_conf, pacc_conf)

    logger.info(
        f"[2CLASS-OCR] Ergebnis: status={ocr_result.get('status')}, "
        f"text='{ocr_result.get('text')}', conf={ocr_result.get('confidence', 0.0):.2f} "
        f"({int((time.time()-t0)*1000)}ms)"
    )
    return ocr_result


def _are_boxes_adjacent(box1, box2, max_ratio: float = 2.5) -> bool:
    """
    Prüft ob zwei Bounding Boxes (z.B. DataMatrix und Text) räumlich nah beieinander / unmittelbar aneinander liegen.
    box1, box2: (x1, y1, x2, y2)
    max_ratio: Die maximale Lücke zwischen den Boxen relativ zur Referenz-Größe.
    """
    if not box1 or not box2:
        return False
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    gap_x = max(0, max(x1_1, x1_2) - min(x2_1, x2_2))
    gap_y = max(0, max(y1_1, y1_2) - min(y2_1, y2_2))

    w1 = max(1, x2_1 - x1_1)
    h1 = max(1, y2_1 - y1_1)
    ref_size = max(w1, h1)

    return (gap_x / ref_size <= max_ratio) and (gap_y / ref_size <= max_ratio)


def scan_2class(frame: np.ndarray, detections: list[dict], cancellation_check=None) -> dict:
    """
    Erweiterte 2-Klassen Pipeline für DataMatrix + Text-Detektion.
    Empfängt das Rohbild und eine Liste von YOLO-Detections, sortiert nach Klasse,
    schneidet die jeweiligen Bereiche per deskew_crop() aus und wertet
    DataMatrix und OCR getrennt und parallel aus.
    """
    if cancellation_check and cancellation_check():
        logger.info("[2CLASS] Scan wurde vor Start abgebrochen/verworfen (neuer Trigger).")
        return {"success": False, "result": "ABORTED", "method": "Abgebrochen", "confidence": 0.0, "cancelled": True}

    if frame is None:
        return {
            "success": False, "result": "Kein Bild vorhanden.",
            "method": "Fehler", "confidence": 0.0,
            "dmtx_result": None, "ocr_result": None, "verified": False,
            "ocr_partial_display": None,
        }

    # Detections nach Klasse sortieren (je die mit höchster Konfidenz)
    MIN_DMX_YOLO_CONF = 0.30
    dmx_det = None
    txt_det = None
    for det in detections:
        cls = det.get("cls", -1)
        if cls == 0:  # datamatrix
            if det["conf"] >= MIN_DMX_YOLO_CONF:
                if dmx_det is None or det["conf"] > dmx_det["conf"]:
                    dmx_det = det
            else:
                logger.info(f"[2CLASS] DMX-Detection verworfen: conf={det['conf']:.2f} < {MIN_DMX_YOLO_CONF}")
        elif cls == 1:  # text
            if det["conf"] >= 0.25:
                if txt_det is None or det["conf"] > txt_det["conf"]:
                    txt_det = det

    # Smart Crop Derivation: Falls nur eine der beiden Klassen von YOLO erkannt wurde,
    # leite die Nachbar-Box für die zweite Klasse aus der ersten ab (Etikett-Layout).
    if dmx_det is not None and txt_det is None:
        # Text liegt direkt neben oder unter der DataMatrix -> Erweitere DMX-Box für Text-Crop
        x1_d, y1_d, x2_d, y2_d = dmx_det["box"]
        w_d = x2_d - x1_d
        h_d = y2_d - y1_d
        # Erweiterte Box für Text-Crop
        derived_txt_box = (
            max(0, x1_d - int(w_d * 1.5)),
            max(0, y1_d - int(h_d * 1.5)),
            x2_d + int(w_d * 2.5),
            y2_d + int(h_d * 2.5)
        )
        txt_det = {"cls": 1, "box": derived_txt_box, "conf": dmx_det["conf"], "derived": True}
        logger.info(f"[2CLASS] Text-Box aus DataMatrix-Box abgeleitet: {derived_txt_box}")

    elif txt_det is not None and dmx_det is None:
        # DataMatrix liegt direkt neben dem Text -> Erweitere Text-Box für DMX-Crop
        x1_t, y1_t, x2_t, y2_t = txt_det["box"]
        w_t = x2_t - x1_t
        h_t = y2_t - y1_t
        derived_dmx_box = (
            max(0, x1_t - int(w_t * 1.5)),
            max(0, y1_t - int(h_t * 1.5)),
            x2_t + int(w_t * 2.5),
            y2_t + int(h_t * 2.5)
        )
        dmx_det = {"cls": 0, "box": derived_dmx_box, "conf": txt_det["conf"], "derived": True}
        logger.info(f"[2CLASS] DataMatrix-Box aus Text-Box abgeleitet: {derived_dmx_box}")

    dmx_info = f"DMX=Ja(conf={dmx_det['conf']:.2f})" if dmx_det else "DMX=Nein"
    txt_info = f"TXT=Ja(conf={txt_det['conf']:.2f})" if txt_det else "TXT=Nein"
    logger.info(f"[2CLASS] Detections: {dmx_info}, {txt_info}")

    # Falls weder DataMatrix noch Text erkannt wurde, Fallback auf Vollbild scan()
    if dmx_det is None and txt_det is None:
        logger.info("[2CLASS] Keine YOLO-Detections. Starte Fallback auf scan().")
        return scan(frame)


    # --- Crops erzeugen ---
    dmx_crop = None
    txt_crop = None

    if dmx_det:
        dmx_crop = deskew_crop(frame, dmx_det["box"], padding=40)
        logger.info(f"[2CLASS] DataMatrix-Crop: {dmx_crop.shape[1]}x{dmx_crop.shape[0]}")

    if txt_det:
        txt_crop = deskew_crop(frame, txt_det["box"], padding=30)
        logger.info(f"[2CLASS] Text-Crop: {txt_crop.shape[1]}x{txt_crop.shape[0]}")

    # --- Parallele Auswertung ---
    dmx_result = None
    ocr_result = None

    t_start = time.time()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}

        if dmx_crop is not None:
            futures["dmx"] = executor.submit(scan_datamatrix, dmx_crop)

        if txt_crop is not None:
            futures["ocr"] = executor.submit(scan_ocr, txt_crop)

        if "dmx" in futures:
            try:
                dmx_result = futures["dmx"].result()
            except Exception as e:
                logger.warning(f"[2CLASS] DataMatrix-Scan Fehler: {e}")
                dmx_result = {
                    "status": "blocked", "text": None,
                    "confidence": 0.0, "method_detail": "DMX Fehler",
                    "observed_grid": None,
                }

        if "ocr" in futures:
            try:
                ocr_result = futures["ocr"].result()
            except Exception as e:
                logger.warning(f"[2CLASS] OCR-Scan Fehler: {e}")
                ocr_result = {
                    "status": "failed", "text": None, "confidence": 0.0,
                    "partial_display": None, "readable_chars": None,
                    "missing_positions": [], "raw_candidate": None,
                }

    if cancellation_check and cancellation_check():
        logger.info("[2CLASS] Scan während der Auswertung durch neuen Trigger storniert!")
        return {"success": False, "result": "ABORTED", "method": "Abgebrochen", "confidence": 0.0, "cancelled": True}

    t_total = int((time.time() - t_start) * 1000)

    # --- Fallback-Ergebnisse für fehlende Pipelines ---
    if dmx_result is None:
        dmx_result = {
            "status": "blocked", "text": None,
            "confidence": 0.0, "method_detail": "Nicht erkannt",
            "observed_grid": None,
        }

    if ocr_result is None:
        ocr_result = {
            "status": "failed", "text": None, "confidence": 0.0,
            "partial_display": None, "readable_chars": None,
            "missing_positions": [], "raw_candidate": None,
        }

    # --- Ergebnisse mergen (bestehende Triple-Fusion-Logik) ---
    # Für die merge-Funktion brauchen wir das Frame für eventuelle Rekonstruktionen.
    # Wir nutzen den DMX-Crop, wenn vorhanden, sonst das Gesamtbild.
    merge_frame = dmx_crop if dmx_crop is not None else frame

    result = _merge_results(ocr_result, dmx_result, merge_frame, ref_img_result=None)

    result["_internal_timing"] = {"total_2class_ms": t_total}
    result["_2class_mode"] = True
    result["_detections"] = {
        "dmx_box": dmx_det["box"] if dmx_det else None,
        "dmx_conf": dmx_det["conf"] if dmx_det else 0.0,
        "txt_box": txt_det["box"] if txt_det else None,
        "txt_conf": txt_det["conf"] if txt_det else 0.0,
    }

    logger.info(
        f"[2CLASS] Ergebnis: success={result['success']}, "
        f"method={result['method']}, result='{result['result']}' ({t_total}ms)"
    )

    # --- Cross-Validation & Smart Fallback ---
    # Priorität: KEINE Fehllesungen. Lieber "Fehler" als falscher Code.
    if not result.get("success"):
        # Fall A: 2class hat nichts gefunden → Fallback auf scan()
        logger.info("[2CLASS] 2-Klassen-Crop ohne Erfolg. Starte Fallback auf scan().")
        fallback_res = scan(frame)
        if fallback_res.get("success"):
            fb_verified = fallback_res.get("verified", False)
            fb_conf = fallback_res.get("confidence", 0)
            fb_method = fallback_res.get("method", "")
            # Nur verifizierte Ergebnisse oder OCR mit ausreichend hoher Konfidenz akzeptieren.
            # Gamma-Fallback-OCR mit niedriger Konfidenz (z.B. W631 statt W031) wird abgelehnt.
            # Schwelle 0.68: W031 (0.70) passiert, W631 (0.63) nicht.
            if fb_verified or fb_conf >= 0.68 or fb_method == "Verifiziert":
                return fallback_res
            else:
                logger.warning(
                    f"[2CLASS] scan()-Fallback '{fallback_res.get('result')}' hat niedrige Konfidenz "
                    f"({fb_conf:.2f}, method={fb_method}). Nicht akzeptiert.")
        return result

    # Fall B: 2class hat ein Ergebnis, aber ist es verlässlich?
    is_verified = result.get("verified", False) and result.get("confidence", 0) >= 1.0

    if is_verified:
        # Doppelt verifiziert (DMX + OCR stimmen überein) → direkt akzeptieren
        logger.info(f"[2CLASS] Ergebnis doppelt verifiziert. Akzeptiere '{result['result']}'.")
        return result

    # Nicht verifiziert → Cross-Validation mit scan() auf dem Gesamtbild
    logger.info(
        f"[2CLASS-CROSSVAL] Ergebnis '{result['result']}' nicht verifiziert "
        f"(method={result['method']}, conf={result.get('confidence', 0):.2f}). "
        f"Starte Gegenprobe mit scan() auf Gesamtbild..."
    )
    crossval_res = scan(frame)

    if not crossval_res.get("success"):
        # scan() hat auch nichts gefunden → 2class-Ergebnis NUR akzeptieren wenn es KEINE blinde Rekonstruktion war
        if result.get("confidence", 0) >= 0.95 and result.get("method") != "Rekonstruiert":
            logger.info(
                f"[2CLASS-CROSSVAL] scan() fehlgeschlagen, aber 2class hat hohe Konfidenz "
                f"({result.get('confidence', 0):.2f}). Akzeptiere '{result['result']}'.")
            return result
        else:
            logger.warning(
                f"[2CLASS-CROSSVAL] scan() fehlgeschlagen und 2class-Ergebnis unverifiziert/Rekonstruktion "
                f"({result.get('method')}, conf={result.get('confidence', 0):.2f}). Melde Fehler.")
            return {
                "success": False, "result": "Unsicheres Ergebnis.",
                "method": "Fehler", "confidence": 0.0,
                "dmtx_result": result.get("dmtx_result"),
                "ocr_result": result.get("ocr_result"),
                "verified": False, "ocr_partial_display": result.get("ocr_partial_display"),
            }

    # Beide haben ein Ergebnis
    code_2class = result.get("result")
    code_fullframe = crossval_res.get("result")

    if code_2class == code_fullframe:
        # Beide stimmen überein → hohe Sicherheit
        best_conf = max(result.get("confidence", 0), crossval_res.get("confidence", 0))
        logger.info(
            f"[2CLASS-CROSSVAL] Übereinstimmung! Beide Pipelines: '{code_2class}' "
            f"(Konfidenz={best_conf:.2f})")
        crossval_res["confidence"] = best_conf
        return crossval_res

    # Widerspruch: 2class ≠ scan()
    crossval_verified = crossval_res.get("verified", False) or (
        crossval_res.get("method") == "Verifiziert" and crossval_res.get("confidence", 0) >= 0.98
    )

    if crossval_verified:
        # scan() ist verifiziert → bevorzuge scan()
        logger.info(
            f"[2CLASS-CROSSVAL] Widerspruch: 2class='{code_2class}' vs scan()='{code_fullframe}'. "
            f"scan() ist verifiziert → bevorzuge '{code_fullframe}'.")
        return crossval_res

    # Digit-Confusion-Erkennung: Wenn die Codes nur in einer Ziffer abweichen
    # und diese Ziffer ein bekanntes OCR-Konfusionspaar ist (0↔4, 0↔6, 0↔8, 3↔8),
    # dann behandle als "Soft-Match" und bevorzuge den mit höherer Konfidenz.
    if code_2class and code_fullframe and len(code_2class) == 4 and len(code_fullframe) == 4:
        _CONFUSION_PAIRS = {('0','4'),('4','0'),('0','6'),('6','0'),('0','8'),('8','0'),('3','8'),('8','3'),('0','9'),('9','0')}
        diff_positions = []
        for i in range(4):
            if code_2class[i] != code_fullframe[i]:
                diff_positions.append(i)
        
        if len(diff_positions) == 1:
            pos = diff_positions[0]
            c1, c2 = code_2class[pos], code_fullframe[pos]
            if (c1, c2) in _CONFUSION_PAIRS:
                # Soft-Match: Wähle den Code mit höherer Konfidenz
                conf_2class = result.get("confidence", 0)
                conf_scan = crossval_res.get("confidence", 0)
                if conf_scan >= conf_2class:
                    chosen = crossval_res
                    chosen_code = code_fullframe
                else:
                    chosen = result
                    chosen_code = code_2class
                logger.info(
                    f"[2CLASS-CROSSVAL] Digit-Confusion Soft-Match: '{code_2class}' vs '{code_fullframe}' "
                    f"(Position {pos}: '{c1}'↔'{c2}'). Bevorzuge '{chosen_code}' "
                    f"(Conf 2class={conf_2class:.2f}, scan={conf_scan:.2f}).")
                return chosen

    # Keiner verifiziert + Widerspruch → FEHLER melden (keine Fehllesung!)
    logger.warning(
        f"[2CLASS-CROSSVAL] Widerspruch ohne Verifikation: "
        f"2class='{code_2class}' vs scan()='{code_fullframe}'. Melde Fehler.")
    return {
        "success": False,
        "result": f"Widerspruch: {code_2class} vs {code_fullframe}",
        "method": "Fehler",
        "confidence": 0.0,
        "dmtx_result": result.get("dmtx_result"),
        "ocr_result": crossval_res.get("ocr_result"),
        "verified": False,
        "ocr_partial_display": crossval_res.get("ocr_partial_display"),
    }


def scan(frame: np.ndarray) -> dict:
    """
    Haupt-Scan-Funktion mit Triple-Validation (v5.0).
    Führt OCR-, DataMatrix- und Referenzbild-Erkennung parallel in Threads aus
    und kombiniert die Ergebnisse per Dreifach-Fusion.
    
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
    logger.info(f"Triple-Validation Scan v5.0 gestartet auf Bild mit {w}x{h} Pixeln.")

    try:
        fast_dmx = _scan_datamatrix_pipeline(frame)
    except Exception as e:
        logger.warning(f"DataMatrix Pipeline Exception: {e}")
        fast_dmx = {"status": "blocked", "text": None, "confidence": 0.0}

    if fast_dmx and fast_dmx.get("status") == "decoded" and fast_dmx.get("text") and _is_valid_horden_code(fast_dmx["text"]):
        code = fast_dmx["text"]
        logger.info(f"[FAST-PATH] DataMatrix direkt erkannt '{code}'. Skippe OCR (< 15ms).")
        return {
            "success": True,
            "result": code,
            "method": "Verifiziert",
            "confidence": 1.0,
            "dmtx_result": code,
            "ocr_result": code,
            "verified": True,
            "ocr_partial_display": code,
            "_internal_timing": {"ocr_ms": 0, "dmtx_ms": 5, "refimg_ms": 0},
        }

    # ===== FAST-PATH 2: Horden-DB Referenzbild-Matching (< 10ms) =====
    try:
        import horde_db
        horde_match = horde_db.match_horde_image(frame, min_confidence=0.78)
        if horde_match and horde_match.get("success") and horde_match.get("result"):
            code = horde_match["result"]
            conf = horde_match.get("confidence", 0.85)
            logger.info(f"[FAST-PATH HORDEN-DB] Hordenbild-Match direkt erkannt: '{code}' (Conf: {conf:.2%}). Skippe OCR/Gamma (< 10ms).")
            return {
                "success": True,
                "result": code,
                "method": "HordenDB-Match",
                "confidence": conf,
                "dmtx_result": None,
                "ocr_result": code,
                "verified": False,
                "ocr_partial_display": code,
                "_internal_timing": {"ocr_ms": 0, "dmtx_ms": 5, "refimg_ms": 5},
            }
    except Exception as e:
        logger.warning(f"HordeDB FastPath Fehler: {e}")

    ocr_result = None
    ref_img_result = None
    dmx_result = fast_dmx

    # --- Timing: OCR, DMTX und RefImg separat messen ---
    _t_ocr_start = time.time()
    _t_dmx_start = time.time()
    _t_refimg_start = time.time()
    _t_ocr_end = _t_ocr_start
    _t_dmx_end = _t_dmx_start
    _t_refimg_end = _t_refimg_start

    # OCR und RefImg sequentiell ausführen (verhindert PyTorch LibTorch / OpenCV Multithreading C-Crash)
    _t_ocr_start = time.time()
    try:
        ocr_result = _read_ocr_with_status(frame)
    except Exception as e:
        logger.warning(f"OCR Fehler: {e}")
        ocr_result = {
            "status": "failed", "text": None, "partial_display": None,
            "readable_chars": None, "confidence": 0.0,
            "readable_count": 0, "missing_positions": [],
        }
    _t_ocr_end = time.time()

    _t_refimg_start = time.time()
    try:
        ref_img_result = _scan_reference_image_pipeline(frame)
    except Exception as e:
        logger.warning(f"RefImg Pipeline Fehler: {e}")
        ref_img_result = {
            "status": "blocked", "text": None,
            "confidence": 0.0, "method_detail": "RefImg Fehler",
        }
    _t_refimg_end = time.time()

    if dmx_result is None:
        dmx_result = {
            "status": "blocked", "text": None,
            "method_detail": "Kein DMX", "confidence": 0.0,
            "observed_grid": None,
        }

    # Ergebnisse mergen (Triple-Fusion)
    result = _merge_results(ocr_result, dmx_result, frame, ref_img_result)

    # Internes Timing für den ScanLogger bereitstellen (nicht-brechend)
    result["_internal_timing"] = {
        "ocr_ms": int((_t_ocr_end - _t_ocr_start) * 1000),
        "dmtx_ms": int((_t_dmx_end - _t_dmx_start) * 1000),
        "refimg_ms": int((_t_refimg_end - _t_refimg_start) * 1000),
    }

    # --- Gamma-Korrektur-Fallback für dunkle/unterbelichtete Bilder ---
    # Wenn der normale Scan fehlschlägt, versuche mit aufgehelltem Bild.
    # Multi-Pass-Ansatz: Sammle OCR-Ergebnisse aus verschiedenen Gamma-Werten,
    # dann nutze Voting/Cross-Validation für das beste Ergebnis.
    # Problem: OCR verwechselt systematisch 0↔6 bei dunklen Bildern (W031→W631).
    if not result.get("success"):
        logger.info("[GAMMA-FALLBACK] Normaler Scan fehlgeschlagen. Versuche Gamma-Korrektur...")
        gray_fb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()
        
        # Sammle ALLE OCR-Ergebnisse über verschiedene Aufhellungen
        gamma_readings = []  # [(code, conf, attempt_name)]
        
        for gamma_val in [0.25, 0.3, 0.35, 0.4, 0.5]:
            lut = np.array([((i / 255.0) ** gamma_val) * 255 for i in range(256)]).astype("uint8")
            brightened = cv2.LUT(gray_fb, lut)
            
            ocr_gamma = _read_ocr_with_status(brightened)
            if ocr_gamma.get("status") == "ok" and ocr_gamma.get("text"):
                code = _normalize_ocr_confusions(ocr_gamma["text"])
                conf = ocr_gamma.get("confidence", 0.0)
                if _is_valid_horden_code(code):
                    gamma_readings.append((code, conf, f"gamma_{gamma_val}"))
                    logger.info(f"[GAMMA-FALLBACK] gamma_{gamma_val}: '{code}' (Conf={conf:.2f})")
        
        
        if gamma_readings:
            # Voting: Normalisiere 0↔6 Verwechslungen und zähle
            # Für jeden Code, erzeuge auch die 0↔6-Variante
            from collections import Counter
            
            # Zähle exakte Code-Vorkommen
            code_votes = Counter()
            code_max_conf = {}
            for code, conf, name in gamma_readings:
                code_votes[code] += 1
                if code not in code_max_conf or conf > code_max_conf[code]:
                    code_max_conf[code] = conf
            
            # Wenn ein Code mit 6 und derselbe mit 0 gefunden wurden, bevorzuge den mit 0
            # (weil 0→6 der häufigere OCR-Fehler ist)
            best_code = None
            best_score = 0  # (votes * 10 + conf)
            
            for code in code_votes:
                votes = code_votes[code]
                conf = code_max_conf[code]
                score = votes * 10 + conf
                
                # Bonus: Wenn eine 0↔6 Variante auch existiert, bevorzuge die mit 0
                for i in range(1, 4):
                    if code[i] == '6':
                        alt = list(code)
                        alt[i] = '0'
                        alt_code = ''.join(alt)
                        if alt_code in code_votes:
                            # Beide Varianten existieren → bevorzuge die mit 0
                            if code[i] == '6':
                                score -= 5  # Malus für 6-Variante
                
                if score > best_score:
                    best_score = score
                    best_code = code
            
            if best_code:
                best_conf = code_max_conf[best_code]
                best_votes = code_votes[best_code]
                logger.info(
                    f"[GAMMA-FALLBACK] Voting: '{best_code}' "
                    f"(Votes={best_votes}, MaxConf={best_conf:.2f}, Score={best_score:.1f})")
                result = {
                    "success": True,
                    "result": best_code,
                    "method": "OCR",
                    "confidence": best_conf,
                    "dmtx_result": None,
                    "ocr_result": best_code,
                    "verified": False,  # Gamma-Fallback ist reines OCR (keine DataMatrix-Verifikation)
                    "ocr_partial_display": best_code,
                    "_internal_timing": result.get("_internal_timing", {}),
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
