import os
import re
import time
import logging
import threading
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# --- Standard-Verzeichnis für den Horden-Zusatzordner ---
DEFAULT_HORDE_DB_DIR = "hard_scans_cache"
_HORDEN_PATTERN = re.compile(r"^[ABPW][0-9]{3}$", re.IGNORECASE)

_db_lock = threading.Lock()
_horde_cache = {}  # Map: code -> {"path": str, "gray": np.ndarray, "orb_des": np.ndarray, "mtime": float}
_cache_initialized = False


def _is_valid_horden_code(code: str) -> bool:
    """Prüft ob der Code ein gültiger Horden-Code ist (z. B. W003, A012, B100, P005)."""
    if not code:
        return False
    return bool(_HORDEN_PATTERN.match(code.strip()))


def get_horde_db_dir() -> str:
    """Liefert den absoluten/relativen Pfad des Zusatzordners und stellt sicher, dass er existiert."""
    db_dir = os.path.abspath(DEFAULT_HORDE_DB_DIR)
    os.makedirs(db_dir, exist_ok=True)
    return db_dir


def _describe(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Normierte 320x320-Graustufenvorlage und ORB-Deskriptoren für den Bildabgleich."""
    resized = cv2.resize(gray, (320, 320), interpolation=cv2.INTER_AREA)
    _, des = cv2.ORB_create(nfeatures=400).detectAndCompute(resized, None)
    return resized, des


def _cache_entry(path: str, gray: np.ndarray, mtime: float) -> dict:
    resized, des = _describe(gray)
    return {"path": path, "gray": resized, "orb_des": des, "mtime": mtime}


def load_horde_db(force_reload: bool = False):
    """
    Lädt alle gespeicherten Hordenbilder aus hard_scans_cache/ in den In-Memory-Cache.
    Berechnet ORB-Deskriptoren vorab für blitzschnellen visuellen Bildabgleich.
    """
    global _cache_initialized, _horde_cache
    with _db_lock:
        if _cache_initialized and not force_reload:
            return

        db_dir = get_horde_db_dir()
        t0 = time.time()
        new_cache = {}

        valid_extensions = (".jpg", ".jpeg", ".png")
        for fname in os.listdir(db_dir):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in valid_extensions:
                continue

            code = os.path.splitext(fname)[0].upper()
            if not _is_valid_horden_code(code):
                continue

            fpath = os.path.join(db_dir, fname)
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            new_cache[code] = _cache_entry(fpath, img, os.path.getmtime(fpath))

        _horde_cache = new_cache
        _cache_initialized = True
        logger.info(f"[HORDEN-DB] {len(_horde_cache)} Hordenbilder aus '{db_dir}' geladen ({int((time.time()-t0)*1000)}ms).")


def save_or_update_horde_image(
    code: str,
    frame: np.ndarray,
    is_late_scan: bool = False,
    verified: bool = False,
    confidence: float = 1.0
) -> str | None:
    """
    Speichert oder aktualisiert das Hordenbild im Zusatzordner hard_scans_cache/<CODE>.jpg.
    Garantiert, dass immer das aktuellste Bild der Horde in der Datenbank vorhanden ist.
    
    Args:
        code (str): Der erkannte Horden-Code (z.B. "W003").
        frame (np.ndarray): Das Kamerabild (BGR oder Graustufen).
        is_late_scan (bool): True wenn das Bild aus einer Späterkennung (>6s Timeout) stammt.
        verified (bool): True wenn der Code durch DataMatrix + OCR verifiziert wurde.
        confidence (float): Konfidenzwert der Erkennung.
        
    Returns:
        str | None: Pfad zur gespeicherten Datei oder None bei Fehler.
    """
    if not code or frame is None or frame.size == 0:
        return None

    code_clean = code.strip().upper()
    if not _is_valid_horden_code(code_clean):
        logger.warning(f"[HORDEN-DB] Verwerfe ungültigen Horden-Code '{code}' beim Speichern.")
        return None

    # --- ABSOLUTER SCHUTZ GEGEN DATENBANK-VERGIFTUNG ---
    # Nur 100% verifizierte DataMatrix-Scans (mit Reed-Solomon Fehlerkorrektur) oder exakte 1:1 DMX+OCR-Matches
    # dürfen in hard_scans_cache/ gespeichert werden! Reine OCR-Ergebnisse (selbst mit hoher Konfidenz) werden NIEMALS gespeichert.
    if not verified or confidence < 0.98:
        logger.warning(
            f"[HORDEN-DB] SPEICHERN ABGELEHNT: Hordenbild für '{code_clean}' ist nicht 100% verifiziert "
            f"(Verified={verified}, Conf={confidence:.2f} < 0.98). Speicherung verhindert Datenbankvergiftung!"
        )
        return None

    db_dir = get_horde_db_dir()
    filepath = os.path.join(db_dir, f"{code_clean}.jpg")

    try:
        # Bild-Orientierung normalisieren (Aufrecht ausrichten, falls durch DMX 90° gedreht)
        h, w = frame.shape[:2]
        if h > w * 1.6:  # Vertikal gedrehtes Bild aufrecht drehen
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        with _db_lock:
            # Bild auf Festplatte speichern / überschreiben (Aktualisierung)
            is_update = os.path.exists(filepath)
            success = cv2.imwrite(filepath, frame)
            if not success:
                logger.error(f"[HORDEN-DB] Fehler beim Schreiben der Datei '{filepath}'!")
                return None

            mtime = os.path.getmtime(filepath)

            # In-Memory-Cache direkt aktualisieren
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()
            _horde_cache[code_clean] = _cache_entry(filepath, gray, mtime)

            action = "aktualisiert" if is_update else "neu angelegt"
            tag = " [SPÄTERKENNUNG >6s]" if is_late_scan else ""
            logger.info(f"[HORDEN-DB] Hordenbild für '{code_clean}' {action}{tag}: '{filepath}'")
            return filepath

    except Exception as e:
        logger.error(f"[HORDEN-DB] Ausnahme beim Speichern von '{code_clean}': {e}")
        return None


def match_horde_image(frame: np.ndarray, min_confidence: float = 0.70) -> dict | None:
    """
    Vergleicht ein aufgenommenes Kamerabild gegen alle in hard_scans_cache/ gespeicherten Horden-Bilder.
    Kombiniert ORB-Feature-Matching (Lowe's Ratio Test) mit direkter Graustufen-Korrelation.
    
    Args:
        frame (np.ndarray): Das aktuelle Kamerabild.
        min_confidence (float): Mindest-Konfidenz für eine erfolgreiche Erkennung (0.0 - 1.0).
        
    Returns:
        dict | None: Scan-Ergebnis-Dict bei Treffer oder None.
    """
    if frame is None or frame.size == 0:
        return None

    if not _cache_initialized:
        load_horde_db()

    with _db_lock:
        if not _horde_cache:
            return None
        cache_copy = dict(_horde_cache)

    t0 = time.time()

    # Pre-Processing des Eingabebildes
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()
    resized, des_input = _describe(gray)

    if des_input is None or len(des_input) < 10:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    best_code = None
    best_score = 0.0
    second_score = 0.0
    best_hist_score = 0.0

    # Histogram des Eingabebilds für Bhattacharyya-Distanz
    hist_input = cv2.calcHist([resized], [0], None, [64], [0, 256])
    cv2.normalize(hist_input, hist_input)

    for code, data in cache_copy.items():
        db_des = data.get("orb_des")
        if db_des is None or len(db_des) < 10:
            continue

        # KNN-Matching für Robuste Feature-Zuordnung
        matches = bf.knnMatch(des_input, db_des, k=2)
        good_matches = 0
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good_matches += 1

        # Mindestens 15 gute Feature-Matches erforderlich
        if good_matches < 15:
            continue

        # Score berechnen (Verhältnis gute Matches zu Gesamt-Features + Vorlagen-Korrelation)
        feat_ratio = good_matches / max(min(len(des_input), len(db_des)), 1)

        # Graustufen-Strukturkorrelation (Template Match)
        res_match = cv2.matchTemplate(resized, data["gray"], cv2.TM_CCOEFF_NORMED)
        template_corr = float(res_match[0][0]) if res_match is not None else 0.0
        template_score = max(0.0, template_corr)

        # Histogram-Vergleich (Bhattacharyya-Distanz, niedrig = ähnlich)
        hist_db = cv2.calcHist([data["gray"]], [0], None, [64], [0, 256])
        cv2.normalize(hist_db, hist_db)
        hist_dist = cv2.compareHist(hist_input, hist_db, cv2.HISTCMP_BHATTACHARYYA)
        hist_similarity = max(0.0, 1.0 - hist_dist)

        # Gesamte Konfidenz: Gewichtete Kombination aller 3 Metriken
        combined_score = (feat_ratio * 0.3) + (template_score * 0.4) + (hist_similarity * 0.3)

        if combined_score > best_score:
            second_score = best_score
            best_score = combined_score
            best_code = code
            best_hist_score = hist_similarity
        elif combined_score > second_score:
            second_score = combined_score

    dt_ms = int((time.time() - t0) * 1000)

    # VERSCHÄRFTE Mindestanforderungen:
    # 1. Mindest-Konfidenz: 0.88 (vorher 0.78)
    # 2. Mindest-Margin: 0.15 (vorher 0.05) - deutlich größerer Abstand zum Zweitbesten
    # 3. Histogram-Ähnlichkeit muss mindestens 0.5 sein
    margin = best_score - second_score
    if (best_code and best_score >= min_confidence 
            and margin >= 0.15 
            and best_hist_score >= 0.5):
        logger.info(
            f"[HORDEN-DB MATCH] Horde '{best_code}' erfolgreich per Bildabgleich erkannt! "
            f"(Score={best_score:.2f}, Margin={margin:.2f}, HistSim={best_hist_score:.2f}, Dauer={dt_ms}ms)"
        )
        return {
            "success": True,
            "result": best_code,
            "method": "Bildabgleich",
            "confidence": min(1.0, best_score),
            "dmtx_result": None,
            "ocr_result": None,
            "verified": False,  # Bildabgleich ist NIEMALS verifiziert (keine mathematische Garantie)
            "ocr_partial_display": best_code,
            "_horde_db_matched": True,
        }

    logger.debug(f"[HORDEN-DB MATCH] Kein eindeutiger Bildabgleich-Treffer (Bester: {best_code} mit Score={best_score:.2f}, Margin={margin:.2f}, Dauer={dt_ms}ms).")
    return None
