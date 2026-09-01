import os
import sys
import time
import logging
import cv2
import numpy as np

# Log-Setup für den Test
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Repository Root zum Syspath hinzufügen
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import horde_db
import scanner


def create_dummy_horde_image(text: str = "W003", color=(200, 200, 200), seed=42) -> np.ndarray:
    """Erzeugt ein synthetisches Horden-Testbild mit eindeutigen Texturen und Code-Text."""
    np.random.seed(seed)
    img = np.ones((600, 800, 3), dtype=np.uint8) * 240

    # Rauschen / Horden-Struktur simulieren
    noise = np.random.randint(0, 30, (600, 800, 3), dtype=np.uint8)
    img = cv2.subtract(img, noise)

    # Etiketten-Hintergrund
    cv2.rectangle(img, (200, 150), (600, 450), (255, 255, 255), -1)
    cv2.rectangle(img, (200, 150), (600, 450), (50, 50, 50), 3)

    # DataMatrix Simulation (Muster)
    for row in range(5):
        for col in range(5):
            if (row + col) % 2 == 0:
                cv2.rectangle(img, (240 + col*20, 200 + row*20), (255 + col*20, 215 + row*20), (0, 0, 0), -1)

    # Text auf Etikett schreiben
    cv2.putText(img, text, (380, 320), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (10, 10, 10), 4)
    return img


def test_horde_db_workflow():
    logger.info("=== TEST 1: Speichern & Auslesen in horden_db ===")
    test_code = "W003"
    img1 = create_dummy_horde_image(test_code, seed=100)

    # 1. Speichern
    saved_path = horde_db.save_or_update_horde_image(test_code, img1)
    assert saved_path is not None, "Speichern fehlgeschlagen!"
    assert os.path.exists(saved_path), f"Datei existiert nicht: {saved_path}"
    logger.info(f"[PASS] Bild gespeichert unter: {saved_path}")

    # 2. Bildabgleich
    logger.info("=== TEST 2: Visueller Bildabgleich ===")
    # Erzeuge leicht verändertes Testbild der gleichen Horde (z. B. leichtes Rauschen)
    test_img_input = img1.copy()
    noise = np.random.randint(0, 15, test_img_input.shape, dtype=np.uint8)
    test_img_input = cv2.add(test_img_input, noise)

    match_res = horde_db.match_horde_image(test_img_input, min_confidence=0.60)
    assert match_res is not None, "Bildabgleich hat nichts erkannt!"
    assert match_res["success"] is True, "Bildabgleich meldet success=False"
    assert match_res["result"] == test_code, f"Erwartete '{test_code}', erhielt '{match_res['result']}'"
    logger.info(f"[PASS] Bildabgleich erfolgreich! Treffer: '{match_res['result']}' (Score={match_res['confidence']:.2f})")

    # 3. Datenbankbild Aktualisierung
    logger.info("=== TEST 3: Automatische Datenbank-Aktualisierung ===")
    img2 = create_dummy_horde_image(test_code, seed=200)
    mtime_before = os.path.getmtime(saved_path)
    time.sleep(0.1)

    updated_path = horde_db.save_or_update_horde_image(test_code, img2)
    mtime_after = os.path.getmtime(updated_path)

    assert updated_path == saved_path, "Pfad hat sich geändert"
    assert mtime_after > mtime_before, "Zeitstempel wurde nicht aktualisiert!"
    logger.info(f"[PASS] Datenbankbild erfolgreich überschrieben/aktualisiert ({mtime_before:.2f} -> {mtime_after:.2f}).")

    # 4. Scanner Integration Test
    logger.info("=== TEST 4: Scanner.py Fallback auf Horden-DB ===")
    # Ein unleserliches/unscharfes Bild erzeugen, das weder DMX noch OCR liest, aber visual match triggert
    blurry_img = cv2.GaussianBlur(img1, (21, 21), 0)
    scan_res = scanner.scan(blurry_img)
    logger.info(f"Scanner Ergebnis für unscharfes Bild: {scan_res}")
    assert scan_res.get("success") is True, "Scanner Fallback auf HordeDB fehlgeschlagen!"
    assert scan_res.get("result") == test_code, f"Erwartet '{test_code}', erhalten '{scan_res.get('result')}'"
    logger.info(f"[PASS] Scanner-Integration erfolgreich! Method: {scan_res.get('method')}")

    # 5. Simulated Late Scan Saver Test (>6s Timeout)
    logger.info("=== TEST 5: Späterkennung (>6s Timeout) & Zusatzordner-Speicherung ===")
    late_code = "W005"
    late_img = create_dummy_horde_image(late_code, seed=300)

    # Simuliere Späterkennungs-Aufruf
    late_saved = horde_db.save_or_update_horde_image(late_code, late_img, is_late_scan=True)
    assert late_saved is not None and os.path.exists(late_saved), "Späterkennungs-Speicherung fehlgeschlagen!"
    logger.info(f"[PASS] Späterkennungs-Bild im Zusatzordner gespeichert: {late_saved}")

    logger.info("=== ALLE TESTS ERFOLGREICH BESTANDEN! ===")


if __name__ == "__main__":
    test_horde_db_workflow()
