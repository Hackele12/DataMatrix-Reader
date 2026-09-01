import os
import sys
import cv2
import numpy as np
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import horde_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_horde_guard")

def main():
    dummy_img = np.zeros((320, 320, 3), dtype=np.uint8)

    logger.info("--- TEST 1: Normaler schneller Scan (verified=False, conf=0.80, is_late=False) ---")
    res1 = horde_db.save_or_update_horde_image("W001", dummy_img, is_late_scan=False, verified=False, confidence=0.80)
    logger.info(f"Ergebnis: {res1} (sollte gespeichert werden)")

    logger.info("\n--- TEST 2: Späterkennung (>6s) UNVERIFIZIERT (verified=False, conf=0.70, is_late=True) ---")
    res2 = horde_db.save_or_update_horde_image("W081", dummy_img, is_late_scan=True, verified=False, confidence=0.70)
    logger.info(f"Ergebnis: {res2} (sollte ABGELEHNT werden: None)")

    logger.info("\n--- TEST 3: Späterkennung (>6s) VERIFIZIERT (verified=True, conf=0.98, is_late=True) ---")
    res3 = horde_db.save_or_update_horde_image("W031", dummy_img, is_late_scan=True, verified=True, confidence=0.98)
    logger.info(f"Ergebnis: {res3} (sollte gespeichert werden)")

    assert res2 is None, "Test 2 fehlgeschlagen! Unverifizierter Late-Scan hätte blockiert werden müssen."
    assert res3 is not None, "Test 3 fehlgeschlagen! Verifizierter Late-Scan hätte gespeichert werden müssen."
    
    logger.info("\n✓ ALLE SCHUTZ-TESTS ERFOLGREICH BESTANDEN!")

if __name__ == "__main__":
    main()
