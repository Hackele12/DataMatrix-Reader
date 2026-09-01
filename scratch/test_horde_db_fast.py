import os
import sys
import time
import logging
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import horde_db


def create_dummy_horde_image(text: str = "W003", seed=42) -> np.ndarray:
    np.random.seed(seed)
    img = np.ones((600, 800, 3), dtype=np.uint8) * 240
    noise = np.random.randint(0, 30, (600, 800, 3), dtype=np.uint8)
    img = cv2.subtract(img, noise)
    cv2.rectangle(img, (200, 150), (600, 450), (255, 255, 255), -1)
    cv2.rectangle(img, (200, 150), (600, 450), (50, 50, 50), 3)
    cv2.putText(img, text, (380, 320), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (10, 10, 10), 4)
    return img


def run_fast_tests():
    logger.info("=== TEST 1: Speichern in horden_db ===")
    test_code = "W003"
    img1 = create_dummy_horde_image(test_code, seed=100)

    saved_path = horde_db.save_or_update_horde_image(test_code, img1)
    assert saved_path is not None and os.path.exists(saved_path)
    logger.info(f"OK: Gespeichert -> {saved_path}")

    logger.info("=== TEST 2: Visueller Bildabgleich ===")
    test_input = cv2.add(img1, np.random.randint(0, 10, img1.shape, dtype=np.uint8))
    match_res = horde_db.match_horde_image(test_input, min_confidence=0.60)
    assert match_res is not None and match_res["success"] is True
    assert match_res["result"] == test_code
    logger.info(f"OK: Bildabgleich -> {match_res['result']} (Score: {match_res['confidence']:.2f})")

    logger.info("=== TEST 3: Automatische DB-Aktualisierung ===")
    mtime_1 = os.path.getmtime(saved_path)
    time.sleep(0.1)
    img2 = create_dummy_horde_image(test_code, seed=200)
    updated_path = horde_db.save_or_update_horde_image(test_code, img2)
    mtime_2 = os.path.getmtime(updated_path)
    assert mtime_2 > mtime_1
    logger.info("OK: DB-Bild wurde mit neuem Frame aktualisiert!")

    logger.info("=== TEST 4: Späterkennungs-Speicherung (>6s) ===")
    late_code = "W007"
    late_img = create_dummy_horde_image(late_code, seed=300)
    late_path = horde_db.save_or_update_horde_image(late_code, late_img, is_late_scan=True)
    assert late_path is not None and os.path.exists(late_path)
    logger.info(f"OK: Späterkennung im Zusatzordner -> {late_path}")

    logger.info("=== SÄMTLICHE FUNKTIONSTESTS ERFOLGREICH! ===")


if __name__ == "__main__":
    run_fast_tests()
