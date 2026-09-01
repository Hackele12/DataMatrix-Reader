import os
import sys
import json
import time
import logging
import cv2
import numpy as np

# Umgebungsvariablen für PyTorch / OpenCV
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Systempfad anpassen
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import scanner
from evaluate_scanner import _load_yolo_model, _detect_and_crop


def renumber_training_data(images_dir: str) -> list[str]:
    """
    Benennt alle Bilddateien in images_dir sequentiell in '001.jpg', '002.jpg', ... um.
    """
    valid_exts = (".jpg", ".jpeg", ".png")
    files = sorted([f for f in os.listdir(images_dir) if os.path.splitext(f)[1].lower() in valid_exts])
    total = len(files)
    logger.info(f"Gefunden: {total} Bilder in '{images_dir}'. Starte Umbenennung...")

    # Schritt 1: Temporäre Umbenennung zur Vermeidung von Namenskonflikten
    temp_map = []
    for idx, old_name in enumerate(files, 1):
        ext = os.path.splitext(old_name)[1].lower()
        temp_name = f"__temp_renum_{idx:04d}{ext}"
        old_path = os.path.join(images_dir, old_name)
        temp_path = os.path.join(images_dir, temp_name)
        os.rename(old_path, temp_path)
        temp_map.append((temp_path, ext))

    # Schritt 2: Finale Umbenennung (001.jpg, 002.jpg, ...)
    final_files = []
    pad_len = max(3, len(str(total)))  # mind. 3-stellig (001, 002, ...)
    for idx, (temp_path, ext) in enumerate(temp_map, 1):
        new_name = f"{idx:0{pad_len}d}.jpg"
        new_path = os.path.join(images_dir, new_name)
        os.rename(temp_path, new_path)
        final_files.append(new_name)

    logger.info(f"Umbenennung abgeschlossen! {total} Bilder sind jetzt von {final_files[0]} bis {final_files[-1]} benannt.")
    return final_files


def generate_ground_truth(images_dir: str, output_gt_path: str):
    """
    Wertetet alle durchnummerierten Bilder in images_dir per YOLO + Scanner aus
    und speichert die voraussichtlichen Werte in ground_truth.json.
    """
    model = _load_yolo_model()
    if model is None:
        logger.error("YOLO-Modell konnte nicht geladen werden!")
        sys.exit(1)

    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    logger.info(f"Starte Auswertung von {len(image_files)} Bildern für 'ground_truth.json'...")

    gt_data = {}
    success_count = 0

    t0 = time.time()
    for i, fn in enumerate(image_files, 1):
        img_path = os.path.join(images_dir, fn)
        img = cv2.imread(img_path)

        if img is None:
            logger.warning(f"[{i}/{len(image_files)}] {fn}: Bild konnte nicht geladen werden.")
            gt_data[fn] = "?"
            continue

        # YOLO Zuschnitt + Begradigung
        cropped, yolo_conf = _detect_and_crop(model, img)

        # Scan ausführen
        scan_res = scanner.scan(cropped)

        if scan_res.get("success") and scan_res.get("result"):
            code = scan_res["result"]
            gt_data[fn] = code
            success_count += 1
            logger.info(f"[{i}/{len(image_files)}] {fn} -> {code} ({scan_res.get('method')}, conf={scan_res.get('confidence', 0):.2f})")
        else:
            gt_data[fn] = "?"
            logger.warning(f"[{i}/{len(image_files)}] {fn} -> Nicht erkannt (?)")

    # In ground_truth.json speichern
    with open(output_gt_path, "w", encoding="utf-8") as f:
        json.dump(gt_data, f, indent=2, ensure_ascii=False)

    total_time = time.time() - t0
    logger.info(f"\n==========================================")
    logger.info(f"GROUND TRUTH ERSTELLT: {output_gt_path}")
    logger.info(f"Erkannte Codes: {success_count} / {len(image_files)} ({100*success_count/len(image_files):.1f}%)")
    logger.info(f"Nicht erkannt (?): {len(image_files) - success_count}")
    logger.info(f"Gesamtdauer: {total_time:.2f}s ({total_time/len(image_files)*1000:.0f}ms / Bild)")
    logger.info(f"==========================================")


if __name__ == "__main__":
    app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    training_dir = os.path.join(app_dir, "training_data")
    gt_file = os.path.join(app_dir, "ground_truth.json")

    # 1. Bilder durchnummerieren
    renumber_training_data(training_dir)

    # 2. Voraussichtliche Werte per KI/Scanner ermitteln & ground_truth.json schreiben
    generate_ground_truth(training_dir, gt_file)
