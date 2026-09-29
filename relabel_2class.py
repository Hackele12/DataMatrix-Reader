"""
relabel_2class.py — KI- und Feature-gestütztes Relabeling von 1-Klasse (Horde) auf 2 Klassen (datamatrix + text).

Erkennt automatisch die exakten Positionen im Bild via:
  1. ZXing Barcode-Detektor (exakte DMX-Ecken)
  2. CRAFT/EasyOCR Text-Detektor (exakte Klarschrift-Box)
  3. Geometrische Relativ-Kopplung (falls Barcode verwaschen ist, wird er präzise über der Klarschrift platziert)

Nutzung:
    .venv\\Scripts\\python.exe relabel_2class.py
    .venv\\Scripts\\python.exe relabel_2class.py --visualize
"""

import os
import sys
import shutil
import logging
import argparse

import cv2
import numpy as np
import zxingcpp
import easyocr

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Relabel2Class")

# Globaler OCR-Reader (Lazy Init)
_ocr_reader = None

def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        logger.info("Initialisiere Text-Detektor für präzise Boxen-Erkennung...")
        _ocr_reader = easyocr.Reader(['de', 'en'], gpu=False)
    return _ocr_reader


def detect_exact_boxes(image: np.ndarray) -> tuple[
    tuple[float, float, float, float] | None,
    tuple[float, float, float, float] | None
]:
    """
    Erkennt die exakten Bounding-Boxes für DataMatrix (Klasse 0) und Text (Klasse 1)
    im Bild mittels Computer Vision & OCR.
    
    Returns:
        (dmx_box_norm, text_box_norm) jeweils als (x_center, y_center, w, h) in 0-1 Koordinaten
    """
    h_img, w_img = image.shape[:2]
    
    dmx_abs = None  # (x1, y1, x2, y2)
    text_abs = None # (x1, y1, x2, y2)
    
    # 1. Barcode via ZXing suchen
    zx_res = zxingcpp.read_barcode(image)
    if zx_res and zx_res.position:
        pos = zx_res.position
        pts = [(p.x, p.y) for p in [pos.top_left, pos.top_right, pos.bottom_right, pos.bottom_left]]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        
        # Inklusive großzügiger Quiet Zone (18% Padding)
        pad = max(18, int(max(max(xs)-min(xs), max(ys)-min(ys)) * 0.18))
        x1 = max(0, min(xs) - pad)
        y1 = max(0, min(ys) - pad)
        x2 = min(w_img, max(xs) + pad)
        y2 = min(h_img, max(ys) + pad)
        dmx_abs = (x1, y1, x2, y2)
    
    # 2. Text via EasyOCR suchen
    reader = get_ocr_reader()
    ocr_res = reader.readtext(image)
    
    best_txt_box = None
    best_prob = 0.0
    
    for bbox, text, prob in ocr_res:
        clean = text.strip()
        if len(clean) >= 3 and prob > best_prob:
            best_prob = prob
            pts = np.array(bbox, dtype=np.int32)
            tx1, ty1 = np.min(pts, axis=0)
            tx2, ty2 = np.max(pts, axis=0)
            
            # Großzügiges Text-Padding (14px horizontal, 6px vertikal)
            tx1 = max(0, tx1 - 14)
            ty1 = max(0, ty1 - 6)
            tx2 = min(w_img, tx2 + 14)
            ty2 = min(h_img, ty2 + 6)
            best_txt_box = (tx1, ty1, tx2, ty2)
    
    if best_txt_box:
        text_abs = best_txt_box
    
    # 3. Falls DMX nicht direkt erkannt wurde (z.B. ausgebleicht),
    # berechne DMX exakt basierend auf der Klarschrift-Position
    if text_abs and not dmx_abs:
        tx1, ty1, tx2, ty2 = text_abs
        th = ty2 - ty1
        tw = tx2 - tx1
        
        dmx_h = int(th * 3.3)
        dmx_w = max(int(tw * 1.15), dmx_h)
        dmx_cx = (tx1 + tx2) // 2
        dmx_y2 = max(0, ty1 - int(th * 0.15))
        dmx_y1 = max(0, dmx_y2 - dmx_h)
        dmx_x1 = max(0, dmx_cx - dmx_w // 2)
        dmx_x2 = min(w_img, dmx_cx + dmx_w // 2)
        
        dmx_abs = (dmx_x1, dmx_y1, dmx_x2, dmx_y2)
    
    # 4. Falls Text nicht erkannt wurde, berechne Text direkt unter DMX
    if dmx_abs and not text_abs:
        dx1, dy1, dx2, dy2 = dmx_abs
        dw = dx2 - dx1
        dh = dy2 - dy1
        
        th = int(dh * 0.38)
        tw = int(dw * 1.05)
        tcx = (dx1 + dx2) // 2
        ty1 = min(h_img - th, dy2 + int(dh * 0.05))
        ty2 = min(h_img, ty1 + th)
        tx1 = max(0, tcx - tw // 2)
        tx2 = min(w_img, tcx + tw // 2)
        
        text_abs = (tx1, ty1, tx2, ty2)
    
    if not dmx_abs or not text_abs:
        return None, None
    
    # In normierte YOLO-Koordinaten (0-1) umrechnen: (x_center, y_center, width, height)
    def to_yolo(abs_box):
        x1, y1, x2, y2 = abs_box
        xc = ((x1 + x2) / 2.0) / w_img
        yc = ((y1 + y2) / 2.0) / h_img
        w = (x2 - x1) / w_img
        h = (y2 - y1) / h_img
        return (xc, yc, w, h)
    
    return to_yolo(dmx_abs), to_yolo(text_abs)


def parse_yolo_label(label_path: str) -> list[tuple[int, float, float, float, float]]:
    """Liest eine YOLO-Label-Datei."""
    labels = []
    try:
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    x_c, y_c, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    labels.append((cls, x_c, y_c, w, h))
    except Exception as e:
        logger.error(f"Fehler beim Lesen von {label_path}: {e}")
    return labels


def visualize_labels(image_path: str, label_path: str, window_name: str = "Labels"):
    """Zeigt ein Bild mit eingezeichneten YOLO-Boxen zur visuellen Kontrolle."""
    image = cv2.imread(image_path)
    if image is None:
        return
    
    h_img, w_img = image.shape[:2]
    labels = parse_yolo_label(label_path)
    
    colors = {0: (0, 255, 0), 1: (255, 165, 0)}  # Grün=DMX, Orange=Text
    names = {0: "datamatrix", 1: "text"}
    
    for cls, x_c, y_c, w, h in labels:
        x1 = int((x_c - w / 2) * w_img)
        y1 = int((y_c - h / 2) * h_img)
        x2 = int((x_c + w / 2) * w_img)
        y2 = int((y_c + h / 2) * h_img)
        
        color = colors.get(cls, (128, 128, 128))
        label = names.get(cls, f"cls{cls}")
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Auf sinnvolle Größe skalieren
    max_dim = 900
    scale = min(max_dim / w_img, max_dim / h_img, 1.0)
    if scale < 1.0:
        image = cv2.resize(image, (int(w_img * scale), int(h_img * scale)))
    
    cv2.imshow(window_name, image)
    key = cv2.waitKey(0)
    return key


def main():
    parser = argparse.ArgumentParser(description="KI-gestütztes Relabeling auf 2 Klassen (datamatrix + text).")
    parser.add_argument("--visualize", action="store_true", help="Ergebnisse visuell anzeigen (Leertaste = weiter, ESC = abbrechen)")
    parser.add_argument("--source", default="dataset", help="Quell-Dataset-Ordner (Standard: dataset)")
    parser.add_argument("--target", default="dataset_v2", help="Ziel-Dataset-Ordner (Standard: dataset_v2)")
    args = parser.parse_args()
    
    project_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(project_dir, args.source)
    target_dir = os.path.join(project_dir, args.target)
    
    if not os.path.exists(source_dir):
        logger.error(f"Quell-Verzeichnis nicht gefunden: {source_dir}")
        sys.exit(1)
    
    # Ziel-Verzeichnisstruktur erstellen
    for split in ["train", "val"]:
        os.makedirs(os.path.join(target_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(target_dir, "labels", split), exist_ok=True)
    
    # data.yaml für 2 Klassen erstellen
    data_yaml_path = os.path.join(target_dir, "data.yaml")
    target_dir_clean = target_dir.replace("\\", "/")
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        f.write(f"""path: {target_dir_clean}
train: images/train
val: images/val

names:
  0: datamatrix
  1: text
""")
    logger.info(f"data.yaml erstellt: {data_yaml_path}")
    
    total_success = 0
    total_skipped = 0
    
    for split in ["train", "val"]:
        img_dir = os.path.join(source_dir, "images", split)
        lbl_dir = os.path.join(source_dir, "labels", split)
        
        if not os.path.exists(img_dir):
            continue
        
        image_files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        
        logger.info(f"Verarbeite {len(image_files)} Bilder aus {split}/ ...")
        
        for img_name in image_files:
            base_name = os.path.splitext(img_name)[0]
            label_name = base_name + ".txt"
            
            old_img_path = os.path.join(img_dir, img_name)
            new_img_path = os.path.join(target_dir, "images", split, img_name)
            new_lbl_path = os.path.join(target_dir, "labels", split, label_name)
            
            # Bild laden
            image = cv2.imread(old_img_path)
            if image is None:
                total_skipped += 1
                continue
            
            # Bild kopieren (falls Quelle und Ziel verschieden)
            if os.path.abspath(old_img_path) != os.path.abspath(new_img_path):
                try:
                    shutil.copy2(old_img_path, new_img_path)
                except Exception as e:
                    logger.warning(f"Kopieren fehlgeschlagen ({img_name}): {e}")
            
            # Exakte Bounding-Boxes via Computer Vision & OCR ermitteln
            dmx_box, text_box = detect_exact_boxes(image)
            
            # Fallback für extrem verätzte Bilder: Ursprüngliches Label lesen und aufteilen
            if not dmx_box or not text_box:
                old_lbl_path = os.path.join(lbl_dir, label_name)
                if os.path.exists(old_lbl_path):
                    orig_labels = parse_yolo_label(old_lbl_path)
                    if orig_labels:
                        _, oxc, oyc, ow, oh = orig_labels[0]
                        # Top 55% DMX, Bottom 45% Text
                        top = oyc - oh / 2.0
                        bot = oyc + oh / 2.0
                        dmx_h = oh * 0.55
                        txt_h = oh * 0.40
                        dmx_box = (oxc, top + dmx_h / 2.0, ow * 0.90, dmx_h)
                        text_box = (oxc, bot - txt_h / 2.0, ow * 0.95, txt_h)
                        logger.info(f"Fallback-Aufteilung angewendet für {img_name}")
            
            if dmx_box and text_box:
                new_labels = [
                    f"0 {dmx_box[0]:.6f} {dmx_box[1]:.6f} {dmx_box[2]:.6f} {dmx_box[3]:.6f}",
                    f"1 {text_box[0]:.6f} {text_box[1]:.6f} {text_box[2]:.6f} {text_box[3]:.6f}"
                ]
                with open(new_lbl_path, "w") as f:
                    f.write("\n".join(new_labels) + "\n")
                
                total_success += 1
                
                if args.visualize:
                    key = visualize_labels(new_img_path, new_lbl_path, f"{split}: {img_name}")
                    if key == 27:  # ESC
                        logger.info("Visualisierung abgebrochen.")
                        args.visualize = False
            else:
                logger.warning(f"Konnte keine exakten Boxen ermitteln für {img_name}")
                total_skipped += 1
    
    if args.visualize:
        cv2.destroyAllWindows()
    
    logger.info(f"\n{'=' * 60}")
    logger.info(f"RELABELING ABGESCHLOSSEN (KI-Erkennung)")
    logger.info(f"  Erfolgreich gelabelt: {total_success} Bilder")
    logger.info(f"  Fehlgeschlagen:       {total_skipped} Bilder")
    logger.info(f"  Ziel-Ordner:          {target_dir}")
    logger.info(f"  data.yaml:            {data_yaml_path}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
