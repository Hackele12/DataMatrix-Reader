import os
import sys
import json
import time
import cv2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ultralytics import YOLO
import scanner
import horde_db


def process_image(img_path, model):
    img = cv2.imread(img_path)
    if img is None:
        return "?"

    results = model.predict(img, conf=0.15, verbose=False)
    yolo_detections = []
    if results and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if conf > 0.25:
                yolo_detections.append({"cls": cls_id, "box": (x1, y1, x2, y2), "conf": conf})

    if yolo_detections:
        res = scanner.scan_2class(img, yolo_detections)
        if res.get("success") and res.get("result"):
            return res["result"]

    res = scanner.scan(img)
    if res.get("success") and res.get("result"):
        return res["result"]

    hm = horde_db.match_horde_image(img, min_confidence=0.60)
    if hm and hm.get("success") and hm.get("horde_code"):
        return hm["horde_code"]

    return "?"


def main():
    app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    images_dir = os.path.join(app_dir, "training_data")
    gt_file = os.path.join(app_dir, "ground_truth.json")

    horde_db.load_horde_db()

    model_path = os.path.join(app_dir, "runs", "detect", "training_runs_v2", "horde_2class", "weights", "best.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(app_dir, "runs", "detect", "training_runs", "horde_model", "weights", "best.pt")

    print(f"Lade YOLO Modell: {model_path}")
    model = YOLO(model_path)

    gt_dict = {}
    if os.path.exists(gt_file):
        with open(gt_file, "r", encoding="utf-8") as f:
            gt_dict = json.load(f)

    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    print(f"\n==========================================")
    print(f"Befülle ground_truth.json für {len(image_files)} Bilder...")
    print(f"Bereits vorhanden: {len(gt_dict)} Einträge")
    print(f"==========================================\n")

    t0 = time.time()
    added_count = 0

    for i, fn in enumerate(image_files, 1):
        if fn in gt_dict and gt_dict[fn] != "?":
            continue

        img_path = os.path.join(images_dir, fn)
        code = process_image(img_path, model)
        gt_dict[fn] = code
        added_count += 1

        print(f"  [{i:03d}/{len(image_files)}] {fn} -> {code}", flush=True)

        sorted_gt = {k: gt_dict[k] for k in sorted(gt_dict.keys())}
        with open(gt_file, "w", encoding="utf-8") as f:
            json.dump(sorted_gt, f, indent=2, ensure_ascii=False)

    dt = time.time() - t0
    rec_count = sum(1 for v in gt_dict.values() if v != "?")
    print(f"\n==========================================")
    print(f"GROUND TRUTH ERFOLGREICH VOLLSTÄNDIG ERSTELLT!")
    print(f"Gesamteinträge: {len(gt_dict)} / {len(image_files)}")
    print(f"Erkannte Codes: {rec_count} / {len(gt_dict)} ({100*rec_count/len(gt_dict):.1f}%)")
    print(f"Verarbeitungsdauer: {dt:.2f}s")
    print(f"==========================================")


if __name__ == "__main__":
    main()
