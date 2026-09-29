"""
train_v2.py — Trainings-Skript für YOLOv10 mit 2 Klassen (datamatrix + text) auf dataset_v2/.

Nutzung:
    .venv\\Scripts\\python.exe train_v2.py
    .venv\\Scripts\\python.exe train_v2.py --epochs 100 --device gpu
"""

import os
import sys
import shutil
import logging
import argparse

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("KITraining_v2")


def setup_directories(project_dir: str, dataset_name: str = "dataset_v2"):
    """Erstellt alle benötigten Ordnerstrukturen, falls sie fehlen."""
    dirs = [
        os.path.join(project_dir, dataset_name, "images", "train"),
        os.path.join(project_dir, dataset_name, "images", "val"),
        os.path.join(project_dir, dataset_name, "labels", "train"),
        os.path.join(project_dir, dataset_name, "labels", "val"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def verify_data_yaml(project_dir: str, dataset_name: str = "dataset_v2"):
    """Prüft, ob die data.yaml für 2 Klassen korrekt konfiguriert ist."""
    yaml_path = os.path.join(project_dir, dataset_name, "data.yaml")
    
    if not os.path.exists(yaml_path):
        logger.error(
            f"data.yaml nicht gefunden in {yaml_path}!\n"
            f"Bitte zuerst relabel_2class.py ausführen:\n"
            f"  .venv\\Scripts\\python.exe relabel_2class.py"
        )
        return False
    
    # Inhalt prüfen
    with open(yaml_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    if "datamatrix" not in content or "text" not in content:
        logger.error(
            f"data.yaml hat nicht das 2-Klassen-Format!\n"
            f"Erwartet: names mit 'datamatrix' und 'text'.\n"
            f"Bitte relabel_2class.py erneut ausführen."
        )
        return False
    
    logger.info(f"data.yaml verifiziert: {yaml_path} (2 Klassen: datamatrix, text)")
    return True


def count_images(project_dir: str, dataset_name: str = "dataset_v2") -> tuple[int, int]:
    """Zählt Train- und Val-Bilder."""
    train_dir = os.path.join(project_dir, dataset_name, "images", "train")
    val_dir = os.path.join(project_dir, dataset_name, "images", "val")
    
    train_count = len([
        f for f in os.listdir(train_dir) 
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]) if os.path.exists(train_dir) else 0
    
    val_count = len([
        f for f in os.listdir(val_dir) 
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]) if os.path.exists(val_dir) else 0
    
    return train_count, val_count


def ensure_validation_split(project_dir: str, dataset_name: str = "dataset_v2"):
    """Stellt sicher, dass mindestens einige Bilder im Validation-Ordner liegen."""
    train_img_dir = os.path.join(project_dir, dataset_name, "images", "train")
    val_img_dir = os.path.join(project_dir, dataset_name, "images", "val")
    train_lbl_dir = os.path.join(project_dir, dataset_name, "labels", "train")
    val_lbl_dir = os.path.join(project_dir, dataset_name, "labels", "val")
    
    if not os.path.exists(train_img_dir):
        return
    
    train_images = [f for f in os.listdir(train_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    val_images = [f for f in os.listdir(val_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if train_images and not val_images:
        num_to_move = max(1, int(len(train_images) * 0.2))
        logger.info(f"Verschiebe automatisch {num_to_move} Bilder von train → val ...")
        
        step = max(1, len(train_images) // num_to_move)
        images_to_move = [train_images[i * step] for i in range(num_to_move) if i * step < len(train_images)]
        
        if not images_to_move:
            images_to_move = [train_images[0]]
        
        moved = 0
        for img_name in images_to_move:
            base_name = os.path.splitext(img_name)[0]
            label_name = base_name + ".txt"
            
            try:
                if os.path.exists(os.path.join(train_img_dir, img_name)):
                    shutil.move(
                        os.path.join(train_img_dir, img_name),
                        os.path.join(val_img_dir, img_name)
                    )
                if os.path.exists(os.path.join(train_lbl_dir, label_name)):
                    shutil.move(
                        os.path.join(train_lbl_dir, label_name),
                        os.path.join(val_lbl_dir, label_name)
                    )
                moved += 1
            except Exception as e:
                logger.error(f"Fehler beim Verschieben von {img_name}: {e}")
        
        logger.info(f"{moved} Bilder nach val verschoben.")


def main():
    parser = argparse.ArgumentParser(description="YOLOv10 Training mit 2 Klassen (datamatrix + text).")
    parser.add_argument("--epochs", type=int, default=80, help="Anzahl der Trainings-Epochen (Standard: 80)")
    parser.add_argument("--device", default="cpu", help="Training Device: 'cpu' oder 'gpu' (Standard: cpu)")
    parser.add_argument("--imgsz", type=int, default=640, help="Bildgröße für Training (Standard: 640)")
    parser.add_argument("--batch", type=int, default=8, help="Batch-Größe (Standard: 8)")
    parser.add_argument("--dataset", default="dataset_v2", help="Dataset-Ordner (Standard: dataset_v2)")
    parser.add_argument("--name", default="training_runs_v2/horde_2class",
                        help="Zielordner unter runs/detect (Standard: Produktionsmodell training_runs_v2/horde_2class)")
    parser.add_argument("--weights", default=None,
                        help="Startgewichte (Standard: vorhandenes 2-Klassen-Modell, sonst yolov10n.pt)")
    args = parser.parse_args()
    
    project_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_name = args.dataset
    
    # 1. Verzeichnisse vorbereiten
    setup_directories(project_dir, dataset_name)
    
    # 2. data.yaml prüfen
    if not verify_data_yaml(project_dir, dataset_name):
        sys.exit(1)
    
    # 3. Validation-Split sicherstellen
    ensure_validation_split(project_dir, dataset_name)
    
    # 4. Bilder zählen
    train_count, val_count = count_images(project_dir, dataset_name)
    logger.info(f"Dataset: {train_count} Train-Bilder, {val_count} Val-Bilder")
    
    if train_count == 0:
        logger.error(
            f"Keine Trainingsbilder in {dataset_name}/images/train/ gefunden!\n"
            f"Bitte zuerst relabel_2class.py ausführen."
        )
        sys.exit(1)
    
    # 5. YOLO importieren
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics nicht installiert! pip install ultralytics")
        sys.exit(1)
    
    # 6. Modellpfad bestimmen: Feintuning des vorhandenen 2-Klassen-Modells, sonst Start vom Basismodell
    #    (das alte 1-Klassen-Modell ist wegen der anderen Klassenanzahl ungeeignet)
    v2_model_path = os.path.join(
        project_dir, "runs", "detect", "training_runs_v2",
        "horde_2class", "weights", "best.pt"
    )
    base_model_path = os.path.join(project_dir, "yolov10n.pt")
    
    if args.weights:
        model_path = os.path.abspath(args.weights)
        logger.info(f"Training mit Startgewichten: {model_path}")
    elif os.path.exists(v2_model_path):
        model_path = v2_model_path
        logger.info(f"Feintuning von bestehendem 2-Klassen-Modell: {model_path}")
    elif os.path.exists(base_model_path):
        model_path = base_model_path
        logger.info(f"Neues 2-Klassen-Training von Basis-Modell: {model_path}")
    else:
        logger.error(f"Kein Basis-Modell gefunden: {base_model_path}")
        sys.exit(1)
    
    # 7. Training starten
    data_yaml_path = os.path.join(project_dir, dataset_name, "data.yaml")
    
    device = "0" if args.device.lower() in ("gpu", "cuda") else "cpu"
    
    logger.info(f"\n{'=' * 60}")
    logger.info(f"YOLO v2 TRAINING START")
    logger.info(f"  Modell:   {model_path}")
    logger.info(f"  Dataset:  {data_yaml_path}")
    logger.info(f"  Epochen:  {args.epochs}")
    logger.info(f"  Batch:    {args.batch}")
    logger.info(f"  ImgSize:  {args.imgsz}")
    logger.info(f"  Device:   {device}")
    logger.info(f"{'=' * 60}\n")
    
    try:
        model = YOLO(model_path)
        model.train(
            data=data_yaml_path,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=os.path.join(project_dir, "runs", "detect"),
            name=args.name,
            exist_ok=True,
            device=device,
            # Augmentierung-Parameter für kleine Datasets
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=5.0,
            translate=0.1,
            scale=0.3,
            flipud=0.0,     # Nicht vertikal spiegeln (Schrift wird unleserlich)
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING ERFOLGREICH ABGESCHLOSSEN!")
        
        best_weights = os.path.join(project_dir, "runs", "detect", *args.name.split("/"), "weights", "best.pt")
        if os.path.exists(best_weights):
            logger.info(f"[OK] Bestes Modell: {best_weights}")
            logger.info("\nNächster Schritt: Benchmark ausführen mit")
            logger.info("  .venv\\Scripts\\python.exe benchmark_gui.py --headless")
        
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Fehler während des Trainings: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
