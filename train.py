"""
train.py — Trainings-Skript für YOLOv10 zur Hordenerkennung.
Integriert Active Learning (Importiert automatisch Bilder aus auto_training_data/).
"""
import os
import sys
import shutil
import json
import logging
from ultralytics import YOLO

# Logging einrichten
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("KITraining")

def setup_directories(project_dir):
    """Erstellt alle benötigten Ordnerstrukturen, falls sie fehlen."""
    dirs = [
        os.path.join(project_dir, "dataset", "images", "train"),
        os.path.join(project_dir, "dataset", "images", "val"),
        os.path.join(project_dir, "dataset", "labels", "train"),
        os.path.join(project_dir, "dataset", "labels", "val"),
        os.path.join(project_dir, "auto_training_data"),
        os.path.join(project_dir, "training_data"),
    ]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            logger.info(f"Ordner erstellt: {d}")

def create_data_yaml(project_dir):
    """Erstellt die data.yaml Konfigurationsdatei für YOLO."""
    yaml_path = os.path.join(project_dir, "dataset", "data.yaml")
    # Pfade mit Forward-Slashes für YOLO Kompatibilität
    dataset_path_clean = os.path.join(project_dir, "dataset").replace("\\", "/")
    
    yaml_content = f"""path: {dataset_path_clean}
train: images/train
val: images/val

names:
  0: Horde
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    logger.info(f"data.yaml erstellt unter: {yaml_path}")

def import_active_learning(project_dir):
    """Importiert neue Bilder und Labels aus auto_training_data/ in den dataset/ Ordner."""
    auto_dir = os.path.join(project_dir, "auto_training_data")
    dataset_dir = os.path.join(project_dir, "dataset")
    
    # Finde alle Bilddateien (.jpg, .png)
    files = os.listdir(auto_dir)
    image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        logger.info("Keine neuen Active-Learning-Bilder in 'auto_training_data/' gefunden.")
        return
        
    logger.info(f"Importiere {len(image_files)} Bilder aus 'auto_training_data/'...")
    
    imported_count = 0
    for idx, img_name in enumerate(image_files):
        base_name, _ = os.path.splitext(img_name)
        label_name = base_name + ".txt"
        
        img_src = os.path.join(auto_dir, img_name)
        label_src = os.path.join(auto_dir, label_name)
        
        # Prüfen, ob auch die Label-Datei existiert
        if not os.path.exists(label_src):
            logger.warning(f"Bild {img_name} hat keine zugehörige Label-Datei {label_name}. Überspringe.")
            continue
            
        # Aufteilung in 80% Training und 20% Validation
        split = "train" if (idx % 5 != 0) else "val"
        
        img_dest = os.path.join(dataset_dir, "images", split, img_name)
        label_dest = os.path.join(dataset_dir, "labels", split, label_name)
        
        try:
            # Kopieren und anschließend Löschen (um Probleme bei Abbruch zu verhindern)
            shutil.copy2(img_src, img_dest)
            shutil.copy2(label_src, label_dest)
            
            os.remove(img_src)
            os.remove(label_src)
            imported_count += 1
        except Exception as e:
            logger.error(f"Fehler beim Importieren von {img_name}: {e}")
            
    logger.info(f"Erfolgreich {imported_count} Bilder in den Trainingsdatensatz integriert.")

def ensure_validation_split(project_dir):
    """Stellt sicher, dass mindestens einige Bilder im Validation-Ordner liegen."""
    train_img_dir = os.path.join(project_dir, "dataset", "images", "train")
    val_img_dir = os.path.join(project_dir, "dataset", "images", "val")
    train_lbl_dir = os.path.join(project_dir, "dataset", "labels", "train")
    val_lbl_dir = os.path.join(project_dir, "dataset", "labels", "val")
    
    if not os.path.exists(train_img_dir):
        return
        
    train_images = [f for f in os.listdir(train_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    val_images = [f for f in os.listdir(val_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if train_images and not val_images:
        # Mindestens ein Bild verschieben, idealerweise 20%
        num_to_move = max(1, int(len(train_images) * 0.2))
        logger.info(f"Keine Validierungsbilder gefunden. Verschiebe automatisch {num_to_move} Bilder von 'train' nach 'val'...")
        
        # Jedes n-te Bild auswählen
        step = max(1, len(train_images) // num_to_move)
        images_to_move = [train_images[i * step] for i in range(num_to_move) if i * step < len(train_images)]
        
        # Sicherstellen, dass wir wirklich etwas verschieben
        if not images_to_move:
            images_to_move = [train_images[0]]
            
        moved_count = 0
        for img_name in images_to_move:
            base_name, _ = os.path.splitext(img_name)
            label_name = base_name + ".txt"
            
            img_src = os.path.join(train_img_dir, img_name)
            img_dst = os.path.join(val_img_dir, img_name)
            label_src = os.path.join(train_lbl_dir, label_name)
            label_dst = os.path.join(val_lbl_dir, label_name)
            
            try:
                if os.path.exists(img_src):
                    shutil.move(img_src, img_dst)
                if os.path.exists(label_src):
                    shutil.move(label_src, label_dst)
                moved_count += 1
            except Exception as e:
                logger.error(f"Fehler beim Verschieben von {img_name} nach val: {e}")
        logger.info(f"{moved_count} Bilder erfolgreich nach 'val' verschoben.")

def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Verzeichnisse vorbereiten
    setup_directories(project_dir)
    
    # 2. data.yaml schreiben
    create_data_yaml(project_dir)
    
    # 3. Active Learning Bilder importieren
    import_active_learning(project_dir)
    
    # 3b. Sicherstellen, dass ein Validation Split existiert
    ensure_validation_split(project_dir)
    
    # 4. Prüfen ob überhaupt Bilder zum Trainieren da sind
    train_img_dir = os.path.join(project_dir, "dataset", "images", "train")
    if not os.listdir(train_img_dir):
        logger.error("Fehler: Keine Bilder im Ordner 'dataset/images/train/'.")
        logger.error("Bitte füge zuerst Bilder über Label Studio oder das Active Learning hinzu.")
        sys.exit(1)
        
    # 5. YOLO Modellpfad bestimmen (Feintuning von best.pt falls vorhanden, sonst yolov10n.pt)
    existing_model = os.path.join(project_dir, "runs", "detect", "training_runs", "horde_model", "weights", "best.pt")
    default_model = os.path.join(project_dir, "yolov10n.pt")
    
    if os.path.exists(existing_model):
        model_path = existing_model
        logger.info(f"Feintuning von existierendem Modell: {model_path}")
    elif os.path.exists(default_model):
        model_path = default_model
        logger.info(f"Starte neues Training mit Basis-Modell: {model_path}")
    else:
        logger.error(f"Basis-Modell nicht gefunden! Weder {existing_model} noch {default_model} existiert.")
        sys.exit(1)
        
    # 6. Training starten
    logger.info("Starte YOLOv10-Training...")
    try:
        model = YOLO(model_path)
        model.train(
            data=os.path.join(project_dir, "dataset", "data.yaml"),
            epochs=50,
            imgsz=640,
            project=os.path.join(project_dir, "runs", "detect"),
            name="training_runs/horde_model",
            exist_ok=True,
            device="cpu"  # Kann auf 'gpu' geändert werden, falls CUDA-fähige GPU vorhanden
        )
        logger.info("Training erfolgreich beendet!")
        
        # Kopiere das beste Modell zur Sicherheit an den Zielpfad (wird von YOLO gemacht, aber Log ausgeben)
        best_weights = os.path.join(project_dir, "runs", "detect", "training_runs", "horde_model", "weights", "best.pt")
        if os.path.exists(best_weights):
            logger.info(f"[OK] Das aktualisierte Modell liegt bereit unter: {best_weights}")
            
    except Exception as e:
        logger.error(f"Fehler während des Trainings: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
