"""
train_pacc_real.py — Fine-Tuning des Position-Aware Char Classifier (PACC) mit echten Bildern.

Extrahiert Text-Crops aus den 172 echten Bildern in training_data/ und trainiert das
PACC Klarschrift-Modell mit einer Mischung aus realen und synthetischen Daten nach.
Exportiert anschließend automatisch das neue ONNX-Modell.
"""

import os
import sys
import json
import glob
import random
import logging
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("PACCRealTraining")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
TRAIN_DATA_DIR = os.path.join(PROJECT_DIR, "training_data")
SUMMARY_JSON = os.path.join(PROJECT_DIR, "benchmarks_split_results", "combined_batch_benchmark_summary.json")
GT_JSON = os.path.join(PROJECT_DIR, "ground_truth.json")

sys.path.insert(0, MODEL_DIR)
sys.path.insert(0, PROJECT_DIR)

from char_classifier import create_model, PREFIX_CLASSES
from train_char_classifier import (
    generate_all_codes,
    _find_system_fonts,
    render_code_pil,
    render_code_cv2,
    augment_text_image,
    PIL_AVAILABLE,
    PREFIXES,
    IMG_WIDTH,
    IMG_HEIGHT
)

def load_real_gt_mappings() -> dict[str, str]:
    """Lädt die Ground-Truth Zuordnung für alle 172 echten Bilder."""
    gt_map = {}
    
    # 1. Aus combined_batch_benchmark_summary.json lesen
    if os.path.exists(SUMMARY_JSON):
        try:
            with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data.get("all_details", []):
                    fn = item.get("filename")
                    gt = item.get("expected_gt")
                    if fn and gt and gt != "?":
                        gt_map[fn] = gt
            logger.info(f"Loaded {len(gt_map)} GT entries from summary json.")
        except Exception as e:
            logger.warning(f"Could not load summary json: {e}")

    # 2. Aus ground_truth.json ergänzen
    if os.path.exists(GT_JSON):
        try:
            with open(GT_JSON, "r", encoding="utf-8") as f:
                gt_data = json.load(f)
                for fn, gt in gt_data.items():
                    if fn not in gt_map and gt != "?":
                        gt_map[fn] = gt
        except Exception as e:
            logger.warning(f"Could not load ground_truth.json: {e}")

    return gt_map


def extract_real_crops(gt_map: dict[str, str]) -> list[tuple[np.ndarray, str]]:
    """Extrahiert Text-Crops aus den echten Bildern in training_data/."""
    real_crops = []
    
    try:
        from ultralytics import YOLO
        import scanner
        model_path = os.path.join(PROJECT_DIR, "runs", "detect", "training_runs_v2", "horde_2class", "weights", "best.pt")
        if not os.path.exists(model_path):
            model_path = os.path.join(PROJECT_DIR, "runs", "detect", "training_runs", "horde_model", "weights", "best.pt")
        if not os.path.exists(model_path):
            model_path = os.path.join(PROJECT_DIR, "yolov10n.pt")
        yolo_model = YOLO(model_path) if os.path.exists(model_path) else None
    except Exception as e:
        logger.warning(f"YOLO not available for crop extraction: {e}")
        yolo_model = None

    image_files = sorted(glob.glob(os.path.join(TRAIN_DATA_DIR, "*.jpg")) + glob.glob(os.path.join(TRAIN_DATA_DIR, "*.png")))
    logger.info(f"Extracting crops from {len(image_files)} real images...")

    for img_path in image_files:
        fn = os.path.basename(img_path)
        gt_code = gt_map.get(fn)
        if not gt_code or len(gt_code) != 4 or gt_code[0] not in PREFIXES:
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]

        # Etikett-Zuschnitt via YOLO (falls vorhanden)
        label_crop = img
        if yolo_model is not None and h > 200 and w > 200:
            try:
                results = yolo_model.predict(img, conf=0.15, verbose=False)
                boxes = results[0].boxes
                if len(boxes) > 0:
                    x1, y1, x2, y2 = map(int, boxes[0].xyxy[0])
                    label_crop = scanner.deskew_crop(img, (x1, y1, x2, y2), padding=40)
            except Exception:
                label_crop = img

        ch, cw = label_crop.shape[:2]
        
        # Untere 65% als OCR-Zone
        ocr_zone = label_crop[int(ch * 0.35):, :]
        if ocr_zone.size == 0:
            continue

        if len(ocr_zone.shape) == 3:
            gray_crop = cv2.cvtColor(ocr_zone, cv2.COLOR_BGR2GRAY)
        else:
            gray_crop = ocr_zone

        resized = cv2.resize(gray_crop, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
        real_crops.append((resized, gt_code))

    logger.info(f"Extracted {len(real_crops)} valid real text crops.")
    return real_crops


class CombinedCharDataset(Dataset):
    """Dataset mit synthetischen und echten Bildern (Pre-rendered for high speed)."""
    
    def __init__(self, real_crops: list[tuple[np.ndarray, str]], codes: list[str], fonts: list[str],
                 augs_per_code: int = 2, real_oversample: int = 30):
        self.samples = []
        
        # 1. Echte Bilder augmentiert hinzufügen
        logger.info(f"Preparing {len(real_crops)} real crops x {real_oversample} oversampling...")
        for img_raw, code in real_crops:
            for _ in range(real_oversample):
                aug_img = augment_text_image(img_raw.copy())
                self.samples.append((aug_img, code))

        # 2. Synthetische Bilder vorab im Speicher rendern
        logger.info(f"Pre-rendering synthetic dataset ({len(codes)} codes x {augs_per_code} augs)...")
        fonts_to_use = fonts if fonts else [None]
        for code in codes:
            for _ in range(augs_per_code):
                font_path = random.choice(fonts_to_use)
                font_size = random.randint(18, 26)
                if PIL_AVAILABLE and font_path is not None:
                    img = render_code_pil(code, font_path=font_path, font_size=font_size)
                else:
                    img = render_code_cv2(code, font_scale=random.uniform(0.6, 1.0))
                img = augment_text_image(img)
                self.samples.append((img, code))

        random.shuffle(self.samples)
        logger.info(f"Dataset ready: {len(self.samples)} pre-rendered training images.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, code = self.samples[idx]
        prefix_idx = PREFIXES.index(code[0])
        digit1 = int(code[1])
        digit2 = int(code[2])
        digit3 = int(code[3])

        img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        return img_tensor, prefix_idx, digit1, digit2, digit3


def train_pacc_real(epochs: int = 25, lr: float = 5e-4):
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting Fast PACC Fine-Tuning on {device} ({epochs} epochs, lr={lr})...")

    gt_map = load_real_gt_mappings()
    real_crops = extract_real_crops(gt_map)
    
    if not real_crops:
        logger.error("No real crops found in training_data/! Aborting.")
        return

    all_codes = generate_all_codes()
    fonts = _find_system_fonts()

    # 172 * 20 = 3,440 echte Bilder + 4,000 * 2 = 8,000 synthetische Bilder = ~11.4k Pro-Epoch
    dataset = CombinedCharDataset(real_crops, all_codes, fonts, augs_per_code=2, real_oversample=20)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)

    checkpoint_path = os.path.join(MODEL_DIR, "char_classifier_best.pt")
    model = create_model()

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state)
        logger.info(f"Loaded existing checkpoint from {checkpoint_path}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        correct_full = 0

        for batch_img, batch_p, batch_d1, batch_d2, batch_d3 in loader:
            batch_img = batch_img.to(device)
            targets = [
                batch_p.to(device),
                batch_d1.to(device),
                batch_d2.to(device),
                batch_d3.to(device),
            ]

            optimizer.zero_grad()
            outputs = model(batch_img)

            loss = sum(criterion(out, tgt) for out, tgt in zip(outputs, targets))
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_img.size(0)
            total_samples += batch_img.size(0)

            all_correct = torch.ones(batch_img.size(0), dtype=torch.bool, device=device)
            for out, tgt in zip(outputs, targets):
                pred = out.argmax(dim=1)
                all_correct &= (pred == tgt)
            correct_full += all_correct.sum().item()

        avg_loss = total_loss / total_samples
        full_acc = correct_full / total_samples

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            logger.info(f"Epoch {epoch:2d}/{epochs} | Loss: {avg_loss:.4f} | Full Acc: {full_acc:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_full_accuracy': full_acc,
            }, checkpoint_path)

    logger.info(f"Fine-tuning complete! Best loss: {best_loss:.4f}")
    logger.info("Exporting to ONNX...")

    # ONNX Export aufrufen
    from export_char_classifier_onnx import export_to_onnx
    onnx_path = os.path.join(MODEL_DIR, "char_classifier.onnx")
    export_to_onnx(checkpoint_path, onnx_path, verify=True)
    logger.info(f"New ONNX model exported to {onnx_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PACC Real Fine-Tuning")
    parser.add_argument("--epochs", type=int, default=25, help="Anzahl der Trainings-Epochen")
    parser.add_argument("--lr", type=float, default=5e-4, help="Lernrate")
    args = parser.parse_args()

    train_pacc_real(epochs=args.epochs, lr=args.lr)
