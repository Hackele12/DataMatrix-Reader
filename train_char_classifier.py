"""
train_char_classifier.py — Training des Position-Aware Char Classifier (PACC).

Generiert synthetische Trainingsbilder für alle 4.000 Horden-Codes (A000-W999)
mit verschiedenen Fonts, Augmentierungen und Degradierungen.

Nutzung:
    .venv\\Scripts\\python.exe train_char_classifier.py
    .venv\\Scripts\\python.exe train_char_classifier.py --epochs 50 --augmentations 20
    .venv\\Scripts\\python.exe train_char_classifier.py --device gpu
"""

import os
import sys
import random
import logging
import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("CharClassifierTraining")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_DIR, "models")

# --- Code-Generierung ---
PREFIXES = ['A', 'B', 'P', 'W']
IMG_WIDTH = 128
IMG_HEIGHT = 32

# Verfügbare System-Fonts (Windows)
FONT_CANDIDATES = [
    "arial.ttf",
    "arialbd.ttf",
    "cour.ttf",
    "courbd.ttf",
    "consola.ttf",
    "consolab.ttf",
    "calibri.ttf",
    "calibrib.ttf",
    "verdana.ttf",
    "verdanab.ttf",
    "tahoma.ttf",
    "tahomabd.ttf",
    "trebuc.ttf",
    "trebucbd.ttf",
    "segoeui.ttf",
    "segoeuib.ttf",
]


def _find_system_fonts() -> list[str]:
    """Findet verfügbare System-Fonts auf Windows."""
    fonts = []
    font_dirs = [
        r"C:\Windows\Fonts",
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "Windows", "Fonts"),
    ]
    
    for font_dir in font_dirs:
        if not os.path.exists(font_dir):
            continue
        for font_name in FONT_CANDIDATES:
            font_path = os.path.join(font_dir, font_name)
            if os.path.exists(font_path):
                fonts.append(font_path)
    
    if not fonts:
        logger.warning("Keine System-Fonts gefunden, verwende PIL-Default-Font.")
    else:
        logger.info(f"Gefundene Fonts: {len(fonts)}")
    
    return fonts


def generate_all_codes() -> list[str]:
    """Generiert alle 4.000 gültigen Horden-Codes."""
    codes = []
    for prefix in PREFIXES:
        for num in range(1000):
            codes.append(f"{prefix}{num:03d}")
    return codes


# =============================================================================
# Synthetisches Rendering
# =============================================================================

def render_code_pil(code: str, font_path: str | None = None, font_size: int = 22,
                    img_size: tuple[int, int] = (IMG_WIDTH, IMG_HEIGHT)) -> np.ndarray:
    """
    Rendert einen 4-stelligen Code als Graustufenbild mit PIL.
    
    Args:
        code: Der zu rendernde Code (z.B. "W032")
        font_path: Pfad zur TTF-Font-Datei
        font_size: Schriftgröße in Pixeln
        img_size: (Breite, Höhe) des Ausgabebilds
        
    Returns:
        Graustufenbild als numpy array (uint8)
    """
    img = Image.new('L', img_size, 255)
    draw = ImageDraw.Draw(img)
    
    try:
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    
    # Text zentriert zeichnen
    bbox = draw.textbbox((0, 0), code, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (img_size[0] - text_w) // 2
    y = (img_size[1] - text_h) // 2
    
    draw.text((x, y), code, fill=0, font=font)
    
    return np.array(img)


def render_code_cv2(code: str, font_scale: float = 0.8,
                    img_size: tuple[int, int] = (IMG_WIDTH, IMG_HEIGHT)) -> np.ndarray:
    """
    Rendert einen Code als Graustufenbild mit OpenCV (Fallback ohne PIL-Fonts).
    """
    img = np.full((img_size[1], img_size[0]), 255, dtype=np.uint8)
    
    font = random.choice([
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
    ])
    thickness = random.choice([1, 2])
    
    text_size = cv2.getTextSize(code, font, font_scale, thickness)[0]
    x = (img_size[0] - text_size[0]) // 2
    y = (img_size[1] + text_size[1]) // 2
    
    cv2.putText(img, code, (x, y), font, font_scale, 0, thickness)
    
    return img


def augment_text_image(img: np.ndarray) -> np.ndarray:
    """
    Wendet realistische Augmentierungen auf ein Text-Rendering an.
    Simuliert die Bedingungen auf geätzten Plastikbehältern.
    """
    augmented = img.astype(np.float32)
    h, w = augmented.shape[:2]
    
    # 1. Kontrast-Reduktion (Verblassen)
    alpha = random.uniform(0.3, 0.9)
    augmented = augmented * alpha + 128.0 * (1.0 - alpha)
    
    # 2. Gauss-Blur (Unschärfe)
    if random.random() > 0.3:
        ksize = random.choice([3, 5])
        augmented = cv2.GaussianBlur(augmented, (ksize, ksize), random.uniform(0.3, 2.0))
    
    # 3. Additives Rauschen
    noise_sigma = random.uniform(5, 40)
    noise = np.random.normal(0, noise_sigma, augmented.shape).astype(np.float32)
    augmented = augmented + noise
    
    # 4. Leichte Rotation (±5°)
    if random.random() > 0.4:
        angle = random.uniform(-5, 5)
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        augmented = cv2.warpAffine(augmented, M, (w, h), borderValue=200)
    
    # 5. Leichte perspektivische Verzerrung
    if random.random() > 0.6:
        dx = random.uniform(-3, 3)
        dy = random.uniform(-2, 2)
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([
            [dx, dy], [w - dx, dy],
            [w + dx, h - dy], [-dx, h - dy]
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        augmented = cv2.warpPerspective(augmented, M, (w, h), borderValue=200)
    
    # 6. Morphologische Operationen (simuliert Druckfehler)
    if random.random() > 0.5:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        temp = np.clip(augmented, 0, 255).astype(np.uint8)
        if random.random() > 0.5:
            augmented = cv2.erode(temp, k, iterations=1).astype(np.float32)
        else:
            augmented = cv2.dilate(temp, k, iterations=1).astype(np.float32)
    
    # 7. Helligkeits-Gradient (Glanzstellen)
    if random.random() > 0.4:
        gradient = np.linspace(
            random.uniform(-20, 0), random.uniform(0, 20), w
        ).astype(np.float32)
        augmented = augmented + gradient[np.newaxis, :]
    
    # 8. JPEG-Artefakte simulieren
    if random.random() > 0.5:
        quality = random.randint(30, 80)
        temp = np.clip(augmented, 0, 255).astype(np.uint8)
        _, encoded = cv2.imencode('.jpg', temp, [cv2.IMWRITE_JPEG_QUALITY, quality])
        augmented = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    
    # 9. Invertieren (manchmal ist der Text hell auf dunkel)
    if random.random() > 0.8:
        augmented = 255.0 - augmented
    
    return np.clip(augmented, 0, 255).astype(np.uint8)


# =============================================================================
# Dataset
# =============================================================================

class CharClassifierDataset(Dataset):
    """
    Dataset für den Position-Aware Char Classifier.
    Rendert on-the-fly Trainingsbilder mit verschiedenen Fonts und Augmentierungen.
    """
    
    def __init__(self, codes: list[str], fonts: list[str],
                 augmentations_per_code: int = 12):
        self.codes = codes
        self.fonts = fonts if fonts else [None]
        self.augmentations_per_code = augmentations_per_code
        
        logger.info(f"Dataset: {len(codes)} Codes × {augmentations_per_code} Augs = {len(self)} Bilder")
    
    def __len__(self):
        return len(self.codes) * self.augmentations_per_code
    
    def __getitem__(self, idx):
        code_idx = idx // self.augmentations_per_code
        code = self.codes[code_idx]
        
        # Zufälligen Font wählen
        font_path = random.choice(self.fonts)
        font_size = random.randint(18, 26)
        
        # Bild rendern
        if PIL_AVAILABLE and font_path is not None:
            img = render_code_pil(code, font_path=font_path, font_size=font_size)
        else:
            img = render_code_cv2(code, font_scale=random.uniform(0.6, 1.0))
        
        # Augmentieren
        img = augment_text_image(img)
        
        # Labels extrahieren
        prefix_idx = PREFIXES.index(code[0])
        digit1 = int(code[1])
        digit2 = int(code[2])
        digit3 = int(code[3])
        
        # Tensor erstellen
        img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0  # (1, 32, 128)
        
        return img_tensor, prefix_idx, digit1, digit2, digit3


# =============================================================================
# Training
# =============================================================================

def train_model(
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    augmentations: int = 12,
    device_str: str = "cpu",
    val_split: float = 0.1,
):
    """Trainiert den Position-Aware Char Classifier."""
    
    sys.path.insert(0, MODEL_DIR)
    from char_classifier import create_model
    
    device = torch.device("cuda" if device_str in ("gpu", "cuda") and torch.cuda.is_available() else "cpu")
    logger.info(f"Training auf Device: {device}")
    
    # Codes und Fonts
    all_codes = generate_all_codes()
    fonts = _find_system_fonts()
    
    # Train/Val Split (auf Code-Ebene, nicht auf Bild-Ebene)
    random.shuffle(all_codes)
    val_size = max(1, int(len(all_codes) * val_split))
    val_codes = all_codes[:val_size]
    train_codes = all_codes[val_size:]
    
    logger.info(f"Codes: {len(train_codes)} Train, {len(val_codes)} Val")
    
    train_dataset = CharClassifierDataset(train_codes, fonts, augmentations_per_code=augmentations)
    val_dataset = CharClassifierDataset(val_codes, fonts, augmentations_per_code=max(3, augmentations // 3))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Modell
    model = create_model().to(device)
    logger.info(f"PACC Parameter: {model.count_parameters():,}")
    
    # Loss + Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training Loop
    best_val_acc = 0.0
    best_model_path = os.path.join(MODEL_DIR, "char_classifier_best.pt")
    
    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        train_correct = [0, 0, 0, 0]
        train_total = 0
        
        for batch_img, batch_p, batch_d1, batch_d2, batch_d3 in train_loader:
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
            
            train_loss += loss.item() * batch_img.size(0)
            train_total += batch_img.size(0)
            
            for i, (out, tgt) in enumerate(zip(outputs, targets)):
                pred = out.argmax(dim=1)
                train_correct[i] += (pred == tgt).sum().item()
        
        scheduler.step()
        
        train_loss /= train_total
        train_accs = [c / train_total for c in train_correct]
        
        # --- Validation ---
        model.eval()
        val_correct = [0, 0, 0, 0]
        val_total = 0
        val_full_correct = 0  # Alle 4 Positionen korrekt
        
        with torch.no_grad():
            for batch_img, batch_p, batch_d1, batch_d2, batch_d3 in val_loader:
                batch_img = batch_img.to(device)
                targets = [
                    batch_p.to(device),
                    batch_d1.to(device),
                    batch_d2.to(device),
                    batch_d3.to(device),
                ]
                
                outputs = model(batch_img)
                val_total += batch_img.size(0)
                
                all_correct = torch.ones(batch_img.size(0), dtype=torch.bool, device=device)
                for i, (out, tgt) in enumerate(zip(outputs, targets)):
                    pred = out.argmax(dim=1)
                    correct_mask = (pred == tgt)
                    val_correct[i] += correct_mask.sum().item()
                    all_correct &= correct_mask
                
                val_full_correct += all_correct.sum().item()
        
        val_accs = [c / val_total for c in val_correct]
        val_full_acc = val_full_correct / val_total
        
        # Logging
        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"Val Acc: P={val_accs[0]:.3f} D1={val_accs[1]:.3f} "
                f"D2={val_accs[2]:.3f} D3={val_accs[3]:.3f} | "
                f"Full: {val_full_acc:.3f}"
            )
        
        # Bestes Modell speichern
        if val_full_acc > best_val_acc:
            best_val_acc = val_full_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_full_accuracy': val_full_acc,
                'val_per_position': val_accs,
            }, best_model_path)
            if epoch % 10 == 0:
                logger.info(f"  → Bestes Modell gespeichert (Full Acc: {val_full_acc:.4f})")
    
    logger.info(f"\n{'=' * 60}")
    logger.info(f"TRAINING ABGESCHLOSSEN")
    logger.info(f"  Beste Full-Accuracy: {best_val_acc:.4f}")
    logger.info(f"  Modell gespeichert:  {best_model_path}")
    logger.info(f"\nNächster Schritt: ONNX Export")
    logger.info(f"  .venv\\Scripts\\python.exe export_char_classifier_onnx.py")
    logger.info(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Training des Position-Aware Char Classifier (PACC).")
    parser.add_argument("--epochs", type=int, default=50, help="Trainings-Epochen (Standard: 50)")
    parser.add_argument("--batch", type=int, default=64, help="Batch-Größe (Standard: 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate (Standard: 0.001)")
    parser.add_argument("--augmentations", type=int, default=12, help="Augmentierungen pro Code (Standard: 12)")
    parser.add_argument("--device", default="cpu", help="Training Device: 'cpu' oder 'gpu'")
    args = parser.parse_args()
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    train_model(
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        augmentations=args.augmentations,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
