"""
train_unet_binarizer.py — Training des MicroUNet zur DataMatrix-Binarisierung.

Generiert synthetische Trainingspaare aus den 4.000 DataMatrix-Referenzbildern:
  - Clean (Ground Truth): Sauberes Binärbild aus generated_codes/
  - Degraded (Input):     Realistisch degradierte Version (Fading, Blur, Noise, Erosion)

Nutzung:
    .venv\\Scripts\\python.exe train_unet_binarizer.py
    .venv\\Scripts\\python.exe train_unet_binarizer.py --epochs 100 --augmentations 15
    .venv\\Scripts\\python.exe train_unet_binarizer.py --device gpu
"""

import os
import sys
import random
import logging
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("UNetTraining")

# Projekt-Root
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATED_CODES_DIR = os.path.join(PROJECT_DIR, "generated_codes")
MODEL_OUTPUT_DIR = os.path.join(PROJECT_DIR, "models")

# Training-Konfiguration
IMG_SIZE = 128  # U-Net Input/Output Größe


# =============================================================================
# Synthetische Degradierung
# =============================================================================

def degrade_image(clean: np.ndarray) -> np.ndarray:
    """
    Wendet realistische Degradierungen auf ein sauberes DataMatrix-Bild an.
    Simuliert die Bedingungen auf Plastikbehältern nach dem Ätzbecken.
    
    Args:
        clean: Sauberes Binärbild (uint8, 0/255)
        
    Returns:
        Degradiertes Graustufenbild (uint8, 0-255)
    """
    degraded = clean.astype(np.float32)
    
    # 1. Kontrast reduzieren (simuliert Verbleichen/Ausbleichen)
    alpha = random.uniform(0.15, 0.65)
    degraded = degraded * alpha + 128.0 * (1.0 - alpha)
    
    # 2. Gauss-Blur (simuliert Defokussierung/Unschärfe)
    if random.random() > 0.2:
        ksize = random.choice([3, 5, 7])
        sigma = random.uniform(0.5, 3.0)
        degraded = cv2.GaussianBlur(degraded, (ksize, ksize), sigma)
    
    # 3. Additives Gauss-Rauschen (simuliert Ätzbecken-Textur)
    noise_sigma = random.uniform(10, 60)
    noise = np.random.normal(0, noise_sigma, degraded.shape).astype(np.float32)
    degraded = degraded + noise
    
    # 4. Salt-and-Pepper Rauschen (simuliert Ätzlöcher)
    if random.random() > 0.5:
        density = random.uniform(0.01, 0.08)
        mask = np.random.random(degraded.shape)
        degraded[mask < density / 2] = 0
        degraded[mask > 1 - density / 2] = 255
    
    # 5. Morphologische Erosion (simuliert Modullöcher / dünner werdende Module)
    if random.random() > 0.4:
        k_size = random.choice([2, 3])
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        degraded_uint8 = np.clip(degraded, 0, 255).astype(np.uint8)
        degraded = cv2.erode(degraded_uint8, kernel, iterations=1).astype(np.float32)
    
    # 6. Morphologische Dilation (simuliert verschmolzene Module)
    if random.random() > 0.6:
        k_size = random.choice([2, 3])
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        degraded_uint8 = np.clip(degraded, 0, 255).astype(np.uint8)
        degraded = cv2.dilate(degraded_uint8, kernel, iterations=1).astype(np.float32)
    
    # 7. Lokale Helligkeits-Inhomogenität (Sinus-Gradient → Glanzstellen)
    if random.random() > 0.3:
        h, w = degraded.shape[:2]
        freq_x = random.uniform(0.5, 3.0)
        freq_y = random.uniform(0.5, 3.0)
        phase_x = random.uniform(0, 2 * np.pi)
        phase_y = random.uniform(0, 2 * np.pi)
        
        x_grid = np.linspace(0, freq_x * np.pi, w)
        y_grid = np.linspace(0, freq_y * np.pi, h)
        xv, yv = np.meshgrid(x_grid, y_grid)
        gradient = (np.sin(xv + phase_x) + np.sin(yv + phase_y)) * random.uniform(15, 40)
        degraded = degraded + gradient
    
    # 8. Perspektivische Verzerrung (leicht)
    if random.random() > 0.5:
        h, w = degraded.shape[:2]
        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        
        dx = random.uniform(-0.08, 0.08) * w
        dy = random.uniform(-0.08, 0.08) * h
        dst_pts = np.float32([
            [random.uniform(0, abs(dx)), random.uniform(0, abs(dy))],
            [w - random.uniform(0, abs(dx)), random.uniform(0, abs(dy))],
            [w - random.uniform(0, abs(dx)), h - random.uniform(0, abs(dy))],
            [random.uniform(0, abs(dx)), h - random.uniform(0, abs(dy))]
        ])
        
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        degraded = cv2.warpPerspective(degraded, M, (w, h), borderValue=128)
    
    # 9. Gamma-Korrektur (simuliert Belichtungsunterschiede)
    if random.random() > 0.4:
        gamma = random.uniform(0.5, 2.0)
        degraded_norm = np.clip(degraded / 255.0, 0, 1)
        degraded = (np.power(degraded_norm, gamma) * 255.0)
    
    return np.clip(degraded, 0, 255).astype(np.uint8)


# =============================================================================
# Dataset
# =============================================================================

class DMXBinarizationDataset(Dataset):
    """
    Dataset für U-Net Training: Paare aus (degraded, clean) DataMatrix-Bildern.
    
    Lädt alle sauberen Referenzbilder aus generated_codes/ und erzeugt
    on-the-fly degradierte Versionen.
    """
    
    def __init__(self, codes_dir: str, img_size: int = 128, augmentations_per_image: int = 10):
        self.img_size = img_size
        self.augmentations_per_image = augmentations_per_image
        
        # Alle PNG-Dateien laden
        self.clean_images = []
        self.filenames = []
        
        if not os.path.exists(codes_dir):
            logger.error(f"generated_codes/ nicht gefunden: {codes_dir}")
            return
        
        png_files = sorted([f for f in os.listdir(codes_dir) if f.endswith('.png')])
        logger.info(f"Lade {len(png_files)} Referenzbilder aus {codes_dir} ...")
        
        for fname in png_files:
            img = cv2.imread(os.path.join(codes_dir, fname), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Auf Zielgröße skalieren
                img_resized = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
                # Sicherstellen, dass es wirklich binär ist
                _, img_binary = cv2.threshold(img_resized, 128, 255, cv2.THRESH_BINARY)
                self.clean_images.append(img_binary)
                self.filenames.append(fname)
        
        logger.info(f"  → {len(self.clean_images)} Bilder geladen")
        logger.info(f"  → {len(self) } Trainingspaare (×{augmentations_per_image} Augmentierungen)")
    
    def __len__(self):
        return len(self.clean_images) * self.augmentations_per_image
    
    def __getitem__(self, idx):
        img_idx = idx // self.augmentations_per_image
        clean = self.clean_images[img_idx]
        
        # Degradierte Version erzeugen
        degraded = degrade_image(clean)
        
        # In Tensor konvertieren: (1, H, W), float32, normiert auf [0, 1]
        input_tensor = torch.from_numpy(degraded).float().unsqueeze(0) / 255.0
        target_tensor = torch.from_numpy(clean).float().unsqueeze(0) / 255.0
        
        return input_tensor, target_tensor


# =============================================================================
# Training
# =============================================================================

def train_model(
    epochs: int = 80,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    augmentations: int = 10,
    device_str: str = "cpu",
    val_split: float = 0.1,
):
    """Trainiert das MicroUNet-Modell."""
    
    # Modell importieren
    sys.path.insert(0, os.path.join(PROJECT_DIR, "models"))
    from unet_binarizer import create_model
    
    device = torch.device("cuda" if device_str in ("gpu", "cuda") and torch.cuda.is_available() else "cpu")
    logger.info(f"Training auf Device: {device}")
    
    # Dataset erstellen
    full_dataset = DMXBinarizationDataset(
        codes_dir=GENERATED_CODES_DIR,
        img_size=IMG_SIZE,
        augmentations_per_image=augmentations,
    )
    
    if len(full_dataset) == 0:
        logger.error("Kein Trainingsdata vorhanden!")
        sys.exit(1)
    
    # Train/Val Split
    total = len(full_dataset)
    val_size = max(1, int(total * val_split))
    train_size = total - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    logger.info(f"Training: {train_size} Paare, Validation: {val_size} Paare")
    
    # Modell erstellen
    model = create_model().to(device)
    logger.info(f"MicroUNet Parameter: {model.count_parameters():,}")
    
    # Loss + Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Training Loop
    best_val_loss = float('inf')
    best_model_path = os.path.join(MODEL_OUTPUT_DIR, "unet_binarizer_best.pt")
    
    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for batch_input, batch_target in train_loader:
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)
            
            optimizer.zero_grad()
            output = model(batch_input)
            loss = criterion(output, batch_target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_input.size(0)
        
        train_loss /= train_size
        
        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        
        with torch.no_grad():
            for batch_input, batch_target in val_loader:
                batch_input = batch_input.to(device)
                batch_target = batch_target.to(device)
                
                output = model(batch_input)
                loss = criterion(output, batch_target)
                val_loss += loss.item() * batch_input.size(0)
                
                # Pixel-Accuracy berechnen
                predicted = (torch.sigmoid(output) > 0.5).float()
                correct = (predicted == batch_target).float().mean()
                val_accuracy += correct.item() * batch_input.size(0)
        
        val_loss /= val_size
        val_accuracy /= val_size
        
        scheduler.step(val_loss)
        
        # Logging
        if epoch % 5 == 0 or epoch == 1:
            lr_current = optimizer.param_groups[0]['lr']
            logger.info(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_loss:.5f} | "
                f"Val Loss: {val_loss:.5f} | "
                f"Val Acc: {val_accuracy:.4f} | "
                f"LR: {lr_current:.6f}"
            )
        
        # Bestes Modell speichern
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
            }, best_model_path)
            if epoch % 10 == 0:
                logger.info(f"  → Bestes Modell gespeichert (Val Loss: {val_loss:.5f})")
    
    logger.info(f"\n{'=' * 60}")
    logger.info(f"TRAINING ABGESCHLOSSEN")
    logger.info(f"  Bester Val Loss:     {best_val_loss:.5f}")
    logger.info(f"  Modell gespeichert:  {best_model_path}")
    logger.info(f"\nNächster Schritt: ONNX Export")
    logger.info(f"  .venv\\Scripts\\python.exe export_unet_onnx.py")
    logger.info(f"{'=' * 60}")
    
    return best_model_path


def main():
    parser = argparse.ArgumentParser(description="Training des MicroUNet zur DataMatrix-Binarisierung.")
    parser.add_argument("--epochs", type=int, default=80, help="Trainings-Epochen (Standard: 80)")
    parser.add_argument("--batch", type=int, default=32, help="Batch-Größe (Standard: 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate (Standard: 0.001)")
    parser.add_argument("--augmentations", type=int, default=10, help="Augmentierungen pro Referenzbild (Standard: 10)")
    parser.add_argument("--device", default="cpu", help="Training Device: 'cpu' oder 'gpu' (Standard: cpu)")
    args = parser.parse_args()
    
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    
    train_model(
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        augmentations=args.augmentations,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
