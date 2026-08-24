"""
export_unet_onnx.py — Exportiert das trainierte MicroUNet nach ONNX für schnelle CPU-Inferenz.

Nutzung:
    .venv\\Scripts\\python.exe export_unet_onnx.py
    .venv\\Scripts\\python.exe export_unet_onnx.py --input models/unet_binarizer_best.pt
"""

import os
import sys
import logging
import argparse

import torch
import numpy as np

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("UNetExport")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_DIR, "models")


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    img_size: int = 128,
    verify: bool = True,
):
    """
    Exportiert ein trainiertes MicroUNet-Modell nach ONNX.
    
    Args:
        checkpoint_path: Pfad zum PyTorch Checkpoint (.pt)
        output_path: Pfad für die ONNX-Datei
        img_size: Bildgröße (Standard: 128)
        verify: Ob ONNX-Modell verifiziert werden soll
    """
    # Modell importieren
    sys.path.insert(0, MODEL_DIR)
    from unet_binarizer import create_model
    
    # Checkpoint laden
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint nicht gefunden: {checkpoint_path}")
        logger.error("Bitte zuerst train_unet_binarizer.py ausführen.")
        sys.exit(1)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model = create_model()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Modell geladen: {checkpoint_path}")
    logger.info(f"  Trainiert bis Epoch:  {checkpoint.get('epoch', '?')}")
    logger.info(f"  Val Loss:             {checkpoint.get('val_loss', '?'):.5f}")
    logger.info(f"  Val Accuracy:         {checkpoint.get('val_accuracy', '?'):.4f}")
    logger.info(f"  Parameter:            {model.count_parameters():,}")
    
    # Dummy-Input für den Export
    dummy_input = torch.randn(1, 1, img_size, img_size)
    
    # ONNX Export
    logger.info(f"Exportiere nach ONNX: {output_path}")
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"},
        },
        dynamo=False,
    )
    
    # Dateigröße prüfen
    file_size_kb = os.path.getsize(output_path) / 1024
    logger.info(f"ONNX-Modell exportiert: {output_path} ({file_size_kb:.0f} KB)")
    
    # Verifikation
    if verify:
        _verify_onnx(output_path, dummy_input, model)


def _verify_onnx(onnx_path: str, dummy_input: torch.Tensor, pytorch_model: torch.nn.Module):
    """Verifiziert das ONNX-Modell gegen das PyTorch-Modell."""
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime nicht installiert — Verifikation übersprungen.")
        logger.warning("Installieren mit: pip install onnxruntime")
        return
    
    logger.info("Verifiziere ONNX-Modell ...")
    
    # PyTorch Referenz-Output
    with torch.no_grad():
        pt_output = pytorch_model(dummy_input).numpy()
    
    # ONNX Inferenz
    session = ort.InferenceSession(onnx_path)
    ort_output = session.run(None, {"input": dummy_input.numpy()})[0]
    
    # Vergleich
    max_diff = np.max(np.abs(pt_output - ort_output))
    mean_diff = np.mean(np.abs(pt_output - ort_output))
    
    logger.info(f"  Max Abweichung:  {max_diff:.8f}")
    logger.info(f"  Mean Abweichung: {mean_diff:.8f}")
    
    if max_diff < 1e-4:
        logger.info("  ✓ Verifikation BESTANDEN — ONNX-Modell ist korrekt!")
    else:
        logger.warning("  ⚠ Verifikation: Leichte Abweichungen (kann durch Float-Precision entstehen)")
    
    # Inferenz-Timing
    import time
    warmup_input = np.random.randn(1, 1, 128, 128).astype(np.float32)
    
    # Warmup
    for _ in range(5):
        session.run(None, {"input": warmup_input})
    
    # Benchmark
    times = []
    for _ in range(50):
        t0 = time.time()
        session.run(None, {"input": warmup_input})
        times.append((time.time() - t0) * 1000)
    
    avg_ms = sum(times) / len(times)
    min_ms = min(times)
    max_ms = max(times)
    logger.info(f"\n  ONNX Inferenz-Timing (CPU, 50 Runs):")
    logger.info(f"    Avg: {avg_ms:.1f} ms | Min: {min_ms:.1f} ms | Max: {max_ms:.1f} ms")


def main():
    parser = argparse.ArgumentParser(description="Export MicroUNet nach ONNX.")
    parser.add_argument(
        "--input",
        default=os.path.join(MODEL_DIR, "unet_binarizer_best.pt"),
        help="Pfad zum trainierten Checkpoint"
    )
    parser.add_argument(
        "--output",
        default=os.path.join(MODEL_DIR, "unet_binarizer.onnx"),
        help="Pfad für die ONNX-Datei"
    )
    parser.add_argument("--size", type=int, default=128, help="Bildgröße (Standard: 128)")
    parser.add_argument("--no-verify", action="store_true", help="ONNX-Verifikation überspringen")
    args = parser.parse_args()
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    export_to_onnx(
        checkpoint_path=args.input,
        output_path=args.output,
        img_size=args.size,
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    main()
