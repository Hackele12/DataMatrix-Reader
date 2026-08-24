"""
export_char_classifier_onnx.py — Exportiert den trainierten PACC nach ONNX.

Nutzung:
    .venv\\Scripts\\python.exe export_char_classifier_onnx.py
    .venv\\Scripts\\python.exe export_char_classifier_onnx.py --input models/char_classifier_best.pt
"""

import os
import sys
import time
import logging
import argparse

import torch
import numpy as np

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("PACCExport")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_DIR, "models")


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    verify: bool = True,
):
    """Exportiert den trainierten PACC nach ONNX."""
    
    sys.path.insert(0, MODEL_DIR)
    from char_classifier import create_model, PREFIX_CLASSES
    
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint nicht gefunden: {checkpoint_path}")
        logger.error("Bitte zuerst train_char_classifier.py ausführen.")
        sys.exit(1)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model = create_model()
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict)
    model.eval()
    
    logger.info(f"Modell geladen: {checkpoint_path}")
    logger.info(f"  Trainiert bis Epoch: {checkpoint.get('epoch', '?')}")
    logger.info(f"  Full Accuracy:       {checkpoint.get('val_full_accuracy', '?'):.4f}")
    logger.info(f"  Per-Position Acc:    {checkpoint.get('val_per_position', '?')}")
    logger.info(f"  Parameter:           {model.count_parameters():,}")
    
    # Dummy-Input: (1, 1, 32, 128) — Text-Crop
    dummy_input = torch.randn(1, 1, 32, 128)
    
    # ONNX Export
    # Da das Modell 4 Outputs hat, müssen wir sie benennen
    logger.info(f"Exportiere nach ONNX: {output_path}")
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["pos0_logits", "pos1_logits", "pos2_logits", "pos3_logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "pos0_logits": {0: "batch_size"},
            "pos1_logits": {0: "batch_size"},
            "pos2_logits": {0: "batch_size"},
            "pos3_logits": {0: "batch_size"},
        },
        dynamo=False,
    )
    
    file_size_kb = os.path.getsize(output_path) / 1024
    logger.info(f"ONNX-Modell exportiert: {output_path} ({file_size_kb:.0f} KB)")
    
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
    
    sys.path.insert(0, MODEL_DIR)
    from char_classifier import PREFIX_CLASSES
    
    logger.info("Verifiziere ONNX-Modell ...")
    
    # PyTorch Referenz
    with torch.no_grad():
        pt_outputs = pytorch_model(dummy_input)
        pt_arrays = [o.numpy() for o in pt_outputs]
    
    # ONNX Inferenz
    session = ort.InferenceSession(onnx_path)
    ort_outputs = session.run(None, {"input": dummy_input.numpy()})
    
    # Vergleich
    for i, (pt, ort_out) in enumerate(zip(pt_arrays, ort_outputs)):
        max_diff = np.max(np.abs(pt - ort_out))
        logger.info(f"  Position {i}: Max Abweichung = {max_diff:.8f}")
    
    total_max = max(np.max(np.abs(pt - ort_out)) for pt, ort_out in zip(pt_arrays, ort_outputs))
    if total_max < 1e-4:
        logger.info("  ✓ Verifikation BESTANDEN!")
    else:
        logger.warning("  ⚠ Leichte Abweichungen")
    
    # Inferenz-Test mit bekanntem Code
    logger.info("\nTest-Inferenzen:")
    warmup_input = np.random.randn(1, 1, 32, 128).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        session.run(None, {"input": warmup_input})
    
    # Benchmark
    times = []
    for _ in range(100):
        t0 = time.time()
        session.run(None, {"input": warmup_input})
        times.append((time.time() - t0) * 1000)
    
    avg_ms = sum(times) / len(times)
    min_ms = min(times)
    max_ms = max(times)
    
    logger.info(f"\n  ONNX Inferenz-Timing (CPU, 100 Runs):")
    logger.info(f"    Avg: {avg_ms:.2f} ms | Min: {min_ms:.2f} ms | Max: {max_ms:.2f} ms")
    
    # Decode Test-Output
    def softmax(x):
        e = np.exp(x - np.max(x))
        return e / e.sum()
    
    p0, p1, p2, p3 = ort_outputs
    probs = [softmax(p[0]) for p in [p0, p1, p2, p3]]
    indices = [np.argmax(p) for p in probs]
    confs = [probs[i][indices[i]] for i in range(4)]
    
    code = PREFIX_CLASSES[indices[0]] + str(indices[1]) + str(indices[2]) + str(indices[3])
    logger.info(f"\n  Test-Decode (Random Input): '{code}' (Confs: {[f'{c:.3f}' for c in confs]})")


def main():
    parser = argparse.ArgumentParser(description="Export PACC nach ONNX.")
    parser.add_argument(
        "--input",
        default=os.path.join(MODEL_DIR, "char_classifier_best.pt"),
        help="Pfad zum trainierten Checkpoint"
    )
    parser.add_argument(
        "--output",
        default=os.path.join(MODEL_DIR, "char_classifier.onnx"),
        help="Pfad für die ONNX-Datei"
    )
    parser.add_argument("--no-verify", action="store_true", help="Verifikation überspringen")
    args = parser.parse_args()
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    export_to_onnx(
        checkpoint_path=args.input,
        output_path=args.output,
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    main()
