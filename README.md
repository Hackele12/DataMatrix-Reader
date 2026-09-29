# DataDetector: Headless Industrial 2D DataMatrix & OCR Scanning Pipeline

DataDetector is a high-availability, hybrid multi-modal scanning application designed for 24/7 industrial production environments. By combining state-of-the-art Deep Learning object detection (YOLO) with concurrent dual-channel barcode and optical character recognition (OCR) validation, DataDetector achieves close-to-zero false-positive rates even on damaged, dirty, or occluded physical tags.

The system is optimized to run as a headless background daemon, communicating with industrial PLC (SPS) controllers or network gateways (such as Moxa NPort) via a raw TCP socket connection.

---

## Key Features

- **Headless TCP/IP Interface**: Listen on port `9500` for trigger signals (`+`), run the detection and scan pipeline, and respond with `STX <code> CR LF EOT` (`\x02<code>\r\n\x04`).
- **Object Detection Pre-localization (YOLOv10)**: Fast bounding box localization of label tags to extract the exact Region of Interest (ROI) with safety padding.
- **Multi-Modal Dual-Validation**:
  - **DataMatrix (2D) Decoder**: Direct decoding via `zxing-cpp` (incl. dot-peen preprocessing) with `pylibdmtx` fallbacks (Unsharp Masking, Otsu & Adaptive Thresholding, Morphology, and Cubic Upscaling).
  - **OCR (Optical Character Recognition)**: High-precision text reading using local EasyOCR models.
- **Mathematical Matrix Reconstruction**:
  - Homography correction using perspective warping.
  - Cell sampling of the $10\times 10$ DataMatrix grid.
  - Galois Field $GF(256)$ arithmetic and Reed-Solomon Error Correction parity checking.
  - Template matching based on Hamming-distance voting against a preset list of valid codes.
- **Active Learning Loop**: Automatic collection and labeling (in YOLO format) of edge-case frames during runtime to continuously train and improve the detection model.
- **Industrial-Grade Reliability**: Dedicated Watchdog batch process to auto-restart the application on crash and Win32 error mode overrides to suppress blocking system dialogs.

---

## Repository Structure

- `vision_app_v4_network.py`: Headless multi-camera TCP socket server (`Start_DataDetector_v4_Network.bat`).
- `vision_app.py`: Desktop app for a single camera with live view and TCP trigger (`Start_DataDetector.bat`).
- `scanner/`: Core scan pipeline (DataMatrix decoding, OCR, fusion). Entry points `scanner.scan_2class()` and `scanner.scan()`; feature switches in `scanner/config.py`.
- `yolo_detector.py`: YOLO model selection and conversion of detections for the scanner.
- `scan_logger.py` & `log_analyzer_app.py`: Structured scan logging (JSONL) and log analysis GUI (`Start_LogAnalyzer.bat`).
- `horde_db.py`: Image database of verified scans (`hard_scans_cache/`).
- `benchmark_gui.py`: Benchmark and ground-truth tool (`Start_Benchmark.bat`, headless: `benchmark_gui.py --headless`).
- `train_v2.py` & `Start_Training.bat`: Training of the 2-class YOLOv10 model (`dataset_v2/`, prepared by `prepare_dataset_v2.py` / `relabel_2class.py`).
- `train_char_classifier.py`, `train_pacc_real.py`, `train_unet_binarizer.py`, `export_*_onnx.py`, `models/`: Training and ONNX export of the optional PACC and MicroUNet models.
- `generate_datamatrix.py` & `batch_generate_datamatrix.py`: Reference images in `generated_codes/`.
- `create_v23_release.py` & `setup.bat`: Offline release package and installation.

---

## Installation & Running

1. **Virtual Environment Setup**:
   Create a local virtual environment and install the required dependencies (OpenCV, PyTorch, EasyOCR, Ultralytics YOLOv10, pylibdmtx):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
2. **Start the Headless Server**:
   Launch the system via the watchdog process for continuous execution:
   ```cmd
   Start_DataDetector_v4_Network.bat
   ```
3. **Simulate a Scan Trigger**:
   Send `+` to the camera port (e.g. `9500`) with any TCP client; the server answers with the decoded code or `ERROR`.

---

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. 

Since this project links and depends on software licensed under the AGPL-3.0 (such as the Ultralytics YOLO packages), it is distributed publicly and open-source under the AGPL-3.0 terms to ensure compliance. You are free to copy, modify, and distribute this software, provided that any modified versions run as network services make their source code available to the network users.
