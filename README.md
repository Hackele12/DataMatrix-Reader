# DataDetector: Headless Industrial 2D DataMatrix & OCR Scanning Pipeline

DataDetector is a high-availability, hybrid multi-modal scanning application designed for 24/7 industrial production environments. By combining state-of-the-art Deep Learning object detection (YOLO) with concurrent dual-channel barcode and optical character recognition (OCR) validation, DataDetector achieves close-to-zero false-positive rates even on damaged, dirty, or occluded physical tags.

The DataMatrixReader desktop app communicates with industrial PLC (SPS) controllers or network gateways (such as Moxa NPort) via one raw TCP socket connection per camera.

---

## Key Features

- **TCP/IP Interface per Camera**: While a camera's stream runs, its tab listens on its own port (camera 1: `9500`, camera 2: `9501`, …) for trigger signals (`+`), runs the detection and scan pipeline, and responds with `STX <code> CR LF EOT` (`\x02<code>\r\n\x04`).
- **Object Detection Pre-localization (YOLOv10)**: Fast bounding box localization of label tags to extract the exact Region of Interest (ROI) with safety padding.
- **Multi-Modal Dual-Validation**:
  - **DataMatrix (2D) Decoder**: Direct decoding via `zxing-cpp` (incl. dot-peen preprocessing).
  - **Module Reader** (`scanner/dmx_module_reader.py`) for codes that `zxing-cpp` cannot read (speckled laser marks, sheared or motion-blurred symbols): homography fit on the fixed frame (L-finder, timing pattern, quiet zone), module sampling on a speckle-robust feature image, Reed-Solomon decoding with erasures and correlation against all 4,000 valid codes. Results confirmed by Reed-Solomon count as *Verifiziert*; codebook-only results (*Modulabgleich*) must pass calibrated thresholds (`SOFT_*` in `scanner/config.py`).
  - **OCR (Optical Character Recognition)**: High-precision text reading using local EasyOCR models; when the module reader found the symbol, the text crop is rectified from the symbol geometry.
- **Time Budget**: Fast DataMatrix stages always run; slower fallbacks (OCR, full-frame scan, gamma OCR) only within `SCAN_TIME_BUDGET_S` (default 3 s, overridable via `scan_time_budget_s` in `config.json`).
- **Multi-Frame Reading in the Field**: If the first camera frame yields no Reed-Solomon-verified code, up to `scan_extra_frames` further frames are evaluated; a *Modulabgleich* result is confirmed by a second frame, conflicting frames yield no code (`soft_require_confirmation` makes confirmation mandatory).
- **Mathematical Matrix Reconstruction**:
  - Homography correction using perspective warping.
  - Cell sampling of the $10\times 10$ DataMatrix grid.
  - Galois Field $GF(256)$ arithmetic and Reed-Solomon Error Correction parity checking.
  - Template matching based on Hamming-distance voting against a preset list of valid codes.
- **Active Learning Loop**: Automatic collection and labeling (in YOLO format) of edge-case frames during runtime to continuously train and improve the detection model.
- **Industrial-Grade Reliability**: A watchdog in the DataMatrixReader console restarts the app after a crash, cameras that were streaming reconnect automatically, and Win32 error mode overrides suppress blocking system dialogs.

---

## Repository Structure

- `launcher.py` & `Start_DataDetector.bat`: Start window for DataMatrixReader, Benchmark and Log Analyzer; only the DataMatrixReader gets a console (with crash watchdog).
- `vision_app.py`: DataMatrixReader, one tab per camera with live view, scan and its own TCP port while the stream runs.
- `scanner/`: Core scan pipeline (DataMatrix decoding, OCR, fusion). Entry points `scanner.scan_2class()` and `scanner.scan()`; feature switches in `scanner/config.py`.
- `yolo_detector.py`: YOLO model selection and conversion of detections for the scanner.
- `scan_logger.py` & `log_analyzer_app.py`: Structured scan logging (JSONL, one subfolder per camera) and log analysis GUI.
- `horde_db.py`: Image database of verified scans (`hard_scans_cache/`).
- `benchmark_gui.py`: Benchmark and ground-truth tool in the same light design as the other apps (image gallery, zoomable ground-truth editor, cancellable runs, results table with CSV export, comparison with earlier runs from `benchmark_reports/`). Headless: `benchmark_gui.py --headless [--fail-on-regression]` writes a report, compares it with the previous one and can fail on regressions. The field test set runs with `--images field_data --gt ground_truth_field.json`.
- `train_v2.py` & `Start_Training.bat`: Training of the 2-class YOLOv10 model (`dataset_v2/`, prepared by `prepare_dataset_v2.py`, which labels images from the exact code corners found by `zxing-cpp` or the module reader and skips images without a code).
- `train_char_classifier.py`, `train_pacc_real.py`, `train_unet_binarizer.py`, `export_*_onnx.py`, `models/`: Training and ONNX export of the optional PACC and MicroUNet models.
- `generate_datamatrix.py` & `batch_generate_datamatrix.py`: Reference images in `generated_codes/`.
- `create_v24_release.py` & `setup.bat`: Offline release package and installation.

---

## Installation & Running

1. **Virtual Environment Setup**:
   Create a local virtual environment and install the required dependencies (OpenCV, PyTorch, EasyOCR, Ultralytics YOLOv10, pylibdmtx):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
2. **Start the Programs**:
   Open the start window and choose DataMatrixReader, Benchmark or Log Analyzer:
   ```cmd
   Start_DataDetector.bat
   ```
   In the DataMatrixReader, select a camera per tab and press *Start Stream*; this also opens the tab's TCP port.
3. **Simulate a Scan Trigger**:
   Send `+` to the camera port (e.g. `9500`) with any TCP client; the server answers with the decoded code or `ERROR`.

---

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. 

Since this project links and depends on software licensed under the AGPL-3.0 (such as the Ultralytics YOLO packages), it is distributed publicly and open-source under the AGPL-3.0 terms to ensure compliance. You are free to copy, modify, and distribute this software, provided that any modified versions run as network services make their source code available to the network users.
