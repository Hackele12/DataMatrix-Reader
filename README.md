# DataDetector: Headless Industrial 2D DataMatrix & OCR Scanning Pipeline

DataDetector is a high-availability, hybrid multi-modal scanning application designed for 24/7 industrial production environments. By combining state-of-the-art Deep Learning object detection (YOLO) with concurrent dual-channel barcode and optical character recognition (OCR) validation, DataDetector achieves close-to-zero false-positive rates even on damaged, dirty, or occluded physical tags.

The system is optimized to run as a headless background daemon, communicating with industrial PLC (SPS) controllers or network gateways (such as Moxa NPort) via a raw TCP socket connection.

---

## Key Features

- **Headless TCP/IP Interface**: Listen on port `9500` for trigger signals (`+`), run the detection and scan pipeline, and respond with the output followed by a Carriage Return (`\r`).
- **Object Detection Pre-localization (YOLOv10)**: Fast bounding box localization of label tags to extract the exact Region of Interest (ROI) with safety padding.
- **Multi-Modal Dual-Validation**:
  - **DataMatrix (2D) Decoder**: Direct decoding via `pylibdmtx` with parallel image pre-processing fallbacks (Unsharp Masking, Otsu & Adaptive Thresholding, Morphology, and Cubic Upscaling).
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

- `vision_app_v4_network.py`: Headless TCP socket server daemon, managing the camera frame grabber and orchestrating the YOLO detection and scanner validation.
- `scanner.py`: Core processing pipeline running the concurrent DataMatrix reconstruction and EasyOCR validation.
- `train.py` & `Start_Training.bat`: Training pipeline wrapper to retrain the YOLOv10 model.

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
   Run the test client script to verify connection and communication:
   ```powershell
   .\.venv\Scripts\python.exe test_trigger.py
   ```

---

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. 

Since this project links and depends on software licensed under the AGPL-3.0 (such as the Ultralytics YOLO packages), it is distributed publicly and open-source under the AGPL-3.0 terms to ensure compliance. You are free to copy, modify, and distribute this software, provided that any modified versions run as network services make their source code available to the network users.
