"""
scanner — Dual-Validation Scanner für Horden-Etiketten (DataMatrix + Klarschrift).

Öffentliche Schnittstelle:
    scan_2class(frame, detections)  2-Klassen-Pipeline für YOLO-Detektionen (Produktion)
    scan_fast(frame, detections)    nur die schnellen DataMatrix-Stufen (Mehrbild-Auswertung im Feld)
    new_deadline(budget_s)          Deadline für das Zeitbudget eines Scans
    select_label_detections(dets)   DataMatrix-/Text-Detektion, die scan_2class() auswertet
    scan(frame)                     Gesamtbild-/Etikett-Scan ohne Detektionen
    scan_datamatrix(crop)           nur DataMatrix auf einem Crop
    scan_ocr(crop)                  nur Klarschrift auf einem Crop
    deskew_crop(image, box)         begradigter YOLO-Ausschnitt mit Quiet Zone

Module:
    config       Feature-Schalter, Code-Format, Pfade
    pipeline     Orchestrierung der Scans (Fast-Paths, Gegenprobe, Gamma-Fallback)
    fusion       Zusammenführung von OCR, DataMatrix und Referenzbild; Gitter-Rekonstruktion
    dmx_decoder  zxing-cpp / pylibdmtx Dekodierung, Dot-Peen-Varianten, Rahmen-Rekonstruktion
    dmx_module_reader  Modul-Decoder für gesprenkelte/gescherte/unscharfe Codes (Rahmen-Fit, RS, Codebuch)
    dmx_grid     10x10-Gittergeometrie, Sampling und Abgleich mit Referenzgittern
    dmx_codec    ECC200-Codec für Referenzgitter und Dekodierung (Reed-Solomon mit Löschungen)
    ocr          EasyOCR Multi-Pass und Auswertung von Teillesungen
    code_format  Horden-Code-Format und OCR-Verwechslungskorrektur
    ref_images   Pipeline 3: Hamming-Abgleich gegen generated_codes/
    horde_matching  Horden-DB-Abgleich und OCR-Konfusionskorrektur (Schalter)
    onnx_models  PACC-Zeichenklassifikator und MicroUNet-Binarisierer
    image_ops    Allgemeine Bildfilter und Zuschnitte
    results      Ergebnis-Dictionaries
"""

from .image_ops import deskew_crop
from .pipeline import new_deadline, scan, scan_2class, scan_datamatrix, scan_fast, scan_ocr, select_label_detections

__all__ = ["scan", "scan_2class", "scan_fast", "new_deadline", "scan_datamatrix", "scan_ocr",
           "select_label_detections", "deskew_crop"]
