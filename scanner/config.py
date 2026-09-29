"""Zentrale Konfiguration des Scanners: Feature-Schalter, Code-Format und Pfade."""

import os
import re
import sys

# --- Feature-Schalter (zur Laufzeit über scanner.config.<NAME> änderbar) ---

# zxing-cpp Vorverarbeitungs-Kaskade vor dem pylibdmtx-Fallback in dmx_decoder.read_datamatrix().
USE_ZXING_FASTPATH = True

# PACC-Char-Classifier: überkonfident und meist falsch (Benchmark 57/172) → erst nach Neutraining wieder aktivieren.
USE_PACC = False

# Horden-DB nur speichern, nicht zum Lesen/Korrigieren nutzen (Ganzbild-Abgleich erkennt die Szene, nicht den Code).
USE_HORDE_DB_MATCHING = False

# Gitter-Rekonstruktion gegen Referenzgitter: im Benchmark ohne Treffer, verursacht aber bis zu Minuten Rechenzeit.
USE_GRID_RECONSTRUCTION = False

# Modul-Decoder (dmx_module_reader) für gesprenkelte, gescherte oder unscharfe Codes nach dem zxing-Schnellpfad.
USE_MODULE_READER = True

# Annahme eines Modul-Decoder-Ergebnisses ohne Reed-Solomon-Bestätigung (Korrelation mit dem Codebuch).
# Kalibrierung: echte schwere Codes NCC ≥ 0.72 / Marge ≥ 0.24; Zufallsdaten mit perfektem Rahmen passieren
# 0.70/0.20 mit 4e-6 (1 Mio. Muster), fremde 10x10-Codes mit Horden-Präfix 0 von 200.000.
SOFT_MIN_NCC = 0.70
SOFT_MIN_MARGIN = 0.20
SOFT_MIN_FRAME_T = 3.0

# Alte DataMatrix-Kaskade (pylibdmtx, Gitter-Sampling, Referenzbild-Abgleich) nach den schnellen Stufen.
# Im Benchmark ohne eigenen Treffer, kostet bei unlesbaren Codes aber mehrere Sekunden.
USE_LEGACY_DMX_CASCADE = False

# Zeitbudget eines Scans: schnelle DMX-Stufen laufen immer, danach starten optionale Stufen
# (Gesamtbild-Fallback, Gamma-OCR) nur noch, solange Budget übrig ist. Die OCR-Gegenprobe läuft immer.
SCAN_TIME_BUDGET_S = 3.0

# --- Horden-Code-Format: [A|B|P|W] + 3 Ziffern ---
ALLOWED_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
REQUIRED_LENGTH = 4
VALID_PREFIXES = frozenset("ABPW")
HORDEN_PATTERN = re.compile(r"^[ABPW][0-9]{3}$")

# --- Pfade ---
# Projektordner (Elternordner des Pakets); enthält models/ mit den ONNX-Modellen.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Anwendungsordner für Laufzeitdaten (EasyOCR-Modelle, generated_codes/); bei PyInstaller der EXE-Ordner.
APP_DIR = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else PROJECT_DIR
