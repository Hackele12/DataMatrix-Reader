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
