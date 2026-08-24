# Pipeline-Statusbericht: DataMatrix- & Klarschrift-Erkennung

> Stand: 18.08.2026 | Basierend auf [scanner.py](file:///c:/Users/kremidas/Documents/DataDetector/scanner.py), [vision_app.py](file:///c:/Users/kremidas/Documents/DataDetector/vision_app.py), [benchmark_run_step42.json](file:///c:/Users/kremidas/Documents/DataDetector/benchmark_run_step42.json)

---

## Aktueller Benchmark (Step 42)

| Metrik | Wert |
|---|---|
| **Testbilder** | 66 |
| **Success-Rate** | 63.6% (42/66) |
| **Accuracy (vs. Ground Truth)** | 54.5% (36/66) |
| **∅ Dauer pro Scan** | ~20.3 s |
| **Methoden-Verteilung** | Verifiziert: 26, OCR: 12, Rekonstruiert: 4 |
| **Fehlerkategorien** | `DMTX_AND_OCR_FAILED`: 17, `RECONSTRUCTION_REJECTED`: 7 |
| **OCR-Teillesungen** | 4/4: 42, 3/4: 7, 0/4: 17 |

---

## 1. YOLOv10-Einsatz

### ✅ Bereits implementiert

- **YOLOv10n** als Etikett-Detektor (Klasse `0: Horde`), **nicht** als separater DMX- oder Klarschrift-Detektor
- Schneidet das **gesamte Etikett** (Plastikbehälter-Aufkleber) als eine Bounding Box aus – DataMatrix und Klarschrift werden **gemeinsam** im Crop behandelt
- YOLO → [deskew_crop()](file:///c:/Users/kremidas/Documents/DataDetector/scanner.py#L3856-L3978) (Begradigung via Canny + `minAreaRect` + `warpAffine`) → dann [scan()](file:///c:/Users/kremidas/Documents/DataDetector/scanner.py#L3744-L3853)

```python
# vision_app.py, Zeile 1274-1286:
if self.model is not None:
    with self._model_lock:
        results = self.model.predict(scan_snapshot, conf=0.15, verbose=False)
    boxes = results[0].boxes
    if len(boxes) > 0:
        best_box = boxes[0]
        detection_conf = float(best_box.conf[0])
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        detection_box = (x1, y1, x2, y2)
        scan_frame = scanner.deskew_crop(scan_snapshot, detection_box, padding=60)
```

- Lazy-Loading: YOLO wird erst beim ersten Stream-Start geladen (kein Startup-Delay)
- Trainiert via [train.py](file:///c:/Users/kremidas/Documents/DataDetector/train.py) mit Active Learning Pipeline (`auto_training_data/`)
- Modellpfad: `runs/detect/training_runs/horde_model/weights/best.pt` (Fallback: `yolov10n.pt`)
- YOLO Inferenz-Throttling: max 2× pro Sekunde in der Display-Loop

### 🔲 Noch nicht implementiert / offen

- **Kein getrennter Zuschnitt** für DataMatrix vs. Klarschrift – YOLO hat nur eine Klasse (`Horde`), nicht z.B. `dmx_code` und `klarschrift` separat
- Kein Segmentierungsmodell für die exakte Kontur des Codes (nur Bounding Box)

---

## 2. Bildvorverarbeitung

### ✅ Bereits implementiert (11+ Varianten)

Die Pipeline erzeugt **bis zu 11 Vorverarbeitungsvarianten** und testet sie systematisch. Hier eine Übersicht aller Filter in [scanner.py](file:///c:/Users/kremidas/Documents/DataDetector/scanner.py):

| # | Variante | Funktion | Technik |
|---|---|---|---|
| 1 | Standard | [_preprocess_for_ocr()](file:///c:/Users/kremidas/Documents/DataDetector/scanner.py#L197-L212) | Grayscale → Unsharp Mask → CLAHE (clipLimit=3.0) |
| 2 | Aggressiv | [_preprocess_ocr_variants()](file:///c:/Users/kremidas/Documents/DataDetector/scanner.py#L215-L287) | CLAHE clipLimit=8.0 |
| 3 | Faded Contrast Boost | [_preprocess_faded_contrast()](file:///c:/Users/kremidas/Documents/DataDetector/scanner.py#L305-L330) | Perzentil-Stretching (2%-98%) + TopHat + BottomHat |
| 4 | Etch Denoise | [_preprocess_etch_denoise()](file:///c:/Users/kremidas/Documents/DataDetector/scanner.py#L333-L366) | Stretching (1%-99%) + Ellipse-Morphologie (11×11) + **MORPH_CLOSE (3×3)** + Schärfung |
| 5 | Ridge Enhancement | [_preprocess_ridge_enhancement()](file:///c:/Users/kremidas/Documents/DataDetector/scanner.py#L369-L395) | Sobel-Gradient (8-Bit) + Kantenboost |
| 6 | Sauvola W11 | [_preprocess_sauvola()](file:///c:/Users/kremidas/Documents/DataDetector/scanner.py#L400-L418) | Lokaler Schwellenwert `T = mean * (1 + k * (std/R - 1))`, W=11, k=0.2 |
| 7 | Sauvola W15 | (gleiche Funktion) | Window=15 (Standard) |
| 8 | Niblack | [_preprocess_niblack()](file:///c:/Users/kremidas/Documents/DataDetector/scanner.py#L421-L438) | Lokaler Schwellenwert `T = mean + k * std`, W=21, k=-0.2 |
| 9 | TopHat | [_preprocess_tophat()](file:///c:/Users/kremidas/Documents/DataDetector/scanner.py#L290-L302) | Morphologischer TopHat (15×15) zur Glanz-Neutralisierung |
| 10 | Extrem | In Varianten-Generator | CLAHE clipLimit=15.0 |
| 11 | Invertiert | In Varianten-Generator | `bitwise_not` + CLAHE 8.0 |
| 12 | Binär-Otsu | In Varianten-Generator | Globaler Otsu-Schwellenwert |
| 13 | Adaptiv + MorphClose | In Varianten-Generator | Adaptiver Gauss-Schwellenwert + MORPH_CLOSE (2×2) |

**Morphologische Filter gegen weiße Lücken:**
- **MORPH_CLOSE (3×3)** nach dem Warping in [_warp_and_sample()](file:///c:/Users/kremidas/Documents/DataDetector/scanner.py#L938-L940): Schließt Ätzbecken-Löcher in schwarzen Modulen
- **Directional Close** in [_repair_l_finder()](file:///c:/Users/kremidas/Documents/DataDetector/scanner.py#L702-L717): Horizontal (7×1) + Vertikal (1×7) zur L-Finder-Reparatur
- **MORPH_CLOSE (2×2)** für OCR-Buchstabenstriche

```python
# Aktueller Preprocessing-Ausschnitt (Faded Contrast Boost):
def _preprocess_faded_contrast(image):
    # 1. Perzentil-Stretching
    p_low, p_high = np.percentile(gray, (2, 98))
    stretched = np.clip((gray - p_low) * (255.0 / (p_high - p_low)), 0, 255)
    # 2. TopHat + BottomHat Boost
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    tophat = cv2.morphologyEx(stretched, cv2.MORPH_TOPHAT, kernel)
    bottomhat = cv2.morphologyEx(stretched, cv2.MORPH_BLACKHAT, kernel)
    enhanced = cv2.add(stretched, tophat)
    enhanced = cv2.subtract(enhanced, bottomhat)
    return _sharpen(enhanced)
```

### 🔲 Noch nicht implementiert

- Kein **Deconvolution-Filter** (Wiener/Richardson-Lucy) gegen Unschärfe
- Kein **White-Balance-Korrektur** für Farbabweichungen durch die Ätzchemie
- Kein **Retinex-basierter** Algorithmus (z.B. MSRCR) für ungleichmäßige Beleuchtung

---

## 3. Decoder-Algorithmus (DataMatrix-Lesung)

### ✅ Bereits implementiert (4 Decoder-Stufen)

Die Pipeline nutzt eine **4-stufige Kaskade** mit zunehmendem Aufwand:

#### Stufe 0: zxing-cpp Fast-Path (<5 ms)
[_try_zxing_dmtx()](file:///c:/Users/kremidas/Documents/DataDetector/scanner.py#L89-L135) — C++20-basierter High-Speed-Decoder mit 3D-Homographie:

```python
def _try_zxing_dmtx(image):
    binarizers = [
        zxingcpp.Binarizer.LocalAverage,
        zxingcpp.Binarizer.GlobalHistogram,
        zxingcpp.Binarizer.FixedThreshold,
    ]
    for binarizer in binarizers:
        res = zxingcpp.read_barcode(
            gray, formats=zxingcpp.BarcodeFormat.DataMatrix,
            try_rotate=True, try_downscale=True, try_invert=True,
            binarizer=binarizer,
        )
        if res and res.valid:
            return _clean_to_4chars(res.text)
```

Wird **10× hintereinander** mit verschiedenen Preprocessing-Varianten versucht: Raw → CLAHE 4.0 → CLAHE 10.0 → TopHat → FadedBoost → EtchDenoise → RidgeBoost → Sauvola W11 → Niblack → Sauvola W15.

#### Stufe 1: pylibdmtx Fallback (50-400 ms)
[_read_datamatrix()](file:///c:/Users/kremidas/Documents/DataDetector/scanner.py#L2070-L2372) — libdmtx (C-Bibliothek) via Python-Wrapper:
- Direktscan auf Graustufenbild
- Upscale bei kleinen Bildern (<600px)
- CLAHE-verstärkter Scan (clipLimit 4/8/15)
- Kontur-basiertes ROI-Cropping → Kandidaten-Prüfung
- **10 parallele Filter-Varianten** via ThreadPoolExecutor als letzter Fallback

#### Stufe 2: Eigenes Bit-Parsing (Grid-Rekonstruktion)
[_try_reconstruct()](file:///c:/Users/kremidas/Documents/DataDetector/scanner.py#L1824-L2043) — Komplett eigener Algorithmus:

1. **Gitter-Extraktion:** [_extract_observed_grid()](file:///c:/Users/kremidas/Documents/DataDetector/scanner.py#L1612-L1710) — Konturfindung → `minAreaRect` → [_orient_corners()](file:///c:/Users/kremidas/Documents/DataDetector/scanner.py#L794-L880) (testet alle 4 Rotationen, bewertet L-Finder + Timing-Pattern) → [_refine_corners_ransac()](file:///c:/Users/kremidas/Documents/DataDetector/scanner.py#L735-L791)
2. **Perspektivische Entzerrung:** `getPerspectiveTransform` → `warpPerspective` auf 200×200px
3. **Zell-Sampling:** 10×10 Grid, Median über 6×6-Region pro Zelle → Binärmatrix
4. **Vektorisierter Abgleich** gegen **4.000 vorberechnete Referenz-Grids** (alle Codes A000-W999): [_get_precomputed_4000_grid_matrix()](file:///c:/Users/kremidas/Documents/DataDetector/scanner.py#L1380-L1401)
5. **Reed-Solomon GF(256):** [_compute_rs_ecc()](file:///c:/Users/kremidas/Documents/DataDetector/scanner.py#L1176-L1203) + **Utah-Placement** nach ISO/IEC 16022: [_get_placement_map()](file:///c:/Users/kremidas/Documents/DataDetector/scanner.py#L1244-L1325)
6. **Soft-Matching:** [_warp_and_sample_soft()](file:///c:/Users/kremidas/Documents/DataDetector/scanner.py#L966-L1009) extrahiert Float-Intensitäten (0.0-1.0) statt harter Binärisierung, dann Score = 1 - mean(|P - R|)

#### Stufe 3: Referenzbild-Pipeline (Hamming-Distanz)
[_scan_reference_image_pipeline()](file:///c:/Users/kremidas/Documents/DataDetector/scanner.py#L3077-L3174) — Vergleicht das Kamerabild gegen **~26.000 vorgenerierte PNG-Referenzbilder** in `generated_codes/`:
- Skalierung auf 100×100, Otsu-Binarisierung
- Gepackte Bits (`np.packbits`) + XOR-Popcount → 8× schneller
- Alle 4 Rotationsvarianten + Fullframe-Fallback

#### Zusätzlich: Template-Matching
[_template_match_candidates()](file:///c:/Users/kremidas/Documents/DataDetector/scanner.py#L1454-L1544) — Multi-Scale normalisierte Kreuzkorrelation (`TM_CCOEFF_NORMED`) für extrem beschädigte Bilder.

### ⚠️ Getestet, hat aber oft nicht funktioniert

- **pylibdmtx bei ausgeblichenen Codes:** Timeout-basiert (250-400ms), liefert auf verblassten Plastikbehältern in 17/66 Fällen kein Ergebnis
- **Grid-Rekonstruktion bei ausgeblichenen Codes:** In 7/66 Fällen `RECONSTRUCTION_REJECTED` — der Score liegt über dem Minimum, aber die Margin zum zweitbesten Code ist zu gering (Verwechslungsgefahr)

### 🔲 Noch nicht implementiert

- Kein **Deep-Learning-basierter DataMatrix-Decoder** (z.B. trainiertes CNN/ViT auf Zell-Level)
- Keine **Multi-Frame-Fusion** (Kombination mehrerer Kamerabilder für robusten Mehrheitsentscheid)

---

## 4. Klarschrift (OCR)

### ✅ Bereits implementiert — Wird aktiv ausgewertet!

Die Klarschrift unter dem DataMatrix-Code wird **vollständig** gelesen und ist eine **tragende Säule** der Pipeline.

**OCR-Engine:** [EasyOCR](https://github.com/JaidedAI/EasyOCR) (CRAFT-Detektor + ResNet/VGG Recognition)
- Sprachen: `['de', 'en']`, GPU deaktiviert (`gpu=False`)
- Offline-fähig mit lokalem Modellspeicher (`models/easyocr/`)
- `allowlist=ALLOWED_CHARS` (nur `A-Z0-9`)

**OCR-Pipeline:** [_read_ocr_with_status()](file:///c:/Users/kremidas/Documents/DataDetector/scanner.py#L2393-L2629)
1. Untere 65% des Bildes als OCR-Zone (`frame[int(h * 0.35):]`)
2. Quiet-Zone-Padding (20px weiß)
3. Downscale auf max. 800px Breite
4. **Fast-Mode** (2 Varianten: Standard + Aggressiv), bei Bedarf **Retry mit allen 11 Varianten**
5. Ergebnis-Aggregation: 4-Char-Kandidat → Normalisierung → Format-Validierung

**OCR-Zeichen-Normalisierung:** [_normalize_ocr_confusions()](file:///c:/Users/kremidas/Documents/DataDetector/scanner.py#L521-L557)
- Position 0 (Präfix): `8→B`, `4→A`, `V→W`, `D→B`, `R→P` etc. (26+ Regeln)
- Positionen 1-3 (Ziffern): `O→0`, `I→1`, `Z→2`, `S→5`, `B→8` etc.

**Dual-Validation:** OCR ↔ DataMatrix Cross-Check in [_merge_results()](file:///c:/Users/kremidas/Documents/DataDetector/scanner.py#L3325-L3741):
- Wenn beide übereinstimmen → **Verifiziert** (Konfidenz 1.0)
- Wenn nur OCR → Format-validierter **OCR-Fallback** (ab Konfidenz ≥0.40)
- Wenn OCR nur 3/4 Zeichen liest → **Cross-Validation** mit Grid: [_cross_validate_ocr_dmtx()](file:///c:/Users/kremidas/Documents/DataDetector/scanner.py#L1791-L1821)
- **Bayes-Fusion** bei Widerspruch: [_compute_joint_bayes_confidence()](file:///c:/Users/kremidas/Documents/DataDetector/scanner.py#L3290-L3322)

### ⚠️ Getestet, Schwächen bekannt

- **OCR liest bei ausgeblichenen Codes häufig falsche Ziffern** (z.B. W032 → W037, 2↔7 Verwechslung bei schwachem Kontrast) — Benchmark zeigt 6 Fälle mit `success=true` aber `is_match=false`
- **Nur 3 von 4 Zeichen erkannt** in 7/66 Fällen → Partial-Codes wie `W03?`, `?103`, `?200`

---

## 5. Typischer Fehlerpunkt bei ausgeblichenen Codes

### Analyse der 24 Fehlerfälle (17× `DMTX_AND_OCR_FAILED`, 7× `RECONSTRUCTION_REJECTED`)

> [!IMPORTANT]
> **Hauptengpass:** Die Pipeline scheitert primär an der **Binarisierung/Gitter-Rasterung** — nicht am Zuschnitt (YOLO funktioniert mit >95% Konfidenz) und nicht am Decoder selbst.

#### Fehlerkette bei ausgeblichenen Codes:

```
Kamerabild (verbleicht, kontrastarm)
    │
    ├─ YOLO Crop ✅ (funktioniert fast immer, Konf. >0.94)
    │
    ├─ zxing-cpp ❌ (findet kein L-Pattern/Timing-Pattern im verblassten Bild)
    │   └─ Trotz 10 Preprocessing-Varianten: Die Binarisierung erzeugt
    │      zu viel Rauschen oder verschmilzt Module
    │
    ├─ pylibdmtx ❌ (Timeout: kann L-Finder nicht lokalisieren)
    │
    ├─ Grid-Rekonstruktion ❌ (Hauptproblem)
    │   ├─ Konturfindung scheitert: MORPH_CLOSE (45/35/25/15/9) findet
    │   │   entweder die DMX-Region NICHT oder verschmilzt sie mit dem Etikettenrand
    │   ├─ _orient_corners() Score < 16/40: L-Finder und Timing-Pattern
    │   │   sind so verblasst, dass keine Rotation bestimmbar ist
    │   └─ Margin zum zweitbesten Code < 0.04: Selbst wenn ein Grid extrahiert
    │       wird, ist der Unterschied zwischen ähnlichen Codes zu gering
    │
    ├─ OCR ❌ (0 Zeichen erkannt) ODER ⚠️ (3/4 Zeichen, falsches Ergebnis)
    │   └─ EasyOCR kann bei stark verblasster Klarschrift keinen Text erkennen
    │      oder verwechselt visuell ähnliche Zeichen (2↔7, W↔V)
    │
    └─ RefImg Pipeline ❌ (Hamming-Distanz zu unspezifisch bei verblasstem Code)
```

> [!WARNING]
> **Die kritischste Stelle ist die Binarisierung vor der Grid-Extraktion** (Zeile 1654-1658 in `_extract_observed_grid`). Bei kontrastarmen Plastikbehältern aus dem Ätzbecken liefert weder Otsu noch der adaptive Schwellenwert ein sauberes Schwarz/Weiß-Bild — die Module sind zu schwach abgegrenzt.

---

## Zusammenfassung

| Bereich | Status |
|---|---|
| **YOLO Zuschnitt** | ✅ Funktioniert (>95% Konfidenz), schneidet **Etikett als Ganzes** aus |
| **Bildvorverarbeitung** | ✅ 11+ Varianten implementiert inkl. CLAHE, Sauvola, Niblack, TopHat, MORPH_CLOSE |
| **DataMatrix Decoder** | ✅ 4-stufig: zxing-cpp → pylibdmtx → Grid-Rekonstruktion (Reed-Solomon + 4000-Code Vektor-DB) → RefImg |
| **Klarschrift OCR** | ✅ EasyOCR aktiv, Dual-Validation mit DataMatrix, Fuzzy-Normalisierung, 3/4-Char Partial-Inferenz |
| **Hauptengpass** | ⚠️ **Binarisierung** bei ausgeblichenen Codes: L-Finder nicht erkennbar, Module verschmelzen |
| **Noch offen** | 🔲 Getrennter YOLO-Crop für DMX/Klarschrift, Deconvolution, Multi-Frame-Fusion, DL-basierter Decoder |
