"""
Modul-Decoder für schwer lesbare 10x10-DataMatrix (gesprenkelte Lasermarkierung, Scherung, Bewegungsunschärfe).

Statt zu binarisieren und auf den zxing-Detektor zu hoffen, wird die bekannte Symbolstruktur genutzt:
Kandidatensuche → Homographie-Fit am festen Rahmen (L-Finder, Timing-Muster, Quiet Zone) →
Modul-Sampling auf einem sprenkelrobusten Merkmalsbild → Reed-Solomon (mit Löschungen) und
Korrelationsabgleich gegen alle gültigen Codes.

Stufen des Ergebnisses:
    "rs"          Reed-Solomon-Decode (≤ 2 Fehler) stimmt mit dem Codebuch-Abgleich überein
    "rs_erasure"  Reed-Solomon mit Löschungen (Restredundanz ≥ 1) + Codebuch-Abgleich
    "soft"        nur Codebuch-Abgleich mit kalibrierten Schwellen (config.SOFT_*)
"""

import logging
import time

import cv2
import numpy as np

from . import config
from .code_format import dmx_text_to_code
from .dmx_codec import (GRID_SIZE, codebook_data_patterns, codewords_from_dark_modules, data_module_codewords,
                        decode_ascii_codewords, rs_decode)
from .image_ops import rect_kernel

logger = logging.getLogger(__name__)

_N = GRID_SIZE
TARGET_MODULE_PX = 16.0          # Modulgröße, für die Merkmal und Fit parametriert sind
_NATIVE_MODULE_RANGE = (10.0, 24.0)  # innerhalb dieses Bereichs wird nicht umskaliert
_SEARCH_MODULE_PX = 16           # Modulgröße, für die die Blob-Suche parametriert ist (je Suchskala)
_BLOB_SIDE_PER_MODULE = 11.0     # Blob = 10 Module + Aufweitung durch Glättung und Closing
_SEARCH_SCALES = (1.0, 0.5)      # 0.5 findet Codes mit sehr großen Modulen
MAX_CANDIDATES = 6               # Blob-Kandidaten, die grob geprüft werden
MAX_FULL_FITS = 3                # davon vollständig gefittet (nach Grobprüfung sortiert)
_MIN_SCREEN_T = 2.0              # Kandidaten mit schwächerem Rahmen nach der Grobprüfung werden nicht gefittet
_EAGER_SCREEN_T = 3.3            # so deutlicher Rahmen wird sofort gefittet (echte Codes 3.7-4.4, Fremdes ≤ 2.2)

# Mindest-Rahmengüte je Stufe (t-Statistik dunkle vs. helle Rahmen-/Quiet-Zone-Module)
_RS_MIN_FRAME_T = 2.0
_ERASURE_MIN_FRAME_T = 2.5
_ERASURE_MIN_NCC = 0.50
_ERASURE_MIN_MARGIN = 0.10
_MAX_ERASURES = 3

# --- Rahmenmodell: L-Finder + Timing-Muster (36 Module) und Quiet Zone (44 Module, hell) ---
_frame = []
for _r in range(_N):
    for _c in range(_N):
        if _c == 0 or _r == _N - 1:
            _frame.append((_r, _c, True))
        elif _r == 0:
            _frame.append((_r, _c, _c % 2 == 0))
        elif _c == _N - 1:
            _frame.append((_r, _c, _r % 2 == 1))
for _r in range(-1, _N + 1):
    for _c in range(-1, _N + 1):
        if _r in (-1, _N) or _c in (-1, _N):
            _frame.append((_r, _c, False))
_FRAME_RC = np.array([(r, c) for r, c, _ in _frame], dtype=np.float32)
_FRAME_DARK = np.array([dark for _, _, dark in _frame], dtype=bool)
_FRAME_DARK_IDX = np.flatnonzero(_FRAME_DARK)
_FRAME_LIGHT_IDX = np.flatnonzero(~_FRAME_DARK)
_DATA_RC = np.array([(r, c) for r in range(1, _N - 1) for c in range(1, _N - 1)], dtype=np.float32)


def _sub_offsets(n: int, inner: float) -> np.ndarray:
    """n×n Abtastpunkte im inneren Anteil eines Moduls, als (x, y) in Moduleinheiten."""
    o = 0.5 + ((np.arange(n) + 0.5) / n - 0.5) * inner
    xx, yy = np.meshgrid(o, o)
    return np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)


def _grid_points(rc: np.ndarray, sub: np.ndarray) -> np.ndarray:
    """Homogene Gitterpunkte (x = Spalte, y = Zeile) aller Module × Abtastpunkte."""
    pts = (rc[:, None, ::-1] + sub[None, :, :]).reshape(-1, 2)
    return np.concatenate([pts, np.ones((len(pts), 1), np.float32)], axis=1)


_SUB_FIT = _sub_offsets(3, 0.6)
_SUB_DATA = _sub_offsets(5, 0.6)
_FIT_POINTS = _grid_points(_FRAME_RC, _SUB_FIT)
_DATA_POINTS = _grid_points(_DATA_RC, _SUB_DATA)


# --------------------------------------------------------------------------- #
#  Merkmalsbild, Sampling, Rahmen-Fit                                          #
# --------------------------------------------------------------------------- #

def speckle_feature(gray: np.ndarray) -> np.ndarray:
    """
    Helligkeit ohne Sprenkel: Erosion schließt die hellen Poren gelaserter Module, die lokale Streuung
    markiert texturierte (= markierte) Flächen zusätzlich als dunkel.
    """
    f = gray.astype(np.float32)
    eroded = cv2.GaussianBlur(cv2.erode(f, rect_kernel(3)), (0, 0), 1.0)
    mean = cv2.GaussianBlur(f, (0, 0), 2.0)
    sq_mean = cv2.GaussianBlur(f * f, (0, 0), 2.0)
    return eroded - np.sqrt(np.maximum(sq_mean - mean * mean, 0.0))


def _homographies(quads: np.ndarray) -> np.ndarray:
    """Projektive Abbildungen Gitter [0, N]² → Viereck (TL, TR, BR, BL) für K Vierecke (Heckbert, geschlossen)."""
    q = quads.astype(np.float64)
    x0, y0, x1, y1 = q[:, 0, 0], q[:, 0, 1], q[:, 1, 0], q[:, 1, 1]
    x2, y2, x3, y3 = q[:, 2, 0], q[:, 2, 1], q[:, 3, 0], q[:, 3, 1]
    sx, sy = x0 - x1 + x2 - x3, y0 - y1 + y2 - y3
    dx1, dx2, dy1, dy2 = x1 - x2, x3 - x2, y1 - y2, y3 - y2
    den = dx1 * dy2 - dx2 * dy1
    den = np.where(np.abs(den) < 1e-9, 1e-9, den)
    g = (sx * dy2 - dx2 * sy) / den
    h = (dx1 * sy - sx * dy1) / den
    homographies = np.empty((len(q), 3, 3))
    homographies[:, 0, 0] = (x1 - x0 + g * x1) / _N
    homographies[:, 0, 1] = (x3 - x0 + h * x3) / _N
    homographies[:, 0, 2] = x0
    homographies[:, 1, 0] = (y1 - y0 + g * y1) / _N
    homographies[:, 1, 1] = (y3 - y0 + h * y3) / _N
    homographies[:, 1, 2] = y0
    homographies[:, 2, 0] = g / _N
    homographies[:, 2, 1] = h / _N
    homographies[:, 2, 2] = 1.0
    return homographies


def _module_means(feature: np.ndarray, quads: np.ndarray, points: np.ndarray, n_sub: int) -> np.ndarray:
    """Mittlere Merkmalswerte je Modul für K Vierecke → (K, Module)."""
    proj = np.einsum("kij,pj->kpi", _homographies(quads), points)
    w = proj[..., 2]
    w = np.where(np.abs(w) < 1e-6, 1e-6, w)
    map_x = (proj[..., 0] / w).astype(np.float32)
    map_y = (proj[..., 1] / w).astype(np.float32)
    values = cv2.remap(feature, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return values.reshape(len(quads), -1, n_sub).mean(axis=2)


def _homography_single(quad: np.ndarray) -> np.ndarray:
    """Wie _homographies für ein Viereck, mit Python-Skalaren (deutlich schneller in der Fit-Schleife)."""
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = quad.tolist()
    sx, sy = x0 - x1 + x2 - x3, y0 - y1 + y2 - y3
    dx1, dx2, dy1, dy2 = x1 - x2, x3 - x2, y1 - y2, y3 - y2
    den = dx1 * dy2 - dx2 * dy1
    if abs(den) < 1e-9:
        den = 1e-9
    g = (sx * dy2 - dx2 * sy) / den
    h = (dx1 * sy - sx * dy1) / den
    return np.array([[(x1 - x0 + g * x1) / _N, (x3 - x0 + h * x3) / _N, x0],
                     [(y1 - y0 + g * y1) / _N, (y3 - y0 + h * y3) / _N, y0],
                     [g / _N, h / _N, 1.0]])


def _frame_t_single(feature: np.ndarray, quad: np.ndarray) -> float:
    """
    Trennschärfe dunkle vs. helle Rahmen-/Quiet-Zone-Module (t-Statistik mit gleich gewichteten Klassen;
    eine gepoolte Varianz würde das Viereck nach außen in den gleichmäßigen Hintergrund ziehen).
    """
    proj = _FIT_POINTS @ _homography_single(quad).T
    w = proj[:, 2]
    if np.any(np.abs(w) < 1e-6):
        return -1e9
    map_x = (proj[:, 0] / w).astype(np.float32).reshape(1, -1)
    map_y = (proj[:, 1] / w).astype(np.float32).reshape(1, -1)
    values = cv2.remap(feature, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    modules = values.reshape(-1, len(_SUB_FIT)).mean(axis=1)
    dark = modules[_FRAME_DARK_IDX]
    light = modules[_FRAME_LIGHT_IDX]
    return float((light.mean() - dark.mean()) / np.sqrt((light.var() + dark.var()) / 2 + 1e-6))


def _fit_frame(feature: np.ndarray, quad: np.ndarray, steps) -> tuple[np.ndarray, float]:
    """Koordinatensuche über die 8 Eckkoordinaten; jeder verbessernde Einzelschritt wird sofort übernommen."""
    q = quad.astype(np.float32).copy()
    best_t = _frame_t_single(feature, q)
    for step in steps:
        improved = True
        while improved:
            improved = False
            for corner in range(4):
                for axis in range(2):
                    for sign in (-1.0, 1.0):
                        q[corner, axis] += sign * step
                        t = _frame_t_single(feature, q)
                        if t > best_t:
                            best_t = t
                            improved = True
                        else:
                            q[corner, axis] -= sign * step
    return q, best_t


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Ecken im Uhrzeigersinn ab links oben (Bildkoordinaten)."""
    pts = np.float32(pts)
    center = pts.mean(axis=0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0]))]
    return np.roll(pts, -int(np.argmin(pts.sum(axis=1))), axis=0)


def _contour_quad(contour: np.ndarray) -> np.ndarray:
    """Viereck aus der konvexen Hülle einer Kontur (sonst minAreaRect)."""
    hull = cv2.convexHull(contour)
    for eps in (0.02, 0.03, 0.05, 0.08, 0.12):
        approx = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
        if len(approx) == 4:
            return _order_quad(approx.reshape(-1, 2))
    return _order_quad(cv2.boxPoints(cv2.minAreaRect(contour)))


def _initial_quad(roi: np.ndarray) -> np.ndarray | None:
    """Grobes Symbolviereck: größter dunkler Blob nach Glättung und Closing auf Modulgröße."""
    blur = cv2.GaussianBlur(roi, (0, 0), TARGET_MODULE_PX * 0.35)
    _, binary = cv2.threshold(blur.astype(np.uint8), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, rect_kernel(int(TARGET_MODULE_PX) | 1))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > (TARGET_MODULE_PX * 5) ** 2]
    if not contours:
        return None
    return _contour_quad(max(contours, key=cv2.contourArea))


def _plausible_quad(quad: np.ndarray) -> bool:
    """Konvex, nicht entartet, Seitenverhältnis und Winkel im Rahmen einer (perspektivisch) gesehenen Fläche."""
    if not cv2.isContourConvex(quad.reshape(-1, 1, 2).astype(np.float32)):
        return False
    sides = np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1)
    return bool(sides.min() > TARGET_MODULE_PX * 4 and sides.max() / sides.min() < 2.0)


def _fit_symbol(feature: np.ndarray, quad0: np.ndarray, screened: list, deadline: float | None
                ) -> tuple[np.ndarray | None, float]:
    """
    Orientierung und Skalierung per grobem Multistart (4 Rotationen × 3 Skalen), danach Feinfit bis 0,5 px.
    screened: die 4 Rotations-Fits der Grobprüfung (Schritt 4 px), die hier nur fortgesetzt werden.
    """
    center = quad0.mean(axis=0)
    fits = [_fit_frame(feature, quad, (2.0,)) for quad, _ in screened]
    for scale in (0.94, 1.06):
        if deadline is not None and time.perf_counter() > deadline:
            break
        for rotation in range(4):
            start = center + (np.roll(quad0, rotation, axis=0) - center) * scale
            fits.append(_fit_frame(feature, start, (4.0, 2.0)))
    quad, _ = max(fits, key=lambda f: f[1])
    quad, frame_t = _fit_frame(feature, quad, (1.0, 0.5))
    if not _plausible_quad(quad):
        return None, frame_t
    return quad, frame_t


# --------------------------------------------------------------------------- #
#  Dekodierung der Datenmodule                                                 #
# --------------------------------------------------------------------------- #

def _otsu_threshold(values: np.ndarray) -> float:
    s = np.sort(values)
    n = len(s)
    cumsum = np.cumsum(s)
    best, threshold = -1.0, float(s[n // 2])
    for i in range(4, n - 4):
        mean_a = cumsum[i - 1] / i
        mean_b = (cumsum[-1] - cumsum[i - 1]) / (n - i)
        between = i * (n - i) * (mean_a - mean_b) ** 2
        if between > best:
            best, threshold = between, float((s[i - 1] + s[i]) / 2)
    return threshold


def _rs_text(codewords: list[int], erasures=(), max_errors: int | None = None) -> tuple[str | None, int | None]:
    """Reed-Solomon-korrigierter Inhalt (auch Fremdinhalt, dann ggf. '') und Fehlerzahl; (None, None) ohne Decode."""
    decoded = rs_decode(codewords, erasures, max_errors)
    if decoded is None:
        return None, None
    return decode_ascii_codewords(decoded[0][:3]) or "", decoded[1]


def decode_data_modules(values: np.ndarray, frame_t: float) -> dict:
    """
    64 Datenmodul-Werte (hell = groß) → Code mit Stufe.
    Reed-Solomon und Codebuch-Korrelation müssen übereinstimmen; ein Widerspruch verwirft das Symbol.
    Ein fehlerfrei RS-dekodierbarer Fremdinhalt (kein Horden-Code) wird nie per Korrelation umgedeutet.
    """
    codes, book = codebook_data_patterns()
    out = {"code": None, "tier": None, "ncc": 0.0, "margin": 0.0, "frame_t": float(frame_t),
           "rs_errors": None, "erasures": 0, "best": None}
    centered = values - values.mean()
    norm = float(np.linalg.norm(centered))
    if norm < 1e-6:
        return out
    scores = book @ (centered / norm).astype(np.float32)
    top2 = np.argpartition(scores, -2)[-2:]
    best_idx, second_idx = top2[np.argsort(scores[top2])[::-1]]
    best = codes[best_idx]
    out.update(ncc=float(scores[best_idx]), margin=float(scores[best_idx] - scores[second_idx]), best=best)

    threshold = _otsu_threshold(values)
    dark = values < threshold
    codewords = codewords_from_dark_modules(dark)

    rs_text, rs_errors = _rs_text(codewords)
    if rs_text is not None:
        rs_code = dmx_text_to_code(rs_text)
        if rs_code != best:
            logger.warning(f"[MODUL] Reed-Solomon liefert '{rs_text}', Codebuch '{best}' → verworfen.")
            return out
        if frame_t >= _RS_MIN_FRAME_T:
            out.update(code=rs_code, tier="rs", rs_errors=rs_errors)
            return out

    spread = max(float(values[~dark].mean() - values[dark].mean()) if dark.any() and (~dark).any() else 0.0, 1e-6)
    reliability = np.abs(values - threshold) / spread
    cw_index, _ = data_module_codewords()
    cw_reliability = np.array([reliability[cw_index == i].min() for i in range(len(codewords))])
    unreliable = [int(i) for i in np.argsort(cw_reliability)]
    for n_erasures in range(1, _MAX_ERASURES + 1):
        # 2·Fehler + Löschungen ≤ 4: mindestens ein Prüfsymbol bleibt zur Fehlererkennung übrig
        text, rs_errors = _rs_text(codewords, unreliable[:n_erasures], max_errors=(4 - n_erasures) // 2)
        code = dmx_text_to_code(text)
        if code is None:
            continue  # kein Decode oder Fehlkorrektur in Fremdinhalt
        if code != best:
            logger.warning(f"[MODUL] RS mit Löschungen liefert '{code}', Codebuch '{best}' → verworfen.")
            return out
        if frame_t >= _ERASURE_MIN_FRAME_T and out["ncc"] >= _ERASURE_MIN_NCC and out["margin"] >= _ERASURE_MIN_MARGIN:
            out.update(code=code, tier="rs_erasure", rs_errors=rs_errors, erasures=n_erasures)
            return out
        break

    if frame_t >= config.SOFT_MIN_FRAME_T and out["ncc"] >= config.SOFT_MIN_NCC \
            and out["margin"] >= config.SOFT_MIN_MARGIN:
        out.update(code=best, tier="soft")
    return out


# --------------------------------------------------------------------------- #
#  Kandidatensuche                                                             #
# --------------------------------------------------------------------------- #

def _dark_blobs(gray: np.ndarray, module_px: int) -> list[tuple[int, int, int, int, float, np.ndarray]]:
    """
    Kompakte dunkle Flächen in Symbolgröße (4,5-14 Module) nach lokaler Hintergrundsubtraktion
    → (x, y, w, h, Score, Hüllviereck). Mehrere Schwellen, weil Lüftungsschlitze und Kanten die
    Otsu-Schwelle verschieben und den Code zerteilen oder mit der Klarschrift verschmelzen.
    """
    f = gray.astype(np.float32)
    blur = cv2.GaussianBlur(f, (0, 0), module_px * 0.35)
    background = cv2.GaussianBlur(cv2.dilate(blur, rect_kernel(2 * module_px - 1)), (0, 0), module_px * 0.6)
    darkness = np.clip(background - blur, 0, 255).astype(np.uint8)
    otsu_threshold, _ = cv2.threshold(darkness, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blobs = []
    for factor in (0.35, 0.5, 0.75, 1.0):
        _, binary = cv2.threshold(darkness, max(2.0, otsu_threshold * factor), 255, cv2.THRESH_BINARY)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, rect_kernel(module_px | 1))
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect = w / h
            fill = cv2.contourArea(contour) / float(w * h)
            if 0.7 < aspect < 1.4:
                side = max(w, h)
                if module_px * 4.5 <= side <= module_px * 14 and fill >= 0.35:
                    blobs.append((x, y, w, h, fill * min(w, h) / side, _contour_quad(contour)))
            elif 0.45 < aspect <= 0.7 or 1.4 <= aspect < 2.2:
                # Code und Klarschrift zu einem Blob verschmolzen: beide quadratischen Enden als Kandidaten
                s = min(w, h)
                if not module_px * 4.5 <= s <= module_px * 14:
                    continue
                ends = [(x, y), (x, y + h - s)] if h > w else [(x, y), (x + w - s, y)]
                for ex, ey in ends:
                    square = np.float32([[ex, ey], [ex + s, ey], [ex + s, ey + s], [ex, ey + s]])
                    blobs.append((ex, ey, s, s, 0.8 * fill, square))
    return blobs


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def find_symbol_candidates(gray: np.ndarray, search_boxes=(), max_candidates: int = MAX_CANDIDATES) -> list[dict]:
    """
    Symbolkandidaten in den Suchbereichen (YOLO-DMX-Boxen mit Rand, sonst Gesamtbild).

    Returns:
        [{"box": (x1, y1, x2, y2) Blob, "module_px": geschätzte Modulgröße, "score": float,
          "quad": Hüllviereck des Blobs im Bild}, ...]
    """
    h_img, w_img = gray.shape[:2]
    regions = []
    for box in list(search_boxes)[:2]:
        x1, y1, x2, y2 = (int(v) for v in box)
        pad = max(32, int(max(x2 - x1, y2 - y1) * 0.12))
        regions.append((max(0, x1 - pad), max(0, y1 - pad), min(w_img, x2 + pad), min(h_img, y2 + pad)))
    if not regions:
        regions.append((0, 0, w_img, h_img))

    candidates = []
    for rx1, ry1, rx2, ry2 in regions:
        region = gray[ry1:ry2, rx1:rx2]
        if min(region.shape[:2]) < _SEARCH_MODULE_PX * 5:
            continue
        for scale in _SEARCH_SCALES:
            img = region if scale == 1.0 else cv2.resize(region, None, fx=scale, fy=scale,
                                                         interpolation=cv2.INTER_AREA)
            for x, y, w, h, score, quad in _dark_blobs(img, _SEARCH_MODULE_PX):
                box = (rx1 + x / scale, ry1 + y / scale, rx1 + (x + w) / scale, ry1 + (y + h) / scale)
                candidates.append({"box": box, "module_px": max(w, h) / scale / _BLOB_SIDE_PER_MODULE,
                                   "score": score, "quad": quad / scale + np.float32([rx1, ry1])})

    candidates.sort(key=lambda c: -c["score"])
    kept = []
    for cand in candidates:
        # nur echte Dubletten verwerfen: Code-Blob und Code+Text-Blob überlappen stark, beide bleiben Kandidaten
        if all(_iou(cand["box"], k["box"]) < 0.8 for k in kept):
            kept.append(cand)
    return kept[:max_candidates]


# --------------------------------------------------------------------------- #
#  Öffentliche Schnittstelle                                                   #
# --------------------------------------------------------------------------- #

# Klarschrift unter dem Symbol in Moduleinheiten (0-2 Module Abstand, ~4-5 Module hoch, leicht nach rechts versetzt)
_TEXT_REGION = (0.0, _N + 1.2, _N + 0.15, _N + 7.0)


def text_crop_from_quad(frame: np.ndarray, quad, px_per_module: int = 16) -> np.ndarray:
    """Entzerrter, aufrechter Ausschnitt der Klarschrift unter dem Code (unabhängig von Drehung und Perspektive)."""
    x0, x1, y0, y1 = _TEXT_REGION
    s = float(px_per_module)
    to_grid = np.array([[1.0 / s, 0.0, x0], [0.0, 1.0 / s, y0], [0.0, 0.0, 1.0]])
    matrix = _homographies(np.float32(quad)[None])[0] @ to_grid
    size = (int((x1 - x0) * s), int((y1 - y0) * s))
    return cv2.warpPerspective(frame, matrix, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REPLICATE)


def _rank(result: dict) -> tuple:
    """Reed-Solomon vor Soft (nach Marge) vor nicht angenommenen Kandidaten (nur fürs Log, nach Korrelation)."""
    if result["tier"] in ("rs", "rs_erasure"):
        return (2, result["ncc"])
    return (1, result["margin"]) if result["tier"] == "soft" else (0, result["ncc"])


def _prepare_candidate(gray: np.ndarray, cand: dict) -> dict | None:
    """ROI (Blob + 20 %), ggf. auf die Ziel-Modulgröße skaliert, Merkmalsbild, Startviereck und Grobprüfung."""
    x1, y1, x2, y2 = cand["box"]
    side = max(x2 - x1, y2 - y1)
    margin = side * 0.2
    h_img, w_img = gray.shape[:2]
    rx1, ry1 = int(max(0, x1 - margin)), int(max(0, y1 - margin))
    rx2, ry2 = int(min(w_img, x2 + margin)), int(min(h_img, y2 + margin))
    roi = gray[ry1:ry2, rx1:rx2]
    module_px = cand["module_px"]
    lo, hi = _NATIVE_MODULE_RANGE
    scale = 1.0 if lo <= module_px <= hi else float(np.clip(TARGET_MODULE_PX / max(module_px, 1e-3), 0.25, 3.0))
    if scale == 1.0:
        roi_scaled = roi.astype(np.float32)
    else:
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        roi_scaled = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=interpolation).astype(np.float32)

    quad0 = (cand["quad"] - np.float32([rx1, ry1])) * scale if cand.get("quad") is not None \
        else _initial_quad(roi_scaled)
    if quad0 is None:
        return None
    feature = speckle_feature(roi_scaled)
    screened = [_fit_frame(feature, np.roll(quad0, rotation, axis=0), (4.0,)) for rotation in range(4)]
    return {"feature": feature, "quad0": quad0, "scale": scale, "offset": np.float32([rx1, ry1]),
            "screened": screened, "screen_t": max(t for _, t in screened)}


def _read_prepared(prep: dict, deadline: float | None) -> dict | None:
    quad, frame_t = _fit_symbol(prep["feature"], prep["quad0"], prep["screened"], deadline)
    if quad is None:
        return None
    values = _module_means(prep["feature"], quad[None], _DATA_POINTS, len(_SUB_DATA))[0]
    result = decode_data_modules(values, frame_t)
    result["quad"] = quad / prep["scale"] + prep["offset"]
    return result


def read_dmx_modules(frame: np.ndarray, search_boxes=(), deadline: float | None = None) -> dict:
    """
    Liest eine 10x10-DataMatrix über den Modul-Decoder.

    Args:
        frame: Kamerabild (BGR oder Graustufen).
        search_boxes: YOLO-DataMatrix-Boxen (x1, y1, x2, y2), beste zuerst; leer = Gesamtbild.
        deadline: time.perf_counter()-Zeitpunkt, nach dem keine weiteren Kandidaten mehr geprüft werden.

    Returns:
        dict mit code, tier ("rs" | "rs_erasure" | "soft" | None), ncc, margin, frame_t, rs_errors,
        erasures, quad (4 Ecken im Bild, L-Finder-Ecke links unten = Index 3), candidates, ms.
    """
    t0 = time.perf_counter()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    empty = {"code": None, "tier": None, "ncc": 0.0, "margin": 0.0, "frame_t": 0.0, "rs_errors": None,
             "erasures": 0, "best": None, "quad": None}
    best = None
    fits = 0
    candidates = find_symbol_candidates(gray, search_boxes)
    pending = []
    for index, cand in enumerate(candidates):
        if deadline is not None and index > 0 and time.perf_counter() > deadline:
            break
        try:
            prep = _prepare_candidate(gray, cand)
        except cv2.error as e:
            logger.debug(f"[MODUL] Kandidat {cand['box']} Fehler: {e}")
            continue
        if prep is None:
            continue
        if prep["screen_t"] < _EAGER_SCREEN_T:
            pending.append(prep)
            continue
        # Deutlicher Rahmen: sofort fitten; Reed-Solomon oder Soft beendet die Suche
        fits += 1
        result = _read_prepared(prep, deadline)
        if result is not None and (best is None or _rank(result) > _rank(best)):
            best = result
        if best is not None and best["tier"] is not None:
            break

    if best is None or best["tier"] is None:
        pending.sort(key=lambda p: -p["screen_t"])
        for prep in pending:
            if fits >= MAX_FULL_FITS or prep["screen_t"] < _MIN_SCREEN_T:
                break
            if deadline is not None and fits > 0 and time.perf_counter() > deadline:
                break
            fits += 1
            result = _read_prepared(prep, deadline)
            if result is None:
                continue
            if best is None or _rank(result) > _rank(best):
                best = result
            if best["tier"] is not None:
                break

    out = dict(best or empty)
    out["candidates"] = len(candidates)
    out["ms"] = int((time.perf_counter() - t0) * 1000)
    if out["tier"]:
        logger.info(f"[MODUL] '{out['code']}' ({out['tier']}, NCC={out['ncc']:.2f}, Marge={out['margin']:.2f}, "
                    f"Rahmen t={out['frame_t']:.2f}, RS-Fehler={out['rs_errors']}, Löschungen={out['erasures']}, "
                    f"{out['candidates']} Kandidaten, {out['ms']}ms)")
    else:
        logger.info(f"[MODUL] Kein Code (bester Kandidat '{out['best']}' NCC={out['ncc']:.2f}, "
                    f"Marge={out['margin']:.2f}, t={out['frame_t']:.2f}, {out['candidates']} Kandidaten, {out['ms']}ms)")
    return out
