"""
Geometrie des 10x10-DataMatrix-Gitters: Etikett-/Kandidatensuche, Eckenausrichtung,
perspektivisches Sampling der Module und Abgleich beobachteter Gitter mit Referenzgittern.
"""

import logging

import cv2
import numpy as np

from .code_format import generate_10_candidates_from_partial
from .dmx_codec import all_code_grids, generate_reference_grid
from .image_ops import (clahe, otsu, preprocess_etch_denoise, preprocess_niblack, preprocess_ridge_enhancement,
                        preprocess_sauvola, rect_kernel, to_gray)

logger = logging.getLogger(__name__)

DMTX_GRID_SIZE = 10
DMTX_CELL_PX = 20
DMTX_WARP_SIZE = DMTX_GRID_SIZE * DMTX_CELL_PX

# Binarisierungsvarianten für extract_observed_grid() (Reihenfolge = Priorität)
ALL_GRID_METHODS = ("otsu", "adaptive", "mean", "clahe_otsu", "etch_denoise", "ridge_boost",
                    "sauvola_w11", "sauvola_w21", "niblack")

_EXPECTED_TOP = np.array([0 if i % 2 == 0 else 1 for i in range(DMTX_GRID_SIZE)])
_EXPECTED_RIGHT = _EXPECTED_TOP.copy()
_EXPECTED_RIGHT[DMTX_GRID_SIZE - 1] = 0


def frame_pattern_ok(grid: np.ndarray, min_finder: int, min_timing: int) -> list[bool]:
    """Prüft L-Finder (links, unten) und Timing-Muster (oben, rechts) eines 10x10-Gitters."""
    return [
        np.sum(grid[:, 0] == 0) >= min_finder,
        np.sum(grid[DMTX_GRID_SIZE - 1, :] == 0) >= min_finder,
        np.sum(grid[0, :] == _EXPECTED_TOP) >= min_timing,
        np.sum(grid[:, DMTX_GRID_SIZE - 1] == _EXPECTED_RIGHT) >= min_timing,
    ]


# --------------------------------------------------------------------------- #
#  Etikett- und Kandidatensuche                                                #
# --------------------------------------------------------------------------- #

def find_label_crop(gray: np.ndarray) -> np.ndarray:
    """Grober Etikett-Ausschnitt: größte helle Otsu-Kontur (> 10.000 px²) plus 10 px Rand, sonst das ganze Bild."""
    h, w = gray.shape[:2]
    contours, _ = cv2.findContours(otsu(gray), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 10000:
            x, y, bw, bh = cv2.boundingRect(largest)
            pad = 10
            return gray[max(0, y - pad):min(h, y + bh + pad), max(0, x - pad):min(w, x + bw + pad)]
    return gray


def collect_square_candidates(binary_inv: np.ndarray, kernel_sizes, max_aspect: float,
                              candidates: list, seen_centers: list) -> None:
    """
    Sucht annähernd quadratische Konturen (DMX-Kandidaten) nach morphologischem Schließen.
    Hängt (Kontur, Fläche, minAreaRect) an candidates an; Zentren näher als 20 px gelten als Duplikat.
    """
    max_area = binary_inv.shape[0] * binary_inv.shape[1] * 0.70
    for k_size in kernel_sizes:
        closed = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, rect_kernel(k_size), iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < 400 or area > max_area:
                continue
            rect = cv2.minAreaRect(c)
            rect_w, rect_h = rect[1]
            if rect_w == 0 or rect_h == 0 or max(rect_w, rect_h) / min(rect_w, rect_h) > max_aspect:
                continue
            center = rect[0]
            if any(np.sqrt((center[0] - s[0])**2 + (center[1] - s[1])**2) < 20 for s in seen_centers):
                continue
            seen_centers.append(center)
            candidates.append((c, area, rect))


# --------------------------------------------------------------------------- #
#  Ecken & Sampling                                                            #
# --------------------------------------------------------------------------- #

def repair_l_finder(binary_image: np.ndarray) -> np.ndarray:
    """Repariert durch Ätzung unterbrochene L-Finder-Schenkel (richtungsgebundenes Closing, UND-verknüpft)."""
    if binary_image is None or binary_image.size == 0:
        return binary_image
    closed_h = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1)))
    closed_v = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7)))
    return cv2.bitwise_and(closed_h, closed_v)


def _intersection_of_lines(line1: tuple, line2: tuple) -> np.ndarray | None:
    """Schnittpunkt zweier Geraden (Richtungsvektor + Punkt)."""
    (vx1, vy1, x1, y1) = line1
    (vx2, vy2, x2, y2) = line2
    denom = vx1 * vy2 - vy1 * vx2
    if abs(denom) < 1e-6:
        return None
    t = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / denom
    return np.array([x1 + t * vx1, y1 + t * vy1], dtype=np.float32)


def _sort_by_angle(corners: np.ndarray) -> np.ndarray:
    center = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    return corners[np.argsort(angles)]


def refine_corners_ransac(image: np.ndarray, initial_corners: np.ndarray) -> np.ndarray:
    """
    Verfeinert grobe Ecken (minAreaRect) durch robuste Linienanpassung an die 4 Außenkanten.
    Liefert die unveränderten Ecken, wenn die Verfeinerung unplausibel ist (> 25 px Abweichung).
    """
    if initial_corners is None or len(initial_corners) != 4:
        return initial_corners

    edges = cv2.Canny(to_gray(image), 50, 150)
    sorted_pts = _sort_by_angle(initial_corners)

    lines = []
    for i in range(4):
        pt1 = sorted_pts[i]
        pt2 = sorted_pts[(i + 1) % 4]
        mask = np.zeros_like(edges)
        cv2.line(mask, tuple(pt1.astype(int)), tuple(pt2.astype(int)), 255, 15)
        edge_pts = np.column_stack(np.where((edges > 0) & (mask > 0)))

        if len(edge_pts) >= 10:
            fit = cv2.fitLine(np.float32(edge_pts[:, [1, 0]]), cv2.DIST_HUBER, 0, 0.01, 0.01)
            lines.append((float(fit[0][0]), float(fit[1][0]), float(fit[2][0]), float(fit[3][0])))
        else:
            vx = float(pt2[0] - pt1[0])
            vy = float(pt2[1] - pt1[1])
            norm = float(np.hypot(vx, vy) + 1e-6)
            lines.append((vx / norm, vy / norm, float(pt1[0]), float(pt1[1])))

    refined_corners = []
    for i in range(4):
        intersect = _intersection_of_lines(lines[i], lines[(i + 1) % 4])
        refined_corners.append(intersect if intersect is not None else sorted_pts[(i + 1) % 4])
    refined = np.array(refined_corners, dtype=np.float32)

    if np.isnan(refined).any() or np.isinf(refined).any():
        return initial_corners
    if np.max(np.abs(refined - sorted_pts)) > 25.0:
        return initial_corners
    return refined


def _warp_square(gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
    dst_pts = np.float32([[0, 0], [DMTX_WARP_SIZE, 0], [DMTX_WARP_SIZE, DMTX_WARP_SIZE], [0, DMTX_WARP_SIZE]])
    M = cv2.getPerspectiveTransform(corners, dst_pts)
    return cv2.warpPerspective(gray, M, (DMTX_WARP_SIZE, DMTX_WARP_SIZE),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _valid_corners(corners) -> bool:
    return corners is not None and isinstance(corners, np.ndarray) and corners.shape == (4, 2)


def orient_corners(gray: np.ndarray, corners: np.ndarray, strict: bool = True) -> np.ndarray | None:
    """
    Bestimmt die Ausrichtung der 4 Ecken: testet alle 4 Rotationen und bewertet L-Finder und
    Timing-Muster (max. 40 Punkte). Mindestscore 20 (strict) bzw. 16 für verätzte L-Finder.
    """
    corners = refine_corners_ransac(gray, corners)
    if not _valid_corners(corners):
        return None
    sorted_corners = _sort_by_angle(np.float32(corners))

    cell = DMTX_CELL_PX
    best_score = -1
    best_corners = None
    for rotation in range(4):
        rotated = np.float32(np.roll(sorted_corners, rotation, axis=0))
        repaired_bin = repair_l_finder(otsu(_warp_square(gray, rotated)))

        cells = np.zeros((DMTX_GRID_SIZE, DMTX_GRID_SIZE), dtype=np.uint8)
        for row in range(DMTX_GRID_SIZE):
            for col in range(DMTX_GRID_SIZE):
                cy = row * cell + cell // 2
                cx = col * cell + cell // 2
                region = repaired_bin[cy - 3:cy + 3, cx - 3:cx + 3]
                cells[row, col] = 1 if np.mean(region) > 127 else 0

        score = 0.0
        score += np.sum(cells[:, 0] == 0)
        score += np.sum(cells[DMTX_GRID_SIZE - 1, :] == 0)
        score += np.sum(cells[0, :] == _EXPECTED_TOP)
        score += np.sum(cells[:, DMTX_GRID_SIZE - 1] == _EXPECTED_RIGHT)
        logger.debug(f"Rekonstruktion: Rotation {rotation}, Score {score:.0f}/40")

        if score > best_score:
            best_score = score
            best_corners = rotated.copy()

    min_score = 20 if strict else 16
    if best_score < min_score:
        logger.debug(f"Rekonstruktion: Bester Score {best_score:.0f}/40 zu niedrig (min {min_score}).")
        return None

    logger.info(f"Rekonstruktion: Orientierung gefunden, Score {best_score:.0f}/40.")
    return best_corners


def _binarize_warped(warped: np.ndarray, method: str) -> np.ndarray:
    if method == "adaptive":
        return cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 4)
    if method == "mean":
        _, binary = cv2.threshold(warped, np.mean(warped), 255, cv2.THRESH_BINARY)
        return binary
    if method == "clahe_otsu":
        return otsu(clahe(warped, 8.0, tile=4))
    if method == "etch_denoise":
        return otsu(preprocess_etch_denoise(warped))
    if method == "ridge_boost":
        return otsu(preprocess_ridge_enhancement(warped))
    if method == "sauvola_w11":
        return preprocess_sauvola(warped, window_size=11, k=0.2)
    if method == "sauvola_w21":
        return preprocess_sauvola(warped, window_size=21, k=0.2)
    if method == "niblack":
        return preprocess_niblack(warped, window_size=21, k=-0.2)
    return otsu(warped)


def warp_and_sample(gray: np.ndarray, corners: np.ndarray, binarization_method: str = "otsu",
                    strict_l_finder: bool = True) -> np.ndarray | None:
    """
    Entzerrt die ausgerichteten Ecken und samplet die 10x10-Module (0 = schwarz, 1 = weiß).
    None, wenn der L-Finder nach dem Sampling nicht plausibel ist.
    """
    if gray is None or gray.size == 0 or not _valid_corners(corners):
        return None

    warped_bin = _binarize_warped(_warp_square(gray, np.float32(corners)), binarization_method)
    # Schließt kleine Ätzbecken-Löcher in schwarzen Modulen
    warped_bin_clean = cv2.morphologyEx(warped_bin, cv2.MORPH_CLOSE, rect_kernel(3))

    cell = DMTX_CELL_PX
    half = max(2, cell // 3)
    cells = np.zeros((DMTX_GRID_SIZE, DMTX_GRID_SIZE), dtype=np.uint8)
    for row in range(DMTX_GRID_SIZE):
        for col in range(DMTX_GRID_SIZE):
            cy = row * cell + cell // 2
            cx = col * cell + cell // 2
            region = warped_bin_clean[cy - half:cy + half, cx - half:cx + half]
            if region.size:
                cells[row, col] = 1 if np.median(region) > 127 else 0

    min_l = 5 if strict_l_finder else 4
    left_black = np.sum(cells[:, 0] == 0)
    bottom_black = np.sum(cells[DMTX_GRID_SIZE - 1, :] == 0)
    if left_black < min_l:
        logger.debug(f"Rekonstruktion: L-Finder links nicht ausreichend nach Sampling ({left_black}/10).")
        return None
    if bottom_black < min_l:
        logger.debug(f"Rekonstruktion: L-Finder unten nicht ausreichend nach Sampling ({bottom_black}/10).")
        return None

    logger.info("Rekonstruktion: 10x10 Binärmatrix erfolgreich extrahiert.")
    return cells


def warp_and_sample_soft(gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Wie warp_and_sample, aber mit kontinuierlichen Modulhelligkeiten in [0.0, 1.0] (0.0 = schwarz)."""
    if gray is None or gray.size == 0 or not _valid_corners(corners):
        return np.zeros((DMTX_GRID_SIZE, DMTX_GRID_SIZE), dtype=np.float32)

    warped = _warp_square(gray, np.float32(corners))
    p_low, p_high = np.percentile(warped, (2, 98))
    if p_high > p_low:
        warped_norm = np.clip((warped.astype(np.float32) - p_low) / (p_high - p_low), 0.0, 1.0)
    else:
        warped_norm = warped.astype(np.float32) / 255.0

    cell = DMTX_CELL_PX
    half = max(2, cell // 3)
    soft_matrix = np.zeros((DMTX_GRID_SIZE, DMTX_GRID_SIZE), dtype=np.float32)
    for row in range(DMTX_GRID_SIZE):
        for col in range(DMTX_GRID_SIZE):
            cy = row * cell + cell // 2
            cx = col * cell + cell // 2
            region = warped_norm[cy - half:cy + half, cx - half:cx + half]
            soft_matrix[row, col] = float(np.mean(region)) if region.size else 0.5
    return soft_matrix


def generate_synthetic_dmtx(cells: np.ndarray) -> np.ndarray:
    """Rendert ein 10x10-Gitter (0/1 oder Grauwerte 0..1) als sauberes DataMatrix-Bild mit Quiet Zone."""
    if cells is None or not isinstance(cells, np.ndarray) or cells.ndim != 2:
        return np.full((120, 120), 255, dtype=np.uint8)
    grid = cells.shape[0]
    cell_px = DMTX_CELL_PX
    quiet_zone = cell_px

    img_size = grid * cell_px + 2 * quiet_zone
    img = np.full((img_size, img_size), 255, dtype=np.uint8)
    for row in range(grid):
        for col in range(grid):
            x0 = quiet_zone + col * cell_px
            y0 = quiet_zone + row * cell_px
            val = cells[row, col]
            if isinstance(val, (float, np.floating)):
                color = int(np.clip(val * 255.0, 0, 255))
            else:
                color = 0 if val == 0 else 255
            img[y0:y0 + cell_px, x0:x0 + cell_px] = color

    logger.debug(f"Synthetisches DataMatrix-Bild generiert: {img_size}x{img_size} Pixel.")
    return img


# --------------------------------------------------------------------------- #
#  Beobachtete Gitter extrahieren                                              #
# --------------------------------------------------------------------------- #

def extract_observed_grid(frame: np.ndarray, binarization_method: str = "otsu",
                          strict_l_finder: bool = True) -> np.ndarray | None:
    """Sucht die DataMatrix im Etikett und extrahiert das beobachtete 10x10-Binärgitter oder None."""
    label_crop = find_label_crop(to_gray(frame))

    candidates = []
    seen_centers = []
    label_enhanced = label_crop
    for clip_limit in (3.0, 8.0, 15.0):
        label_enhanced = clahe(label_crop, clip_limit)
        crop_bin_adapt = cv2.adaptiveThreshold(label_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY, 21, 4)
        for b_img in (otsu(label_enhanced), crop_bin_adapt):
            collect_square_candidates(cv2.bitwise_not(b_img), (9,), 2.0, candidates, seen_centers)

    candidates.sort(key=lambda x: x[1], reverse=True)

    # Ausrichtung und Sampling laufen auf dem stärksten CLAHE-Bild (clipLimit 15)
    for _, _, rect in candidates:
        oriented = orient_corners(label_enhanced, np.float32(cv2.boxPoints(rect)), strict=strict_l_finder)
        if oriented is None:
            continue
        # Feinabstimmung der Ecken: Skalierung und Verschiebung, bestes L-Finder-Sampling gewinnt
        best_cells = None
        best_l_score = -1
        center = oriented.mean(axis=0)
        for scale in (0.96, 1.00, 1.04):
            scaled = center + (oriented - center) * scale
            for dx in (-4, 0, 4):
                for dy in (-4, 0, 4):
                    shifted = (scaled + np.array([dx, dy])).astype(np.float32)
                    cells = warp_and_sample(label_enhanced, shifted, binarization_method=binarization_method,
                                            strict_l_finder=strict_l_finder)
                    if cells is not None:
                        l_score = np.sum(cells[:, 0] == 0) + np.sum(cells[9, :] == 0)
                        if l_score > best_l_score:
                            best_l_score = l_score
                            best_cells = cells
        if best_cells is not None:
            return best_cells

    return None


def extract_observed_grid_soft(frame: np.ndarray) -> np.ndarray | None:
    """Sucht die DataMatrix im Etikett und extrahiert die kontinuierliche 10x10-Helligkeitsmatrix oder None."""
    label_crop = find_label_crop(to_gray(frame))

    candidates = []
    seen_centers = []
    for clip_limit in (3.0, 8.0, 15.0):
        crop_inv = cv2.bitwise_not(otsu(clahe(label_crop, clip_limit)))
        collect_square_candidates(crop_inv, (35, 25, 15, 9), 2.0, candidates, seen_centers)

    candidates.sort(key=lambda x: x[1], reverse=True)

    for _, _, rect in candidates:
        oriented = orient_corners(label_crop, np.float32(cv2.boxPoints(rect)), strict=False)
        if oriented is not None:
            return warp_and_sample_soft(label_crop, oriented)
    return None


# --------------------------------------------------------------------------- #
#  Abgleich mit Referenzgittern                                                #
# --------------------------------------------------------------------------- #

def match_soft_grid_matrix(soft_matrix: np.ndarray,
                           candidate_codes: set[str] | list[str] = None) -> tuple[str | None, float, float]:
    """
    Soft-Matching einer 10x10-Helligkeitsmatrix P gegen die Referenzgitter R:
    Score(C) = 1 - mean(|P - R(C)|).

    Returns:
        (bester Code, bester Score, Abstand zum Zweitbesten)
    """
    all_codes, all_matrix = all_code_grids()
    if soft_matrix is None or soft_matrix.shape != (10, 10):
        return None, 0.0, 0.0

    soft_flat = soft_matrix.flatten().astype(np.float32)
    scores = 1.0 - np.mean(np.abs(all_matrix.astype(np.float32) - soft_flat), axis=1)

    if candidate_codes and len(candidate_codes) < 4000:
        cand_indices = np.array([all_codes.index(c) for c in list(candidate_codes) if c in all_codes])
        if cand_indices.size == 0:
            return None, 0.0, 0.0
        sub_scores = scores[cand_indices]
        sorted_arg = np.argsort(sub_scores)[::-1]
        best_code = all_codes[cand_indices[sorted_arg[0]]]
        best_score = float(sub_scores[sorted_arg[0]])
        second_score = float(sub_scores[sorted_arg[1]]) if len(sorted_arg) > 1 else 0.0
        return best_code, best_score, best_score - second_score

    top2 = np.argpartition(scores, -2)[-2:]
    best_idx, second_idx = top2[np.argsort(scores[top2])[::-1]]
    return all_codes[best_idx], float(scores[best_idx]), float(scores[best_idx] - scores[second_idx])


def cross_validate_ocr_dmtx(ocr_partial: str, observed_grid: np.ndarray) -> tuple[str | None, float]:
    """Wählt unter den Teilcode-Kandidaten das Referenzgitter mit der höchsten Modul-Übereinstimmung (≥ 70 %)."""
    if not ocr_partial or observed_grid is None or observed_grid.shape != (10, 10):
        return None, 0.0

    cands = generate_10_candidates_from_partial(ocr_partial)
    if not cands:
        return None, 0.0

    all_codes, all_matrix = all_code_grids()
    obs_flat = observed_grid.flatten()
    best_cand = None
    best_score = -1.0
    for cand in cands:
        if cand in all_codes:
            match_score = np.mean(all_matrix[all_codes.index(cand)] == obs_flat)
            if match_score > best_score:
                best_score = match_score
                best_cand = cand

    if best_score >= 0.70:
        logger.info(f"Cross-Validation ERFOLGREICH: Teilcode '{ocr_partial}' -> DataMatrix Match '{best_cand}' (Score: {best_score:.1%})")
        return best_cand, float(best_score)
    return None, 0.0


def template_match_candidates(frame: np.ndarray, candidates: list[str]) -> tuple[str | None, float, float]:
    """
    Multi-Scale Template-Matching synthetischer DMX-Bilder der Kandidaten gegen das Bild (TM_CCOEFF_NORMED).
    Letzte Rückfallebene, wenn kein Gitter extrahierbar ist.

    Returns:
        (bester Kandidat, Score, Abstand zum Zweitbesten)
    """
    if not candidates or frame is None:
        return None, 0.0, 0.0

    gray = to_gray(frame)
    h, w = gray.shape[:2]
    enhanced = clahe(gray, 8.0)

    templates = {}
    for cand in candidates:
        ref_grid = generate_reference_grid(cand)
        if ref_grid is not None:
            templates[cand] = generate_synthetic_dmtx(ref_grid)
    if not templates:
        return None, 0.0, 0.0

    # Die DataMatrix nimmt typischerweise 15-80 % der kürzeren Bildseite ein
    min_tmpl_size = max(30, int(min(w, h) * 0.15))
    max_tmpl_size = min(int(min(w, h) * 0.8), max(w, h))
    scales = np.linspace(min_tmpl_size, max_tmpl_size, 8).astype(int)

    best_scores = {}
    for cand, synth in templates.items():
        cand_best_score = -1.0
        synth_h, synth_w = synth.shape[:2]
        for target_size in scales:
            scale_factor = target_size / synth_w
            new_w = int(synth_w * scale_factor)
            new_h = int(synth_h * scale_factor)
            if new_w >= w or new_h >= h or new_w < 20 or new_h < 20:
                continue
            resized = cv2.resize(synth, (new_w, new_h), interpolation=cv2.INTER_AREA)
            for img_variant in (enhanced, gray):
                result = cv2.matchTemplate(img_variant, resized, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                if max_val > cand_best_score:
                    cand_best_score = max_val
        best_scores[cand] = cand_best_score

    sorted_cands = sorted(best_scores.items(), key=lambda x: x[1], reverse=True)
    best_cand, best_score = sorted_cands[0]
    second_score = sorted_cands[1][1] if len(sorted_cands) > 1 else 0.0
    margin = best_score - second_score

    logger.info(
        f"Template-Matching: Bester='{best_cand}' Score={best_score:.3f}, "
        f"Zweitbester='{sorted_cands[1][0] if len(sorted_cands) > 1 else '-'}' "
        f"Score={second_score:.3f}, Margin={margin:.3f}"
    )
    return best_cand, float(best_score), float(margin)
