"""Allgemeine Bildoperationen: Graustufen, Kontrast-/Binarisierungsfilter und Zuschnitte."""

from functools import lru_cache

import cv2
import numpy as np


def to_gray(image: np.ndarray) -> np.ndarray:
    """Graustufen-Kopie eines BGR- oder Graustufenbildes."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


def disk(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def rect_kernel(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))


def otsu(gray: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def clahe(gray: np.ndarray, clip_limit: float, tile: int = 8) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile)).apply(gray)


@lru_cache(maxsize=None)
def gamma_lut(gamma: float) -> np.ndarray:
    return np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")


def sharpen(image: np.ndarray) -> np.ndarray:
    """Unsharp-Mask-Schärfung."""
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)


def percentile_stretch(gray: np.ndarray, low: float, high: float) -> np.ndarray:
    """Spreizt das Grauwertspektrum zwischen den Perzentilen low/high auf 0..255."""
    p_low, p_high = np.percentile(gray, (low, high))
    if p_high > p_low:
        return np.clip((gray.astype(np.float32) - p_low) * (255.0 / (p_high - p_low)), 0, 255).astype(np.uint8)
    return gray


def preprocess_tophat(image: np.ndarray) -> np.ndarray:
    """Entfernt Spiegelungen und Glanzstellen auf Metall- und Plastiketiketten per Top-Hat-Filter."""
    gray = to_gray(image)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rect_kernel(15))
    return sharpen(cv2.addWeighted(gray, 1.0, tophat, 1.5, 0))


def preprocess_faded_contrast(image: np.ndarray) -> np.ndarray:
    """Für verbleichte, kontrastarme Codes: Perzentil-Stretching (2-98 %) + TopHat/BlackHat-Boost."""
    stretched = percentile_stretch(to_gray(image), 2, 98)
    kernel = rect_kernel(9)
    tophat = cv2.morphologyEx(stretched, cv2.MORPH_TOPHAT, kernel)
    bottomhat = cv2.morphologyEx(stretched, cv2.MORPH_BLACKHAT, kernel)
    enhanced = cv2.subtract(cv2.add(stretched, tophat), bottomhat)
    return sharpen(enhanced)


def preprocess_etch_denoise(image: np.ndarray) -> np.ndarray:
    """Neutralisiert Ätzspuren, Säureflecken und Lochfraß auf metallischen/geätzten Horden."""
    stretched = percentile_stretch(to_gray(image), 1, 99)

    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    tophat = cv2.morphologyEx(stretched, cv2.MORPH_TOPHAT, kernel_large)
    blackhat = cv2.morphologyEx(stretched, cv2.MORPH_BLACKHAT, kernel_large)
    enhanced = cv2.addWeighted(stretched, 1.0, tophat, 1.2, 0)
    enhanced = cv2.subtract(enhanced, (blackhat * 1.2).astype(np.uint8))

    # Schließt Modullöcher/Ätzspuren
    closed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, rect_kernel(3))
    return sharpen(closed)


def preprocess_ridge_enhancement(image: np.ndarray) -> np.ndarray:
    """Sobel-Kantenverstärkung für stark verblasste Ätzpunkte und schwache Buchstabenstriche."""
    stretched = percentile_stretch(to_gray(image), 2, 98)
    abs_grad_x = cv2.convertScaleAbs(cv2.Sobel(stretched, cv2.CV_16S, 1, 0, ksize=3))
    abs_grad_y = cv2.convertScaleAbs(cv2.Sobel(stretched, cv2.CV_16S, 0, 1, ksize=3))
    grad_mag = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return sharpen(cv2.addWeighted(stretched, 0.7, grad_mag, 0.5, 0))


def _local_mean_std(gray_f: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    mean = cv2.boxFilter(gray_f, cv2.CV_32F, (window_size, window_size))
    sqr_mean = cv2.boxFilter(gray_f**2, cv2.CV_32F, (window_size, window_size))
    return mean, np.sqrt(np.maximum(0, sqr_mean - mean**2))


def preprocess_sauvola(image: np.ndarray, window_size: int = 15, k: float = 0.2) -> np.ndarray:
    """Lokale Sauvola-Binarisierung: T = mean * (1 + k * (std / R - 1))."""
    gray_f = to_gray(image).astype(np.float32)
    mean, std = _local_mean_std(gray_f, window_size)
    R = 128.0
    thresh = mean * (1.0 + k * (std / R - 1.0))
    return np.where(gray_f >= thresh, 255, 0).astype(np.uint8)


def preprocess_niblack(image: np.ndarray, window_size: int = 21, k: float = -0.2) -> np.ndarray:
    """Lokale Niblack-Binarisierung: T = mean + k * std."""
    gray_f = to_gray(image).astype(np.float32)
    mean, std = _local_mean_std(gray_f, window_size)
    thresh = mean + k * std
    return np.where(gray_f >= thresh, 255, 0).astype(np.uint8)


def crop_with_margin(gray: np.ndarray, box, margin_ratio: float) -> np.ndarray:
    x1, y1, x2, y2 = (int(v) for v in box)
    margin = int(max(x2 - x1, y2 - y1) * margin_ratio)
    h, w = gray.shape[:2]
    return gray[max(0, y1 - margin):min(h, y2 + margin), max(0, x1 - margin):min(w, x2 + margin)]


def deskew_crop(image: np.ndarray, box: tuple[int, int, int, int], padding: int = 60) -> np.ndarray:
    """
    Schneidet eine YOLO-Box aus und begradigt sie (Deskewing), falls das Etikett schräg liegt.
    Garantiert immer eine Quiet Zone von 25 px für DataMatrix/OCR.

    Args:
        image: Das Originalbild (BGR).
        box: Die YOLO Bounding Box (x1, y1, x2, y2).
        padding: Suchrand um die Box für die Winkelbestimmung.
    """
    h_img, w_img = image.shape[:2]
    x1, y1, x2, y2 = box
    w_box = x2 - x1
    h_box = y2 - y1

    pad_safe = 25
    padded_crop_direct = image[max(0, y1 - pad_safe):min(h_img, y2 + pad_safe),
                               max(0, x1 - pad_safe):min(w_img, x2 + pad_safe)]

    px1 = max(0, x1 - padding)
    py1 = max(0, y1 - padding)
    px2 = min(w_img, x2 + padding)
    py2 = min(h_img, y2 + padding)
    crop = image[py1:py2, px1:px2]
    if crop.size == 0:
        return padded_crop_direct

    # Dominanten Kantenwinkel bestimmen (Canny + Closing verbindet die Etikettkanten)
    blurred = cv2.GaussianBlur(to_gray(crop), (5, 5), 0)
    closed = cv2.morphologyEx(cv2.Canny(blurred, 30, 100), cv2.MORPH_CLOSE, rect_kernel(9))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_rects = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > 1500:
            valid_rects.append((area, cv2.minAreaRect(c)))
    if not valid_rects:
        return padded_crop_direct

    _, best_rect = max(valid_rects, key=lambda x: x[0])
    _, size, angle = best_rect
    w_rect, h_rect = size
    if w_rect < h_rect:
        angle += 90.0
    if angle > 45.0:
        angle -= 90.0
    elif angle < -45.0:
        angle += 90.0
    if abs(angle) < 1.0:
        return padded_crop_direct

    # Um den Mittelpunkt der YOLO-Box (relativ zum Suchausschnitt) rotieren
    cx_crop = (x1 + x2) / 2.0 - px1
    cy_crop = (y1 + y2) / 2.0 - py1
    M = cv2.getRotationMatrix2D((cx_crop, cy_crop), angle, 1.0)
    rotated = cv2.warpAffine(crop, M, (crop.shape[1], crop.shape[0]),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Begradigtes Etikett in YOLO-Box-Größe plus Quiet Zone ausschneiden
    rx1 = max(0, int(cx_crop - (w_box / 2.0)) - pad_safe)
    ry1 = max(0, int(cy_crop - (h_box / 2.0)) - pad_safe)
    rx2 = min(rotated.shape[1], int(cx_crop + (w_box / 2.0)) + pad_safe)
    ry2 = min(rotated.shape[0], int(cy_crop + (h_box / 2.0)) + pad_safe)
    final_crop = rotated[ry1:ry2, rx1:rx2]
    if final_crop.size == 0:
        return padded_crop_direct
    return final_crop
