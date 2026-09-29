"""
DataMatrix ECC200-Encoder für 10x10-Symbole (4-stellige Horden-Codes):
ASCII-Codewörter, Reed-Solomon GF(256) und Utah-Placement nach ISO/IEC 16022.
Liefert die fehlerfreien Referenzgitter für den Abgleich mit beobachteten Gittern.
"""

import logging
from functools import lru_cache

import numpy as np

from .config import ALLOWED_CHARS, REQUIRED_LENGTH

logger = logging.getLogger(__name__)

GRID_SIZE = 10
_DATA_SIZE = 8          # innerer Datenbereich (ohne L-Finder und Timing-Muster)
_NUM_DATA_CW = 3
_NUM_ECC_CW = 5

# --- Reed-Solomon: GF(256) mit Primitivpolynom 0x12D ---
_GF_POLY = 0x12D
_gf_exp = [0] * 512
_gf_log = [0] * 256


def _init_gf_tables():
    x = 1
    _gf_exp[0] = 1
    for i in range(1, 255):
        x <<= 1
        if x & 0x100:
            x ^= _GF_POLY
        _gf_exp[i] = x
        _gf_log[x] = i
    for i in range(255, 512):
        _gf_exp[i] = _gf_exp[i - 255]


_init_gf_tables()


def _gf_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return _gf_exp[_gf_log[a] + _gf_log[b]]


def compute_rs_ecc(data: list[int]) -> list[int]:
    """Berechnet die 5 Reed-Solomon-Fehlerkorrektur-Codewörter für 3 Datencodewörter."""
    g = [1]
    for i in range(1, _NUM_ECC_CW + 1):
        alpha_i = _gf_exp[i]
        next_g = [0] * (len(g) + 1)
        for j in range(len(g)):
            next_g[j] ^= g[j]
            next_g[j + 1] ^= _gf_mul(g[j], alpha_i)
        g = next_g

    ecc = [0] * _NUM_ECC_CW
    for byte in data:
        feedback = byte ^ ecc[0]
        ecc = ecc[1:] + [0]
        if feedback != 0:
            for i in range(_NUM_ECC_CW):
                ecc[i] ^= _gf_mul(g[i + 1], feedback)
    return ecc


def encode_ascii_codewords(text: str) -> list[int] | None:
    """Enkodiert einen 4-stelligen Code in 3 ASCII-Codewörter (Ziffernpaare kompakt, Rest mit Padding 129)."""
    if not text or len(text) != REQUIRED_LENGTH:
        return None
    text = text.upper()
    codewords = []
    i = 0
    while i < len(text):
        c = text[i]
        if c not in ALLOWED_CHARS:
            return None
        if c.isdigit() and i + 1 < len(text) and text[i + 1].isdigit():
            codewords.append(130 + int(c) * 10 + int(text[i + 1]))
            i += 2
        else:
            codewords.append(ord(c) + 1)
            i += 1
    while len(codewords) < _NUM_DATA_CW:
        codewords.append(129)
    if len(codewords) != _NUM_DATA_CW:
        return None
    return codewords


@lru_cache(maxsize=1)
def _placement_map() -> tuple[tuple, ...]:
    """Utah-Placement für den 8x8-Datenbereich: (Codewort-Index, Bit-Index) je Modul oder None."""
    nrow = ncol = _DATA_SIZE
    grid = [[None] * ncol for _ in range(nrow)]

    def place_bit(r, c, cw_idx, bit_idx):
        if r < 0:
            r += nrow
            c += 4 - ((nrow + 4) % 8)
        if c < 0:
            c += ncol
            r += 4 - ((ncol + 4) % 8)
        if 0 <= r < nrow and 0 <= c < ncol and grid[r][c] is None:
            grid[r][c] = (cw_idx, bit_idx)

    def place_utah(r, c, cw_idx):
        place_bit(r - 2, c - 2, cw_idx, 1)
        place_bit(r - 2, c - 1, cw_idx, 2)
        place_bit(r - 1, c - 2, cw_idx, 3)
        place_bit(r - 1, c - 1, cw_idx, 4)
        place_bit(r - 1, c,     cw_idx, 5)
        place_bit(r,     c - 2, cw_idx, 6)
        place_bit(r,     c - 1, cw_idx, 7)
        place_bit(r,     c,     cw_idx, 8)

    def place_corner1(cw_idx):
        place_bit(nrow - 1, 0, cw_idx, 1)
        place_bit(nrow - 1, 1, cw_idx, 2)
        place_bit(nrow - 1, 2, cw_idx, 3)
        place_bit(0, ncol - 2, cw_idx, 4)
        place_bit(0, ncol - 1, cw_idx, 5)
        place_bit(1, ncol - 1, cw_idx, 6)
        place_bit(2, ncol - 1, cw_idx, 7)
        place_bit(3, ncol - 1, cw_idx, 8)

    r, c = 4, 0
    idx = 0
    while True:
        if r == nrow and c == 0:
            place_corner1(idx)
            idx += 1
            r -= 2
            c += 2
        else:
            # Diagonal nach oben rechts
            while True:
                if r < nrow and c >= 0 and grid[r][c] is None:
                    place_utah(r, c, idx)
                    idx += 1
                r -= 2
                c += 2
                if not (r >= 0 and c < ncol):
                    break
            r += 1
            c += 3
            # Diagonal nach unten links
            while True:
                if r >= 0 and c < ncol and grid[r][c] is None:
                    place_utah(r, c, idx)
                    idx += 1
                r += 2
                c -= 2
                if not (r < nrow and c >= 0):
                    break
            r += 3
            c += 1

        if not (r < nrow or c < ncol):
            break

    return tuple(tuple(row) for row in grid)


def frame_grid() -> np.ndarray:
    """10x10-Gitter mit L-Finder (links/unten schwarz) und Timing-Muster (oben/rechts), Datenbereich weiß."""
    full = np.ones((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    full[:, 0] = 0
    full[GRID_SIZE - 1, :] = 0
    for c in range(GRID_SIZE):
        full[0, c] = 0 if c % 2 == 0 else 1
    for r in range(GRID_SIZE):
        full[r, GRID_SIZE - 1] = 1 if r % 2 == 0 else 0
    full[GRID_SIZE - 1, GRID_SIZE - 1] = 0
    return full


def generate_reference_grid(text: str) -> np.ndarray | None:
    """Fehlerfreies 10x10-Referenzgitter eines Codes (0 = schwarz, 1 = weiß) oder None."""
    data_cw = encode_ascii_codewords(text)
    if data_cw is None:
        return None
    all_cw = data_cw + compute_rs_ecc(data_cw)

    placement = _placement_map()
    data_grid = np.ones((_DATA_SIZE, _DATA_SIZE), dtype=np.uint8)
    for r in range(_DATA_SIZE):
        for c in range(_DATA_SIZE):
            entry = placement[r][c]
            if entry is not None:
                cw_idx, bit_idx = entry
                if cw_idx < len(all_cw):
                    bit_val = (all_cw[cw_idx] >> (8 - bit_idx)) & 1
                    data_grid[r, c] = 0 if bit_val else 1

    full = frame_grid()
    full[1:9, 1:9] = data_grid
    return full


@lru_cache(maxsize=512)
def cached_reference_grid(text: str) -> np.ndarray | None:
    return generate_reference_grid(text)


@lru_cache(maxsize=1)
def all_code_grids() -> tuple[list[str], np.ndarray]:
    """Alle 4.000 gültigen Codes (A000-W999) und ihre flachen Referenzgitter als Matrix (4000, 100)."""
    codes = [f"{prefix}{num:03d}" for prefix in "ABPW" for num in range(1000)]
    rows = []
    for code in codes:
        grid = generate_reference_grid(code)
        rows.append(grid.flatten() if grid is not None else np.ones(GRID_SIZE * GRID_SIZE, dtype=np.uint8))
    matrix = np.array(rows, dtype=np.uint8)
    logger.info(f"4.000-Code DataMatrix Vektor-Datenbank erfolgreich vorberechnet ({matrix.shape}).")
    return codes, matrix
