"""
DataMatrix ECC200-Codec für 10x10-Symbole (4-stellige Horden-Codes):
ASCII-Codewörter, Reed-Solomon GF(256) und Utah-Placement nach ISO/IEC 16022.
Liefert die fehlerfreien Referenzgitter für den Abgleich mit beobachteten Gittern und
dekodiert beobachtete Codewörter (Reed-Solomon mit Fehlern und Löschungen).
"""

import logging
from functools import lru_cache
from itertools import combinations

import numpy as np

from .config import ALLOWED_CHARS, REQUIRED_LENGTH, VALID_PREFIXES

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
    codes = [f"{prefix}{num:03d}" for prefix in sorted(VALID_PREFIXES) for num in range(1000)]
    rows = []
    for code in codes:
        grid = generate_reference_grid(code)
        rows.append(grid.flatten() if grid is not None else np.ones(GRID_SIZE * GRID_SIZE, dtype=np.uint8))
    matrix = np.array(rows, dtype=np.uint8)
    logger.info(f"4.000-Code DataMatrix Vektor-Datenbank erfolgreich vorberechnet ({matrix.shape}).")
    return codes, matrix


# --------------------------------------------------------------------------- #
#  Dekodierung beobachteter Symbole                                            #
# --------------------------------------------------------------------------- #

NUM_CODEWORDS = _NUM_DATA_CW + _NUM_ECC_CW


def _gf_inv(a: int) -> int:
    return _gf_exp[255 - _gf_log[a]]


def _codeword_locator(position: int) -> int:
    """Fehlerlokator X = alpha^(n-1-position); das erste Codewort ist der höchste Polynomgrad."""
    return _gf_exp[NUM_CODEWORDS - 1 - position]


def rs_syndromes(codewords: list[int]) -> list[int]:
    """Syndrome S_1..S_5 (alle 0 = gültiges Codewort)."""
    syndromes = []
    for j in range(1, _NUM_ECC_CW + 1):
        s = 0
        for pos, value in enumerate(codewords):
            if value:
                s ^= _gf_mul(value, _gf_exp[(j * (NUM_CODEWORDS - 1 - pos)) % 255])
        syndromes.append(s)
    return syndromes


def _solve_gf(matrix: list[list[int]], rhs: list[int]) -> list[int] | None:
    """Löst ein (überbestimmtes) lineares Gleichungssystem über GF(256); None, wenn es widersprüchlich ist."""
    rows = [row[:] + [value] for row, value in zip(matrix, rhs)]
    n_unknowns = len(matrix[0]) if matrix else 0
    pivot_row = 0
    pivot_cols = []
    for col in range(n_unknowns):
        pivot = next((r for r in range(pivot_row, len(rows)) if rows[r][col]), None)
        if pivot is None:
            return None  # singulär: Lokatoren sind verschieden, kommt praktisch nicht vor
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        inv = _gf_inv(rows[pivot_row][col])
        rows[pivot_row] = [_gf_mul(v, inv) for v in rows[pivot_row]]
        for r in range(len(rows)):
            if r != pivot_row and rows[r][col]:
                factor = rows[r][col]
                rows[r] = [v ^ _gf_mul(factor, p) for v, p in zip(rows[r], rows[pivot_row])]
        pivot_cols.append(col)
        pivot_row += 1
    if any(rows[r][-1] for r in range(pivot_row, len(rows))):
        return None
    return [rows[i][-1] for i in range(n_unknowns)]


def rs_decode(codewords: list[int], erasures=(), max_errors: int | None = None) -> tuple[list[int], int] | None:
    """
    Reed-Solomon-Dekodierung (8 Codewörter, 5 ECC) mit Fehlern und Löschungen: 2·Fehler + Löschungen ≤ 5.
    Die kleinste Fehlerzahl gewinnt; innerhalb dieser Schranke ist die Lösung eindeutig (Mindestabstand 6).

    Returns:
        (korrigierte Codewörter, Anzahl korrigierter Fehler ohne Löschungen) oder None.
    """
    if len(codewords) != NUM_CODEWORDS:
        return None
    erasures = sorted(set(int(p) for p in erasures))
    if len(erasures) > _NUM_ECC_CW:
        return None
    syndromes = rs_syndromes(codewords)
    if not any(syndromes):
        return list(codewords), 0

    limit = (_NUM_ECC_CW - len(erasures)) // 2
    if max_errors is not None:
        limit = min(limit, max_errors)
    candidates = [p for p in range(NUM_CODEWORDS) if p not in erasures]
    for n_errors in range(limit + 1):
        for error_positions in combinations(candidates, n_errors):
            positions = sorted(erasures + list(error_positions))
            if not positions:
                continue
            locators = [_codeword_locator(p) for p in positions]
            matrix = [[_gf_exp[(_gf_log[x] * j) % 255] for x in locators] for j in range(1, _NUM_ECC_CW + 1)]
            values = _solve_gf(matrix, syndromes)
            if values is None:
                continue
            if any(values[positions.index(p)] == 0 for p in error_positions):
                continue  # kein echter Fehler an dieser Stelle → eine kleinere Fehlerzahl passt bereits
            corrected = list(codewords)
            for p, v in zip(positions, values):
                corrected[p] ^= v
            if not any(rs_syndromes(corrected)):
                return corrected, n_errors
    return None


def decode_ascii_codewords(data: list[int]) -> str | None:
    """ASCII-Datencodewörter → Text (1-128 Zeichen, 130-229 Ziffernpaar, 129 Padding); sonst None."""
    chars = []
    for value in data:
        if value == 129:
            break
        if 1 <= value <= 128:
            chars.append(chr(value - 1))
        elif 130 <= value <= 229:
            chars.append(f"{value - 130:02d}")
        else:
            return None  # Umschaltcodes (C40, Base256 …) kommen im Horden-Format nicht vor
    return "".join(chars)


@lru_cache(maxsize=1)
def data_module_codewords() -> tuple[np.ndarray, np.ndarray]:
    """Für die 64 Datenmodule (zeilenweise im inneren 8x8-Bereich): Codewort-Index und Bitwertigkeit."""
    placement = _placement_map()
    cw_index = np.zeros(_DATA_SIZE * _DATA_SIZE, dtype=np.int64)
    bit_weight = np.zeros(_DATA_SIZE * _DATA_SIZE, dtype=np.int64)
    for r in range(_DATA_SIZE):
        for c in range(_DATA_SIZE):
            cw_idx, bit_idx = placement[r][c]
            cw_index[r * _DATA_SIZE + c] = cw_idx
            bit_weight[r * _DATA_SIZE + c] = 1 << (8 - bit_idx)
    return cw_index, bit_weight


def codewords_from_dark_modules(dark: np.ndarray) -> list[int]:
    """64 Datenmodule (True = dunkel = Bit 1) → 8 Codewörter."""
    cw_index, bit_weight = data_module_codewords()
    codewords = np.zeros(NUM_CODEWORDS, dtype=np.int64)
    np.add.at(codewords, cw_index, np.where(np.asarray(dark, dtype=bool), bit_weight, 0))
    return [int(v) for v in codewords]


@lru_cache(maxsize=1)
def codebook_data_patterns() -> tuple[list[str], np.ndarray]:
    """Codebuch für den Korrelationsabgleich: Datenmodule aller gültigen Codes, zentriert und normiert (1 = hell)."""
    codes, matrix = all_code_grids()
    data = matrix.reshape(-1, GRID_SIZE, GRID_SIZE)[:, 1:GRID_SIZE - 1, 1:GRID_SIZE - 1]
    data = data.reshape(len(codes), -1).astype(np.float32)
    data -= data.mean(axis=1, keepdims=True)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    return codes, data
