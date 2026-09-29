"""Horden-Code-Format ^[ABPW][0-9]{3}$: Validierung, OCR-Verwechslungskorrektur und Teilcode-Kandidaten."""

import logging

from .config import ALLOWED_CHARS, HORDEN_PATTERN, REQUIRED_LENGTH, VALID_PREFIXES

logger = logging.getLogger(__name__)

# Ziffer → Buchstabe für Position 0
DIGIT_TO_LETTER = {'8': 'B', '4': 'A', '9': 'P'}

# Visuell ähnliche Buchstaben → nächster gültiger Präfix (z. B. abgebleichte Codes: W wie V, B wie D)
FUZZY_PREFIX_MAP = {
    'V': 'W',   # V ↔ W  (sehr ähnliche Form)
    'U': 'W',   # U ↔ W  (offene Unterseite)
    'M': 'W',   # M = umgekehrtes W
    'Y': 'W',   # Y unterer Teil ähnelt V/W
    'N': 'W',   # N Diagonalstrich ähnelt W
    'D': 'B',   # D ↔ B  (runde rechte Seite)
    'R': 'P',   # R = P mit Bein
    'F': 'P',   # F ↔ P  (obere Hälfte ähnlich)
    'T': 'P',   # T oberer Balken ähnelt P
    'H': 'A',   # H ↔ A  (Querbalken zwischen Strichen)
    'K': 'A',   # K Diagonalstriche ähneln A
    'X': 'A',   # X konvergierende Linien wie A
    'L': 'A',   # L = A ohne Spitze (bei Serifen)
    'O': 'B',   # O rund wie B
    'C': 'B',   # C = offenes B/D
    'E': 'B',   # E ≈ B (horizontale Striche)
    'G': 'B',   # G ↔ B/D (runde Form)
    'S': 'B',   # S Kurven ähneln B
    'Q': 'B',   # Q ↔ O ↔ B
    'I': 'B',   # I mit Serifen ↔ B
    'J': 'B',   # J Bogen ähnelt B
    'Z': 'A',   # Z Diagonale ähnelt A
}

# Buchstabe → visuell naheliegendste Ziffer für Positionen 1-3 (alle 26 Buchstaben)
LETTER_TO_DIGIT = {
    'O': '0', 'D': '0', 'Q': '0',
    'I': '1', 'L': '1',
    'Z': '2',
    'S': '5',
    'G': '6',
    'B': '8',
    'A': '4',
    'P': '9',
    'C': '0',   # offener Kreis
    'E': '3',
    'F': '7',
    'H': '4',   # Querbalken
    'J': '0',   # oft eine linksseitig ausgebleichte 0
    'K': '4',
    'M': '0',   # breiter Kreis
    'N': '0',
    'R': '8',
    'T': '7',
    'U': '0',   # offener Bogen
    'V': '0',
    'W': '0',
    'X': '8',
    'Y': '9',
}


def is_valid_horden_code(code: str) -> bool:
    return bool(HORDEN_PATTERN.match(code))


def is_prefix_like(ch: str) -> bool:
    """True, wenn das Zeichen ein gültiger Präfix ist oder zu einem korrigiert werden kann."""
    return ch in VALID_PREFIXES or ch in FUZZY_PREFIX_MAP or ch in DIGIT_TO_LETTER


def clean_chars(text: str) -> str:
    """Großbuchstaben, nur erlaubte Zeichen."""
    return ''.join(c for c in text.upper() if c in ALLOWED_CHARS)


def dmx_text_to_code(text: str | None) -> str | None:
    """DataMatrix-Inhalt ist Reed-Solomon-geprüft und exakt: nur Formatprüfung, keine OCR-Normalisierung."""
    if not text:
        return None
    code = clean_chars(text)
    return code if HORDEN_PATTERN.match(code) else None


def normalize_ocr_confusions(text: str) -> str:
    """
    Normalisiert einen 4-stelligen OCR-Text auf das Horden-Format:
    Stelle 0 → Präfix (Ziffer über DIGIT_TO_LETTER, andere Buchstaben über FUZZY_PREFIX_MAP),
    Stellen 1-3 → Ziffern (LETTER_TO_DIGIT); eine 6 an Stelle 1 wird zur 0.
    """
    if not text or len(text) != 4:
        return text.upper() if text else text

    chars = list(text.upper())

    if chars[0] not in VALID_PREFIXES:
        if chars[0] in DIGIT_TO_LETTER:
            chars[0] = DIGIT_TO_LETTER[chars[0]]
        elif chars[0] in FUZZY_PREFIX_MAP:
            original = chars[0]
            chars[0] = FUZZY_PREFIX_MAP[original]
            logger.debug(f"Fuzzy-Präfix: '{original}' → '{chars[0]}' (visuell ähnlichster Buchstabe)")

    for i in range(1, 4):
        if chars[i] in LETTER_TO_DIGIT:
            chars[i] = LETTER_TO_DIGIT[chars[i]]

    # Auf dunklen/kontrastarmen Bildern liest EasyOCR die 0 an der 1. Ziffernstelle systematisch als 6 (W631 → W031).
    if chars[1] == '6':
        chars[1] = '0'

    return ''.join(chars)


def normalize_partial_3chars(text: str) -> tuple[str, bool]:
    """
    Normalisiert einen 3-stelligen Teilcode.

    Returns:
        (normalisierter Text, True wenn Stelle 0 ein (korrigierbarer) Präfix ist).
        Mit Präfix werden die Stellen 1-2, ohne Präfix alle 3 Stellen als Ziffern normalisiert.
    """
    if not text or len(text) != 3:
        return (text.upper() if text else text), False

    chars = list(text.upper())
    prefix_detected = False
    if chars[0] in VALID_PREFIXES:
        prefix_detected = True
    elif chars[0] in FUZZY_PREFIX_MAP:
        original = chars[0]
        chars[0] = FUZZY_PREFIX_MAP[original]
        prefix_detected = True
        logger.debug(f"Partial Fuzzy-Präfix: '{original}' → '{chars[0]}'")
    elif chars[0] in DIGIT_TO_LETTER:
        chars[0] = DIGIT_TO_LETTER[chars[0]]
        prefix_detected = True

    for i in range(1 if prefix_detected else 0, 3):
        if chars[i] in LETTER_TO_DIGIT:
            chars[i] = LETTER_TO_DIGIT[chars[i]]

    return ''.join(chars), prefix_detected


def clean_to_4chars(text: str) -> str | None:
    """Bereinigt und normalisiert OCR-Text; liefert den Code nur, wenn er dem Horden-Format entspricht."""
    if not text:
        return None
    clean = clean_chars(text)
    if len(clean) != REQUIRED_LENGTH:
        return None
    normalized = normalize_ocr_confusions(clean)
    return normalized if HORDEN_PATTERN.match(normalized) else None


def format_partial_display(readable_chars: str, missing_positions: list[int]) -> str:
    """Anzeige einer Teillesung mit '?' an den fehlenden Stellen (3 Ziffern ohne Präfix → '?' an Stelle 0)."""
    if not readable_chars:
        return "????"

    chars = ''.join(c for c in readable_chars if c in ALLOWED_CHARS)

    if len(chars) == 3 and not is_prefix_like(chars[0]):
        return "?" + chars

    if not missing_positions:
        return chars[:4] if len(chars) >= 4 else chars

    result = list("????")
    char_idx = 0
    for pos in range(4):
        if pos not in missing_positions and char_idx < len(chars):
            result[pos] = chars[char_idx]
            char_idx += 1

    display = ''.join(result)
    if display[0] in '0123456789' and len(chars) == 3:
        return "?" + chars
    return display


def generate_10_candidates_from_partial(ocr_partial: str) -> list[str]:
    """Alle formatgültigen 4-Zeichen-Codes zu einer Teillesung ('W03?' oder 3 erkannte Zeichen)."""
    if not ocr_partial:
        return []

    cands = []
    if len(ocr_partial) == 4 and '?' in ocr_partial:
        q_pos = ocr_partial.find('?')
        fillers = VALID_PREFIXES if q_pos == 0 else '0123456789'
        for filler in fillers:
            cand = ocr_partial[:q_pos] + filler + ocr_partial[q_pos + 1:]
            if is_valid_horden_code(cand):
                cands.append(cand)
        return cands

    partial_norm, prefix_detected = normalize_partial_3chars(ocr_partial)
    if prefix_detected:
        prefix, digits = partial_norm[0], partial_norm[1:]
        for insert_pos in range(3):
            for d in '0123456789':
                cand = prefix + digits[:insert_pos] + d + digits[insert_pos:]
                if is_valid_horden_code(cand) and cand not in cands:
                    cands.append(cand)
    else:
        for prefix in VALID_PREFIXES:
            cand = prefix + partial_norm
            if is_valid_horden_code(cand) and cand not in cands:
                cands.append(cand)
    return cands
