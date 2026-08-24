"""
Batch Generator Script for DataMatrix Codes (Horden-Codes)
Generiert alle möglichen Code-Kombinationen (A-Z, 000-999) als Schwarz-Weiß PNGs im Ordner 'generated_codes'.
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pystrich.datamatrix import DataMatrixEncoder

OUTPUT_DIR = "generated_codes"
CELLSIZE = 10

def generate_single_code(code: str) -> str:
    """Generiert ein einzelnes DataMatrix Monochrom-PNG."""
    encoder = DataMatrixEncoder(code)
    img = encoder.get_pilimage(cellsize=CELLSIZE).convert("1")
    file_path = os.path.join(OUTPUT_DIR, f"{code}.png")
    img.save(file_path, optimize=True)
    return file_path

def generate_all_codes(prefixes=None, start_num=0, end_num=999):
    """
    Generiert alle Code-Kombinationen parallel.
    Standard: Alle 26 Buchstaben (A-Z) mit 3-stelligen Ziffern (000-999) -> 26.000 Codes!
    """
    if prefixes is None:
        # Beschränkt auf A, B, P, W
        prefixes = ["A", "B", "P", "W"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Erstelle Liste aller auszuführenden Codes
    codes = [f"{prefix}{num:03d}" for prefix in prefixes for num in range(start_num, end_num + 1)]
    total_count = len(codes)
    
    print(f"Starte Generierung von {total_count} DataMatrix-Codes in '{OUTPUT_DIR}'...")
    t0 = time.time()
    
    # Parallelisierung mit ThreadPoolExecutor
    completed = 0
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(generate_single_code, code) for code in codes]
        for future in futures:
            future.result()
            completed += 1
            if completed % 5000 == 0 or completed == total_count:
                elapsed = time.time() - t0
                print(f"Fortschritt: {completed}/{total_count} Codes generiert ({elapsed:.1f}s)")

    duration = time.time() - t0
    print(f"\nFERTIG! {completed} DataMatrix-Codes in {duration:.2f} Sekunden in '{OUTPUT_DIR}' generiert.")

if __name__ == "__main__":
    generate_all_codes()
