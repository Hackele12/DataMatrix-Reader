"""
DataMatrix Generator Script using pyStrich
Erstellt Schwarz-Weiß DataMatrix-Codes als hochkomprimierte PNG- oder SVG-Dateien.
"""

import sys
from pystrich.datamatrix import DataMatrixEncoder

def generate_datamatrix(data: str, output_path: str = "datamatrix.png", cellsize: int = 10):
    """
    Generiert einen DataMatrix-Code (Schwarz auf Weiß) und speichert ihn als komprimiertes PNG oder SVG.
    
    :param data: Text oder Daten, die im DataMatrix-Code kodiert werden sollen.
    :param output_path: Zielpfad für das Bild (.png oder .svg).
    :param cellsize: Größe/Skalierung eines einzelnen DataMatrix-Quadrats in Pixeln.
    """
    encoder = DataMatrixEncoder(data)
    
    if output_path.lower().endswith(".svg"):
        encoder.save_svg(output_path, cellsize=cellsize)
        print(f"DataMatrix SVG erfolgreich gespeichert unter: {output_path}")
    else:
        # Als PIL Image abrufen und als 1-Bit Monochrom-PNG extrem komprimiert speichern
        img = encoder.get_pilimage(cellsize=cellsize).convert("1")
        img.save(output_path, optimize=True)
        print(f"DataMatrix PNG (Schwarz/Weiß, komprimiert) erfolgreich gespeichert unter: {output_path}")

if __name__ == "__main__":
    # Beispieltext zum Testen
    text = sys.argv[1] if len(sys.argv) > 1 else "DATAMATRIX-CODE-12345"
    generate_datamatrix(text, "datamatrix_sample.png", cellsize=10)
