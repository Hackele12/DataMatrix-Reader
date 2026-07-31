import os
import sys
import zipfile

def create_zip():
    zip_filename = "DataDetector_v5_Release.zip"
    if os.path.exists(zip_filename):
        os.remove(zip_filename)

    exclude_dirs = {".git", ".venv", "__pycache__", ".pytest_cache", ".vscode", ".idea"}
    exclude_files = {zip_filename, ".DS_Store"}

    print(f"Erstelle Zip-Archiv '{zip_filename}'...")
    count = 0

    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk("."):
            # Filter excluded directories in place
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                if file in exclude_files:
                    continue
                if file.endswith(".zip") and file != zip_filename:
                    continue
                    
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, ".")
                zf.write(full_path, rel_path)
                count += 1

    print(f"Zip-Archiv mit {count} Dateien erfolgreich erstellt.")
    
    # Validieren
    with zipfile.ZipFile(zip_filename, "r") as zf:
        bad_file = zf.testzip()
        if bad_file is None:
            print("Zip-Archiv Validierung ERFOLGREICH (100% intakt).")
        else:
            print(f"FEHLER in Datei: {bad_file}")

if __name__ == "__main__":
    create_zip()
