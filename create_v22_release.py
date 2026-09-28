"""create_v22_release.py — Packaging Script for DataDetector_v22_Release.zip."""

import os
import zipfile
import time

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RELEASES_DIR = os.path.join(PROJECT_DIR, "releases")
os.makedirs(RELEASES_DIR, exist_ok=True)

OUTPUT_ZIP = os.path.join(RELEASES_DIR, "DataDetector_v22_Release.zip")

EXCLUDE_DIRS = {
    ".git", ".venv", "__pycache__", ".pytest_cache", ".vscode", ".idea",
    "AppDevProjekt_KLIQ", "android", "build", "node_modules", ".gradle",
    "scratch", ".gemini", "releases", "temp_ls_export"
}
EXCLUDE_EXTENSIONS = {".zip", ".pyc"}
EXCLUDE_FILES = {"app_debug.log", "vision_network_main.log", "diagnose_output.txt", "test_5_output.txt", "debug_gamma_output.txt", "crash.log"}

def should_include(rel_path: str) -> bool:
    parts = rel_path.replace("\\", "/").split("/")
    
    for part in parts[:-1]:
        if part in EXCLUDE_DIRS:
            return False
            
    filename = parts[-1]
    if filename in EXCLUDE_FILES:
        return False
        
    ext = os.path.splitext(filename)[1].lower()
    if ext in EXCLUDE_EXTENSIONS:
        return False
        
    return True

def create_release():
    print(f"Creating v22 release archive: {OUTPUT_ZIP} ...")
    t0 = time.time()
    
    added_count = 0
    total_uncompressed = 0
    
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for root, dirs, files in os.walk(PROJECT_DIR):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, PROJECT_DIR)
                
                if os.path.abspath(abs_path) == os.path.abspath(OUTPUT_ZIP):
                    continue
                    
                if should_include(rel_path):
                    try:
                        z.write(abs_path, rel_path)
                        added_count += 1
                        total_uncompressed += os.path.getsize(abs_path)
                        if added_count % 500 == 0:
                            print(f"  Added {added_count} files ({total_uncompressed / 1e6:.1f} MB)...")
                    except Exception as e:
                        print(f"  [SKIP] Could not add {rel_path}: {e}")
                        
    elapsed = time.time() - t0
    zip_size_mb = os.path.getsize(OUTPUT_ZIP) / (1024 * 1024)
    raw_size_mb = total_uncompressed / (1024 * 1024)
    
    print("\n==============================================")
    print("      DATA DETECTOR RELEASE V22 COMPLETE      ")
    print("==============================================")
    print(f"Archive:            {OUTPUT_ZIP}")
    print(f"Total files:        {added_count:,}")
    print(f"Uncompressed size: {raw_size_mb:.2f} MB")
    print(f"Compressed size:   {zip_size_mb:.2f} MB")
    print(f"Time taken:        {elapsed:.1f} seconds")
    print("==============================================")

if __name__ == "__main__":
    create_release()
