"""
create_v16_release.py — Packaging Script for DataDetector_v16_Release.zip
"""

import os
import zipfile
import time

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_ZIP = os.path.join(PROJECT_DIR, "DataDetector_v16_Release.zip")

EXCLUDE_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".vscode", ".idea"}
EXCLUDE_EXTENSIONS = {".zip", ".pyc"}
EXCLUDE_FILES = {"app_debug.log", "vision_network_main.log"}

def should_include(rel_path: str) -> bool:
    parts = rel_path.replace("\\", "/").split("/")
    
    # Check directory exclusion
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
    print(f"Creating release archive: {OUTPUT_ZIP} ...")
    t0 = time.time()
    
    added_count = 0
    total_uncompressed = 0
    
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for root, dirs, files in os.walk(PROJECT_DIR):
            # Modify dirs in-place to skip excluded directories
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, PROJECT_DIR)
                
                # Don't include the output zip itself
                if os.path.abspath(abs_path) == os.path.abspath(OUTPUT_ZIP):
                    continue
                    
                if should_include(rel_path):
                    z.write(abs_path, rel_path)
                    added_count += 1
                    total_uncompressed += os.path.getsize(abs_path)
                    if added_count % 500 == 0:
                        print(f"  Added {added_count} files ({total_uncompressed / 1e6:.1f} MB)...")
                        
    elapsed = time.time() - t0
    zip_size_mb = os.path.getsize(OUTPUT_ZIP) / (1024 * 1024)
    raw_size_mb = total_uncompressed / (1024 * 1024)
    
    print("\n--- Release Build Complete ---")
    print(f"Archive:            {OUTPUT_ZIP}")
    print(f"Total files:        {added_count:,}")
    print(f"Uncompressed size: {raw_size_mb:.2f} MB")
    print(f"Compressed size:   {zip_size_mb:.2f} MB")
    print(f"Time taken:        {elapsed:.1f} seconds")

if __name__ == "__main__":
    create_release()
