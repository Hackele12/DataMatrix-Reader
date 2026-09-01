"""Debug: Was gibt _read_ocr_with_status auf gamma 0.3 zurück?"""
import os, sys, cv2, json, numpy as np
sys.stdout = open("debug_gamma_output.txt", "w", encoding="utf-8")

import scanner

img_name = "SCN-20260820-130950-0021.jpg"
img = cv2.imread(os.path.join("training_data", img_name))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Gamma 0.3
lut = np.array([((i/255.0)**0.3)*255 for i in range(256)]).astype("uint8")
bright = cv2.LUT(gray, lut)
print(f"Bild: {img_name}, Shape gray: {gray.shape}, Shape bright: {bright.shape}")
print(f"Gray: mean={gray.mean():.1f}, min={gray.min()}, max={gray.max()}")
print(f"Bright: mean={bright.mean():.1f}, min={bright.min()}, max={bright.max()}")

# Direkt _read_ocr_with_status
print("\n--- _read_ocr_with_status(bright) ---")
res = scanner._read_ocr_with_status(bright)
print(f"status={res.get('status')}, text={res.get('text')}, conf={res.get('confidence'):.2f}")
print(f"partial={res.get('partial_display')}, raw={res.get('raw_candidate')}")

# Direkt scan(bright) als 3-Kanal
print("\n--- scan(bright als BGR) ---")
bright_bgr = cv2.cvtColor(bright, cv2.COLOR_GRAY2BGR)
scan_res = scanner.scan(bright_bgr)
print(f"success={scan_res.get('success')}, result={scan_res.get('result')}, method={scan_res.get('method')}, conf={scan_res.get('confidence')}")

# Was passiert wenn ich scan() mit dem Original aufrufe?
print("\n--- scan(original) ---")
scan_orig = scanner.scan(img)
print(f"success={scan_orig.get('success')}, result={scan_orig.get('result')}, method={scan_orig.get('method')}, conf={scan_orig.get('confidence')}")

sys.stdout.close()
sys.stdout = sys.__stdout__
print("Debug fertig -> debug_gamma_output.txt")
