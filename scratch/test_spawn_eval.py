import os
import sys
import time
import cv2
import json
import glob
import multiprocessing as mp

app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, app_dir)

import scanner

def worker_scan(crop):
    return scanner.scan(crop)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    images_dir = os.path.join(app_dir, "training_data")
    files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))[45:65] # 046 to 065
    
    print(f"Testing spawn worker on {len(files)} images (046 to 065)...", flush=True)
    
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=1) as pool:
        for i, img_path in enumerate(files, 46):
            fn = os.path.basename(img_path)
            img = cv2.imread(img_path)
            t0 = time.time()
            res = pool.apply(worker_scan, (img,))
            dur = int((time.time() - t0) * 1000)
            print(f"[{i}/65] {fn} -> {res.get('result')} ({res.get('method')}, {dur}ms)", flush=True)

    print("ALL 046-065 COMPLETED CLEANLY!", flush=True)
