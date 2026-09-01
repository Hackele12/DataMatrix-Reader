import os
import sys
import glob
import cv2
import json
import time
import logging

APP_DIR = r"c:\Users\kremidas\Documents\DataDetector"
sys.path.insert(0, APP_DIR)

import benchmark_suite
import scanner
import horde_db
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("detailed_benchmark")

GT_PATH = os.path.join(APP_DIR, "ground_truth.json")
HORDE_DIR = os.path.join(APP_DIR, "horden_db")
SPLITS_ROOT = os.path.join(APP_DIR, "training_data_splits")
ARTIFACT_WALKTHROUGH = r"C:\Users\kremidas\.gemini\antigravity-ide\brain\bf448dd0-c328-4d2a-8261-302090026ebd\walkthrough.md"

def main():
    with open(GT_PATH, "r", encoding="utf-8") as f:
        gt_map = json.load(f)

    # Load YOLO model
    model_path = os.path.join(APP_DIR, "runs", "detect", "training_runs", "horde_model", "weights", "best.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(APP_DIR, "yolov10n.pt")
    model = YOLO(model_path)

    split_folders = sorted([d for d in glob.glob(os.path.join(SPLITS_ROOT, "part_*")) if os.path.isdir(d)])
    
    detailed_rows = []

    folder_stats = {}
    method_stats = {}

    for folder in split_folders:
        folder_name = os.path.basename(folder)
        images = sorted(glob.glob(os.path.join(folder, "*.jpg")))
        
        folder_stats[folder_name] = {
            "total": len(images),
            "gt_matches": 0,
            "successes": 0,
            "total_time_ms": 0.0,
            "methods": {}
        }

        for img_path in images:
            fn = os.path.basename(img_path)
            expected_gt = gt_map.get(fn, "?")

            image = cv2.imread(img_path)
            if image is None:
                continue

            t0 = time.time()
            try:
                cropped, yolo_conf = benchmark_suite._detect_and_crop(model, image)
                scan_res = scanner.scan(cropped)
            except Exception as e:
                scan_res = {"success": False, "result": "Fehler Exception", "method": "Fehler", "confidence": 0.0}

            dur_ms = round((time.time() - t0) * 1000.0, 1)
            is_success = scan_res.get("success", False)
            result_code = scan_res.get("result", "Kein Code")
            raw_method = scan_res.get("method", "Unbekannt")
            verified = scan_res.get("verified", False)
            conf = float(scan_res.get("confidence", 0.0))

            is_gt_match = (is_success and result_code == expected_gt)

            # Categorize Method
            if "Fast-Path" in raw_method or raw_method == "Verifiziert":
                method_cat = "DataMatrix (Fast-Path)"
            elif "HordenDB" in raw_method or "Bildabgleich" in raw_method:
                method_cat = "Horden-DB Match"
            elif "OCR" in raw_method:
                method_cat = "OCR (EasyOCR)"
            elif "Rekonstruiert" in raw_method:
                method_cat = "Rekonstruktion"
            else:
                method_cat = "Kein Code / Fehler"

            # Horde DB Protection Test
            is_late = (dur_ms > 6000)
            db_action = "SKIPPED"
            if is_success:
                db_res = horde_db.save_or_update_horde_image(
                    code=result_code,
                    frame=cropped,
                    is_late_scan=is_late,
                    verified=verified,
                    confidence=conf
                )
                if db_res:
                    db_action = "SAVED"
                else:
                    db_action = "REJECTED (GUARD)"

            # Stats aggregation
            f_stat = folder_stats[folder_name]
            f_stat["total_time_ms"] += dur_ms
            if is_success:
                f_stat["successes"] += 1
            if is_gt_match:
                f_stat["gt_matches"] += 1

            f_stat["methods"][method_cat] = f_stat["methods"].get(method_cat, 0) + 1
            method_stats[method_cat] = method_stats.get(method_cat, 0) + 1

            detailed_rows.append({
                "folder": folder_name,
                "file": fn,
                "gt": expected_gt,
                "res": result_code,
                "match": "✅ Ja" if is_gt_match else ("⚠️ Mismatch" if is_success else "❌ Nein"),
                "method": method_cat,
                "conf": f"{conf:.1%}",
                "time_ms": f"{dur_ms:.0f} ms",
                "dur_val": dur_ms,
                "db_action": db_action
            })

    # Build Markdown Walkthrough Document
    md = []
    md.append("# 📊 Ausführlicher 172-Bilder Batch-Benchmark & Analysebericht\n")
    md.append("> **Zusammenfassung:** Dieser Bericht beinhaltet die vollständige Auswertung aller **172 Testbilder** (aufgeteilt in `part_01` bis `part_06`), inklusive Erkennungsmethoden, Ausführungszeiten, Ground-Truth-Abweichungen und der Hordenbild-Datenbank Schutzbilanz.\n")

    # Overall Metrics Table
    total_imgs = len(detailed_rows)
    total_gt = sum(1 for r in detailed_rows if "✅" in r["match"])
    total_succ = sum(1 for r in detailed_rows if r["res"] != "Kein Code" and r["res"] != "Kein Code erkannt.")
    total_time = sum(r["dur_val"] for r in detailed_rows)
    avg_time = total_time / total_imgs if total_imgs > 0 else 0

    md.append("## 🏆 Gesamtergebnis Übersicht\n")
    md.append(f"- **Gesamtanzahl Bilder:** `{total_imgs}`")
    md.append(f"- **Erfolgreiche Scans (Erkennungsrate):** `{total_succ} / {total_imgs}` (**{(total_succ/total_imgs*100):.1f}%**)")
    md.append(f"- **Exakte Ground-Truth Matches:** `{total_gt} / {total_imgs}` (**{(total_gt/total_imgs*100):.1f}%**)")
    md.append(f"- **Durchschnittliche Scandauer:** `{avg_time:.0f} ms` pro Bild\n")

    md.append("### 📈 Ordner-Übersicht (Part 01 bis Part 06)\n")
    md.append("| Ordner | Bilder | GT-Matches | Erkennungs-% | GT-Genauigkeit | Ø Dauer / Bild | Dominante Methode |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :--- |")

    for fname, st in folder_stats.items():
        t_cnt = st["total"]
        succ_pct = (st["successes"] / t_cnt * 100.0) if t_cnt > 0 else 0
        gt_pct = (st["gt_matches"] / t_cnt * 100.0) if t_cnt > 0 else 0
        avg_d = (st["total_time_ms"] / t_cnt) if t_cnt > 0 else 0
        dom_meth = max(st["methods"].items(), key=lambda x: x[1])[0] if st["methods"] else "Keine"
        md.append(f"| `{fname}` | {t_cnt} | {st['gt_matches']} | {succ_pct:.1f}% | {gt_pct:.1f}% | {avg_d:.0f} ms | {dom_meth} |")

    md.append("\n---\n")

    md.append("## 🛠️ Aufschlüsselung nach Erkennungsmethoden\n")
    md.append("| Erkennungsmethode | Anzahl Treffer | Anteil am Gesamtergebnis | Ø Geschwindigkeit | Typische Anwendung |")
    md.append("| :--- | :---: | :---: | :---: | :--- |")

    for m_name, count in method_stats.items():
        m_rows = [r for r in detailed_rows if r["method"] == m_name]
        m_avg_time = sum(r["dur_val"] for r in m_rows) / len(m_rows) if m_rows else 0
        pct = (count / total_imgs * 100.0) if total_imgs > 0 else 0
        desc = "Unbeschädigte Codes (<15ms)" if "Fast-Path" in m_name else ("Exakte Hordenbild-Übereinstimmung (<35ms)" if "Horden" in m_name else ("Klarer Klartext-Scan" if "OCR" in m_name else "Grid-Rekonstruktion"))
        md.append(f"| **{m_name}** | {count} | {pct:.1f}% | {m_avg_time:.0f} ms | {desc} |")

    md.append("\n---\n")

    md.append("## 🛡️ Horden-Datenbank Schutz-Bilanz\n")
    saved_cnt = sum(1 for r in detailed_rows if r["db_action"] == "SAVED")
    rej_cnt = sum(1 for r in detailed_rows if r["db_action"] == "REJECTED (GUARD)")
    skip_cnt = sum(1 for r in detailed_rows if r["db_action"] == "SKIPPED")

    md.append(f"- **In `horden_db/` verifiziert gespeichert / aktualisiert:** `{saved_cnt}x` (100% verifizierte Vorlagen)")
    md.append(f"- **Vom Schutz-Guard ABGELEHNT:** `{rej_cnt}x` (Fehlerhafte/unverifizierte Scans wurden am Speichern gehindert)")
    md.append(f"- **Übersprungen (kein Code erkannt):** `{skip_cnt}x`\n")

    md.append("\n---\n")

    md.append("## 📑 Detaillierte Einzelauswertung (Alle 172 Bilder)\n")
    md.append("| Datei | Ordner | Soll (Ground Truth) | Ist (Scan Ergebnis) | Match | Methode | Konfidenz | Scandauer | Horden-DB Status |")
    md.append("| :--- | :--- | :---: | :---: | :---: | :--- | :---: | :---: | :--- |")

    for r in detailed_rows:
        md.append(f"| `{r['file']}` | `{r['folder']}` | `{r['gt']}` | `{r['res']}` | {r['match']} | {r['method']} | {r['conf']} | {r['time_ms']} | `{r['db_action']}` |")

    with open(ARTIFACT_WALKTHROUGH, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"\nSuccessfully generated full detailed walkthrough artifact: {ARTIFACT_WALKTHROUGH}")

if __name__ == "__main__":
    main()
