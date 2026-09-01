import os
import json

APP_DIR = r"c:\Users\kremidas\Documents\DataDetector"
GT_PATH = os.path.join(APP_DIR, "ground_truth.json")
SUMMARY_PATH = os.path.join(APP_DIR, "benchmarks_split_results", "combined_batch_benchmark_summary.json")

def main():
    with open(GT_PATH, "r", encoding="utf-8") as f:
        gt_map = json.load(f)

    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    all_details = summary_data.get("all_details", [])
    
    late_scans = []
    false_positive_horde_saves = []

    for item in all_details:
        fn = item.get("filename")
        duration = item.get("duration_ms", 0)
        success = item.get("success", False)
        result_code = item.get("result_code", "")
        # Updated ground truth check
        expected_gt = gt_map.get(fn, item.get("expected_code", "?"))
        
        is_match = (success and result_code == expected_gt)
        is_late = (duration >= 6000)

        if is_late:
            entry = {
                "filename": fn,
                "duration_ms": duration,
                "success": success,
                "result_code": result_code,
                "expected_gt": expected_gt,
                "is_match": is_match,
                "method": item.get("method"),
                "confidence": item.get("confidence"),
                "dmtx_result": item.get("dmtx_result"),
                "ocr_result": item.get("ocr_result"),
                "fail_reason": item.get("fail_reason")
            }
            late_scans.append(entry)

            if success and not is_match:
                false_positive_horde_saves.append(entry)

    print("=" * 95)
    print("       AUDIT ANALYSE: SPÄTERKENNUNGEN (>6s) UND FALSCH ABGESPEICHERTE HORDEN-BILDER")
    print("=" * 95)
    print(f" Analysierte Bilder (bisheriger Benchmark): {len(all_details)}")
    print(f" Bilder mit Scandauer > 6.000 ms:            {len(late_scans)}")
    print(f" Davon erfolgreich ausgewertet:               {sum(1 for s in late_scans if s['success'])}")
    print(f" Davon FALSCH ERKANNT (GT Mismatch):          {len(false_positive_horde_saves)}")
    print("=" * 95 + "\n")

    if false_positive_horde_saves:
        print("-------------------------------------------------------------------------------------------")
        print("  [!] FALSCH ABGESPEICHERTE HORDEN-BILDER IN horden_db/ (>6s & GT-Mismatch):")
        print("-------------------------------------------------------------------------------------------")
        for item in false_positive_horde_saves:
            print(f"  • Datei: {item['filename']:<10} | Dauer: {item['duration_ms']:<6} ms | Methode: {item['method']:<12}")
            print(f"    Erkannt: '{item['result_code']}'  vs  Soll (GT): '{item['expected_gt']}'")
            print(f"    Konfidenz: {item['confidence']} | OCR-Text: '{item['ocr_result']}' | DMTX: '{item['dmtx_result']}'")
            print("  -----------------------------------------------------------------------------------------")
    else:
        print("✓ Keine falsch abgespeicherten Hordenbilder in den vorhandenen Late-Scans gefunden!")

    # Write output log
    out_path = os.path.join(APP_DIR, "benchmarks_split_results", "late_scans_mismatches_log.json")
    report = {
        "total_audited": len(all_details),
        "total_late_scans": len(late_scans),
        "total_false_positive_horde_saves": len(false_positive_horde_saves),
        "false_positive_horde_saves": false_positive_horde_saves,
        "all_late_scans": late_scans
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\nBericht gespeichert unter: {out_path}")

if __name__ == "__main__":
    main()
