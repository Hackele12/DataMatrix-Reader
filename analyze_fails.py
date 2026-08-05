import json

d = json.load(open("benchmark_run_tmpl_match.json"))
fails = [r for r in d["details"] if not r["success"]]
print(f"Anzahl Fehlbilder: {len(fails)}\n")
print(f"{'#':<4} {'Dateiname':<45} {'GT':<6} {'OCR':<8} {'Fehlergrund':<25} {'Pfad'}")
print("=" * 160)
for r in fails:
    ocr = r.get('ocr_partial_text') or '—'
    print(f"{r['index']:<4} {r['filename']:<45} {r['expected_code']:<6} {ocr:<8} {r['fail_reason']:<25} {r['full_path']}")
