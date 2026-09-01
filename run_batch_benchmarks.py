"""
run_batch_benchmarks.py - Split training dataset into chunks of ~30 images,
run benchmark_suite on each chunk, and aggregate total results.
"""

import os
import sys
import glob
import shutil
import json
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("batch_benchmark")

# Path definitions
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(APP_DIR, "training_data")
SPLITS_ROOT = os.path.join(APP_DIR, "training_data_splits")
OUTPUT_DIR = os.path.join(APP_DIR, "benchmarks_split_results")

BATCH_SIZE = 30


def prepare_split_folders(batch_size: int = BATCH_SIZE) -> list[str]:
    """Finds images in training_data and splits them into subfolders of ~batch_size images."""
    if not os.path.exists(SOURCE_DIR):
        raise FileNotFoundError(f"Source folder '{SOURCE_DIR}' not found!")

    image_extensions = ("*.jpg", "*.jpeg", "*.png")
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(SOURCE_DIR, ext)))
    image_paths = sorted(image_paths)

    total_images = len(image_paths)
    if total_images == 0:
        raise ValueError(f"No images found in '{SOURCE_DIR}'!")

    logger.info(f"Total training images found: {total_images}")

    # Ensure SPLITS_ROOT exists
    os.makedirs(SPLITS_ROOT, exist_ok=True)

    # Chunk images
    chunks = [image_paths[i:i + batch_size] for i in range(0, total_images, batch_size)]
    folder_paths = []

    for idx, chunk in enumerate(chunks, 1):
        folder_name = f"part_{idx:02d}"
        target_folder = os.path.join(SPLITS_ROOT, folder_name)
        os.makedirs(target_folder, exist_ok=True)

        for img_path in chunk:
            target_img = os.path.join(target_folder, os.path.basename(img_path))
            if not os.path.exists(target_img):
                shutil.copy2(img_path, target_folder)

        folder_paths.append(target_folder)
        logger.info(f"Split '{folder_name}' has {len(chunk)} images.")

    return folder_paths


def run_batch_benchmarks(folder_paths: list[str]) -> list[str]:
    """Runs benchmark_suite on each split folder via subprocess and saves report JSON files."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_paths = []
    python_exe = sys.executable

    for idx, folder in enumerate(folder_paths, 1):
        folder_name = os.path.basename(folder)
        output_filename = f"benchmark_{folder_name}.json"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        report_paths.append(output_path)

        if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
            logger.info(f"Report for {folder_name} already exists ({output_path}). Skipping rerun.")
            continue

        logger.info(f"\n==========================================")
        logger.info(f" Running Benchmark for Folder {idx}/{len(folder_paths)}: {folder_name}")
        logger.info(f"==========================================")

        cmd = [
            python_exe,
            os.path.join(APP_DIR, "benchmark_suite.py"),
            "--dirs", folder,
            "--output", output_path
        ]
        import subprocess
        res = subprocess.run(cmd, cwd=APP_DIR)
        if res.returncode != 0:
            logger.error(f"Benchmark for {folder_name} exited with code {res.returncode}")
        else:
            logger.info(f"Benchmark for {folder_name} finished successfully.")

    return report_paths


def aggregate_results(report_paths: list[str]) -> dict:
    """Combines statistics from all individual JSON benchmark reports."""
    aggregated = {
        "timestamp": datetime.now().isoformat(),
        "total_folders": len(report_paths),
        "total_images": 0,
        "total_successes": 0,
        "total_gt_matches": 0,
        "total_gt_evaluated": 0,
        "total_duration_ms": 0,
        "folder_summaries": [],
        "combined_ocr_partial_breakdown": {"4/4": 0, "3/4": 0, "2/4": 0, "1/4": 0, "0/4": 0},
        "combined_method_distribution": {},
        "combined_failure_categories": {},
        "all_details": []
    }

    for path in report_paths:
        if not os.path.exists(path):
            logger.warning(f"Report file missing: {path}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        summary = data.get("summary", {})
        details = data.get("details", [])
        folder_name = os.path.basename(path).replace("benchmark_", "").replace(".json", "")

        total_img = summary.get("total_images", 0)
        succ_cnt = summary.get("success_count", 0)
        succ_pct = summary.get("success_rate_pct", 0.0)
        acc_str = summary.get("accuracy_gt_count", "0/0")
        acc_pct = summary.get("accuracy_pct", 0.0)
        avg_dur = summary.get("avg_duration_ms", 0.0)

        # Parse GT match info
        gt_match_parts = acc_str.split("/")
        gt_matches = int(gt_match_parts[0]) if len(gt_match_parts) == 2 else 0
        gt_total = int(gt_match_parts[1]) if len(gt_match_parts) == 2 else 0

        aggregated["total_images"] += total_img
        aggregated["total_successes"] += succ_cnt
        aggregated["total_gt_matches"] += gt_matches
        aggregated["total_gt_evaluated"] += gt_total
        aggregated["total_duration_ms"] += (avg_dur * total_img)

        # Merge breakdown dicts
        for key, val in summary.get("ocr_partial_breakdown", {}).items():
            aggregated["combined_ocr_partial_breakdown"][key] = (
                aggregated["combined_ocr_partial_breakdown"].get(key, 0) + val
            )

        for key, val in summary.get("method_distribution", {}).items():
            aggregated["combined_method_distribution"][key] = (
                aggregated["combined_method_distribution"].get(key, 0) + val
            )

        for key, val in summary.get("failure_categories", {}).items():
            aggregated["combined_failure_categories"][key] = (
                aggregated["combined_failure_categories"].get(key, 0) + val
            )

        aggregated["folder_summaries"].append({
            "folder": folder_name,
            "total_images": total_img,
            "success_count": succ_cnt,
            "success_rate_pct": succ_pct,
            "gt_matches": gt_matches,
            "gt_total": gt_total,
            "accuracy_pct": acc_pct,
            "avg_duration_ms": avg_dur
        })

        aggregated["all_details"].extend(details)

    # Compute overall averages / percentages
    tot_img = aggregated["total_images"]
    tot_succ = aggregated["total_successes"]
    tot_gt_m = aggregated["total_gt_matches"]
    tot_gt_e = aggregated["total_gt_evaluated"]

    aggregated["overall_success_rate_pct"] = round((tot_succ / tot_img * 100), 1) if tot_img > 0 else 0.0
    aggregated["overall_accuracy_pct"] = round((tot_gt_m / tot_gt_e * 100), 1) if tot_gt_e > 0 else 0.0
    aggregated["overall_avg_duration_ms"] = round(aggregated["total_duration_ms"] / tot_img, 1) if tot_img > 0 else 0.0

    return aggregated


def print_aggregated_report(agg: dict):
    """Prints a clean tabular report of all individual folders and the combined grand total."""
    print("\n" + "=" * 100)
    print("                      BATCH BENCHMARK SUMMARY REPORT")
    print("=" * 100)

    header = f"{'Folder':<12} | {'Images':<8} | {'Successes':<10} | {'Success %':<10} | {'GT Accuracy':<12} | {'Avg Ms/Img':<10}"
    print(header)
    print("-" * len(header))

    for item in agg["folder_summaries"]:
        f_name = item["folder"]
        n_img = item["total_images"]
        n_succ = item["success_count"]
        s_pct = f"{item['success_rate_pct']:.1f}%"
        gt_acc = f"{item['gt_matches']}/{item['gt_total']} ({item['accuracy_pct']:.1f}%)"
        dur = f"{item['avg_duration_ms']:.0f} ms"

        print(f"{f_name:<12} | {n_img:<8} | {n_succ:<10} | {s_pct:<10} | {gt_acc:<12} | {dur:<10}")

    print("=" * len(header))
    tot_img = agg["total_images"]
    tot_succ = agg["total_successes"]
    tot_spct = f"{agg['overall_success_rate_pct']:.1f}%"
    tot_gt = f"{agg['total_gt_matches']}/{agg['total_gt_evaluated']} ({agg['overall_accuracy_pct']:.1f}%)"
    tot_dur = f"{agg['overall_avg_duration_ms']:.0f} ms"
    print(f"{'GRAND TOTAL':<12} | {tot_img:<8} | {tot_succ:<10} | {tot_spct:<10} | {tot_gt:<12} | {tot_dur:<10}")
    print("=" * len(header) + "\n")

    print("--- COMBINED OCR RECOGNITION RATE ---")
    for key, count in sorted(agg["combined_ocr_partial_breakdown"].items(), key=lambda x: x[0], reverse=True):
        pct = (count / tot_img * 100) if tot_img > 0 else 0
        print(f"  {key} recognized: {count:>3}x ({pct:.1f}%)")

    print("\n--- COMBINED METHOD DISTRIBUTION (SUCCESS) ---")
    for m_name, count in sorted(agg["combined_method_distribution"].items(), key=lambda x: -x[1]):
        pct = (count / tot_img * 100) if tot_img > 0 else 0
        print(f"  {m_name:<20}: {count:>3}x ({pct:.1f}%)")

    print("\n--- COMBINED FAILURE CATEGORIES (FAIL) ---")
    if agg["combined_failure_categories"]:
        for f_reason, count in sorted(agg["combined_failure_categories"].items(), key=lambda x: -x[1]):
            pct = (count / tot_img * 100) if tot_img > 0 else 0
            print(f"  {f_reason:<25}: {count:>3}x ({pct:.1f}%)")
    else:
        print("  None! (100% Recognition)")
    print("=" * 100 + "\n")


def main():
    print("Starting Batch Benchmark Process...")
    folder_paths = prepare_split_folders(batch_size=BATCH_SIZE)
    report_paths = run_batch_benchmarks(folder_paths)
    agg_results = aggregate_results(report_paths)
    print_aggregated_report(agg_results)

    # Save combined report
    combined_summary_path = os.path.join(OUTPUT_DIR, "combined_batch_benchmark_summary.json")
    with open(combined_summary_path, "w", encoding="utf-8") as f:
        json.dump(agg_results, f, indent=2, ensure_ascii=False)

    logger.info(f"Combined benchmark summary saved to: {combined_summary_path}")


if __name__ == "__main__":
    main()
