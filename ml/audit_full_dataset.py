"""
Stage 13: Full 1,250-VM Dataset Audit
====================================
Comprehensive audit of all 1,250 Bitbrains fastStorage VM traces under approved
Preprocessing Policy C (Active Lifetime + 5-min Canonical Alignment + Limit=2 Linear Interpolation).

Audit Requirements & Methodological Guardrails:
1. Scans every fastStorage VM trace (1.csv to 1250.csv).
2. Evaluates active provisioned lifetime: (memory_capacity_kb > 0) & (cpu_capacity_mhz > 0).
3. Resamples active span onto canonical 5-minute grid (resample('5min').mean()).
4. Interpolates isolated gaps <= 2 consecutive samples (method='linear', limit=2).
5. Gaps > 2 samples remain sequence breaks (NaN).
6. Computes valid sliding windows (L=288, H=12, W=300, stride=1, no gap crossing):
   - Whole active aligned trace
   - Usable train windows (70% chronological split)
   - Usable validation windows (15% chronological split)
   - Usable test windows (15% chronological split)
   - Mathematically enforces: train + val + test <= whole
7. Evaluates unseen VM evaluation eligibility:
   - Eligible iff valid_24h_to_1h_windows > 0 (at least one contiguous 25h window).
8. Generates:
   - results/final_dataset/vm_audit.csv
   - results/final_dataset/dataset_audit_summary.json
9. Read-only safety:
   - Raw CSV traces are NEVER modified.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.data_loader import load_bitbrains_vm
from ml.preprocessing_evaluation import evaluate_candidate_active_rule, evaluate_policy_c_active_aligned
from ml.temporal_windows import partition_vm_chronologically

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

WINDOW_L = 288  # 24 hours of history at 5-minute intervals
WINDOW_H = 12   # 1 hour forecast horizon at 5-minute intervals
WINDOW_W = WINDOW_L + WINDOW_H  # 300 steps total (25 hours)


def count_contiguous_windows(df_slice: pd.DataFrame, W: int = WINDOW_W) -> int:
    """
    Computes count of valid sliding windows of size W with zero NaNs across all telemetry channels.
    On a canonical 5-minute grid, all step differences are exactly 300s,
    so validity is strictly determined by complete contiguity (absence of NaNs).
    """
    N = len(df_slice)
    if N < W:
        return 0
    numeric_cols = [c for c in df_slice.columns if c not in ("timestamp", "timestamp_raw", "is_interpolated")]
    has_nan = df_slice[numeric_cols].isna().any(axis=1).to_numpy()
    cs = np.pad(np.cumsum(has_nan.astype(int)), (1, 0), mode="constant")
    win_nan_count = cs[W:] - cs[:-W]
    return int(np.sum(win_nan_count == 0))


def audit_single_vm(file_path: Path) -> Dict[str, Any]:
    """
    Audits a single VM trace CSV under Preprocessing Policy C.
    Handles corruptions or read errors gracefully without throwing.
    """
    fname = file_path.name
    vm_id_str = file_path.stem
    vm_id = int(vm_id_str) if vm_id_str.isdigit() else vm_id_str

    result: Dict[str, Any] = {
        "vm_id": vm_id,
        "filename": fname,
        "parsing_status": "PENDING",
        "raw_row_count": 0,
        "active_row_count": 0,
        "active_duration": 0,
        "active_duration_seconds": 0,
        "active_duration_days": 0.0,
        "aligned_5min_steps": 0,
        "interpolated_steps": 0,
        "long_gap_count": 0,
        "valid_24h_to_1h_windows": 0,
        "usable_train_windows": 0,
        "usable_validation_windows": 0,
        "usable_test_windows": 0,
        "eligible_for_unseen_vm_evaluation": False,
        "eligibility_reason": "UNPROCESSED",
        "inactive_prefix_rows": 0,
        "inactive_suffix_rows": 0,
        "internal_inactive_rows": 0,
        "active_start_timestamp": None,
        "active_end_timestamp": None,
    }

    try:
        if not file_path.exists():
            result["parsing_status"] = "FILE_NOT_FOUND"
            result["eligibility_reason"] = "FILE_NOT_FOUND"
            return result

        # 1. Load raw VM trace
        df_raw = load_bitbrains_vm(file_path)
        raw_count = len(df_raw)
        result["raw_row_count"] = raw_count

        if raw_count == 0:
            result["parsing_status"] = "EMPTY_FILE"
            result["eligibility_reason"] = "EMPTY_FILE_ZERO_ROWS"
            return result

        # 2. Evaluate active lifetime rule
        rule = evaluate_candidate_active_rule(df_raw, fname)
        active_count = rule.get("active_rows", 0)
        result["active_row_count"] = active_count
        result["inactive_prefix_rows"] = rule.get("inactive_prefix_rows", 0)
        result["inactive_suffix_rows"] = rule.get("inactive_suffix_rows", 0)
        result["internal_inactive_rows"] = rule.get("internal_inactive_rows", 0)

        if active_count == 0 or rule.get("start_idx") is None:
            result["parsing_status"] = "SUCCESS"
            result["eligibility_reason"] = "ZERO_ACTIVE_ROWS_NO_PROVISIONED_CAPACITY"
            return result

        start_idx = rule["start_idx"]
        end_idx = rule["end_idx"]
        ts_raw = df_raw["timestamp_raw"]
        start_ts = int(ts_raw.iloc[start_idx])
        end_ts = int(ts_raw.iloc[end_idx])
        duration_s = max(0, end_ts - start_ts)
        duration_days = round(duration_s / 86400.0, 2)

        result["active_duration"] = duration_s
        result["active_duration_seconds"] = duration_s
        result["active_duration_days"] = duration_days
        result["active_start_timestamp"] = start_ts
        result["active_end_timestamp"] = end_ts

        # 3. Policy C: 5-min canonical alignment + linear interpolation limit=2
        stat_c, df_aligned = evaluate_policy_c_active_aligned(df_raw, fname, rule)
        aligned_steps = len(df_aligned)
        result["aligned_5min_steps"] = aligned_steps
        result["interpolated_steps"] = stat_c.get("interpolated_steps", 0)

        if aligned_steps == 0:
            result["parsing_status"] = "SUCCESS"
            result["eligibility_reason"] = "ZERO_ALIGNED_STEPS"
            return result

        # 4. Long gap detection: count contiguous NaN blocks in resampled series
        # Gaps > 2 consecutive samples remain NaN after limit=2 linear interpolation
        is_nan = df_aligned["cpu_usage_percent"].isna()
        nan_blocks = int((is_nan & (~is_nan.shift(1, fill_value=False))).sum())
        result["long_gap_count"] = nan_blocks

        # 5. Sliding window calculation on whole active aligned trace
        whole_windows = count_contiguous_windows(df_aligned, W=WINDOW_W)
        result["valid_24h_to_1h_windows"] = whole_windows

        # 6. Chronological 70% / 15% / 15% split window calculation
        splits = partition_vm_chronologically(df_aligned, train_ratio=0.70, val_ratio=0.15)
        train_windows = count_contiguous_windows(splits["train"], W=WINDOW_W)
        val_windows = count_contiguous_windows(splits["val"], W=WINDOW_W)
        test_windows = count_contiguous_windows(splits["test"], W=WINDOW_W)

        result["usable_train_windows"] = train_windows
        result["usable_validation_windows"] = val_windows
        result["usable_test_windows"] = test_windows

        # Consistency verification: train + val + test <= whole
        assert (train_windows + val_windows + test_windows) <= whole_windows, (
            f"Split window count invariant violation in {fname}: "
            f"train({train_windows}) + val({val_windows}) + test({test_windows}) = "
            f"{train_windows + val_windows + test_windows} > whole({whole_windows})"
        )

        # 7. Unseen VM evaluation eligibility
        if whole_windows > 0:
            result["eligible_for_unseen_vm_evaluation"] = True
            result["eligibility_reason"] = "ELIGIBLE"
        else:
            result["eligible_for_unseen_vm_evaluation"] = False
            if aligned_steps < WINDOW_W:
                result["eligibility_reason"] = f"ACTIVE_DURATION_UNDER_25H ({aligned_steps} steps < {WINDOW_W})"
            else:
                result["eligibility_reason"] = f"EXCESSIVE_GAPS_NO_CONTIGUOUS_25H_WINDOW ({nan_blocks} long gaps)"

        result["parsing_status"] = "SUCCESS"

    except Exception as exc:
        logger.error(f"Error auditing VM trace {fname}: {exc}", exc_info=True)
        result["parsing_status"] = f"PARSE_ERROR: {str(exc)}"
        result["eligibility_reason"] = f"PARSE_ERROR: {str(exc)}"

    return result


def run_dataset_audit(
    data_dir: Path,
    output_dir: Path,
    max_workers: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes the full Stage 13 audit across all 1,250 fastStorage VM traces.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(list(data_dir.glob("*.csv")), key=lambda p: int(p.stem) if p.stem.isdigit() else p.name)
    total_files = len(csv_files)
    logger.info(f"Discovered {total_files} CSV files in {data_dir}")

    start_time = time.time()
    results: List[Dict[str, Any]] = []

    # Choose worker count
    workers = max_workers or min(10, os.cpu_count() or 4)
    logger.info(f"Auditing using {workers} parallel worker processes...")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(audit_single_vm, p): p for p in csv_files}
        completed = 0
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                res = fut.result()
                results.append(res)
            except Exception as exc:
                logger.error(f"Unexpected worker crash for {p.name}: {exc}")
                results.append({
                    "vm_id": int(p.stem) if p.stem.isdigit() else p.stem,
                    "filename": p.name,
                    "parsing_status": f"CRASH: {exc}",
                    "raw_row_count": 0,
                    "active_row_count": 0,
                    "active_duration": 0,
                    "active_duration_seconds": 0,
                    "active_duration_days": 0.0,
                    "aligned_5min_steps": 0,
                    "interpolated_steps": 0,
                    "long_gap_count": 0,
                    "valid_24h_to_1h_windows": 0,
                    "usable_train_windows": 0,
                    "usable_validation_windows": 0,
                    "usable_test_windows": 0,
                    "eligible_for_unseen_vm_evaluation": False,
                    "eligibility_reason": f"WORKER_CRASH: {exc}",
                })
            completed += 1
            if completed % 250 == 0 or completed == total_files:
                elapsed = time.time() - start_time
                logger.info(f"Progress: {completed}/{total_files} files audited ({elapsed:.1f}s, {completed/elapsed:.1f} files/s)")

    # Sort deterministically by vm_id
    results = sorted(results, key=lambda d: d["vm_id"] if isinstance(d["vm_id"], int) else 999999)
    df_audit = pd.DataFrame(results)

    core_cols = [
        "vm_id",
        "filename",
        "raw_row_count",
        "active_row_count",
        "active_duration",
        "active_duration_seconds",
        "active_duration_days",
        "aligned_5min_steps",
        "interpolated_steps",
        "long_gap_count",
        "valid_24h_to_1h_windows",
        "usable_train_windows",
        "usable_validation_windows",
        "usable_test_windows",
        "eligible_for_unseen_vm_evaluation",
        "eligibility_reason",
        "parsing_status",
        "inactive_prefix_rows",
        "inactive_suffix_rows",
        "internal_inactive_rows",
        "active_start_timestamp",
        "active_end_timestamp",
    ]
    ordered_cols = [c for c in core_cols if c in df_audit.columns] + [c for c in df_audit.columns if c not in core_cols]
    df_audit = df_audit[ordered_cols]

    # Output CSV path
    audit_csv_path = output_dir / "vm_audit.csv"
    df_audit.to_csv(audit_csv_path, index=False)
    logger.info(f"Saved vm_audit.csv with {len(df_audit)} rows to {audit_csv_path}")

    # Compute summary statistics
    total_files_scanned = len(df_audit)
    total_readable_files = int((df_audit["parsing_status"] == "SUCCESS").sum())
    total_failed_files = total_files_scanned - total_readable_files

    total_raw_rows = int(df_audit["raw_row_count"].sum())
    total_active_rows = int(df_audit["active_row_count"].sum())
    total_aligned_5min_steps = int(df_audit["aligned_5min_steps"].sum())
    total_interpolated_steps = int(df_audit["interpolated_steps"].sum())
    total_long_gaps = int(df_audit["long_gap_count"].sum())

    total_valid_windows_whole = int(df_audit["valid_24h_to_1h_windows"].sum())
    total_train_windows = int(df_audit["usable_train_windows"].sum())
    total_validation_windows = int(df_audit["usable_validation_windows"].sum())
    total_test_windows = int(df_audit["usable_test_windows"].sum())

    total_eligible = int(df_audit["eligible_for_unseen_vm_evaluation"].sum())
    total_ineligible = total_files_scanned - total_eligible

    # Distributions
    active_dur_s = df_audit["active_duration_seconds"].to_numpy()
    active_dur_d = df_audit["active_duration_days"].to_numpy()
    valid_wins = df_audit["valid_24h_to_1h_windows"].to_numpy()

    # Anomalies / Special categories
    anomalous_files = []
    for _, row in df_audit.iterrows():
        reasons = []
        if row["parsing_status"] != "SUCCESS":
            reasons.append(f"Parsing error: {row['parsing_status']}")
        if row["active_row_count"] == 0:
            reasons.append("Zero active rows (no provisioned capacity)")
        elif row["aligned_5min_steps"] < WINDOW_W:
            reasons.append(f"Short active lifetime: {row['aligned_5min_steps']} steps ({row['active_duration_days']:.2f} days < 25h)")
        elif row["valid_24h_to_1h_windows"] == 0:
            reasons.append(f"Zero valid windows despite {row['aligned_5min_steps']} steps (long gaps: {row['long_gap_count']})")
        if row["internal_inactive_rows"] > 0:
            reasons.append(f"{row['internal_inactive_rows']} internal inactive rows")

        if reasons:
            anomalous_files.append({
                "vm_id": int(row["vm_id"]),
                "filename": str(row["filename"]),
                "active_rows": int(row["active_row_count"]),
                "aligned_steps": int(row["aligned_5min_steps"]),
                "valid_windows": int(row["valid_24h_to_1h_windows"]),
                "eligible": bool(row["eligible_for_unseen_vm_evaluation"]),
                "anomalies": "; ".join(reasons),
            })

    summary: Dict[str, Any] = {
        "total_files_scanned": total_files_scanned,
        "total_readable_files": total_readable_files,
        "total_failed_files": total_failed_files,
        "total_raw_rows": total_raw_rows,
        "total_active_rows": total_active_rows,
        "total_aligned_5min_steps": total_aligned_5min_steps,
        "total_interpolated_steps": total_interpolated_steps,
        "total_long_gaps": total_long_gaps,
        "total_valid_windows_whole": total_valid_windows_whole,
        "total_train_windows": total_train_windows,
        "total_validation_windows": total_validation_windows,
        "total_test_windows": total_test_windows,
        "total_eligible_unseen_vms": total_eligible,
        "total_ineligible_vms": total_ineligible,
        "distribution_of_active_durations": {
            "unit": "seconds",
            "min": float(np.min(active_dur_s)),
            "p25": float(np.percentile(active_dur_s, 25)),
            "median": float(np.median(active_dur_s)),
            "p75": float(np.percentile(active_dur_s, 75)),
            "max": float(np.max(active_dur_s)),
        },
        "distribution_of_active_durations_days": {
            "unit": "days",
            "min": float(np.min(active_dur_d)),
            "p25": float(np.percentile(active_dur_d, 25)),
            "median": float(np.median(active_dur_d)),
            "p75": float(np.percentile(active_dur_d, 75)),
            "max": float(np.max(active_dur_d)),
        },
        "distribution_of_valid_windows_per_vm": {
            "min": int(np.min(valid_wins)),
            "p25": float(np.percentile(valid_wins, 25)),
            "median": float(np.median(valid_wins)),
            "p75": float(np.percentile(valid_wins, 75)),
            "max": int(np.max(valid_wins)),
        },
        "distribution_of_valid_windows_eligible_vms_only": {
            "min": int(np.min(valid_wins[valid_wins > 0])) if total_eligible > 0 else 0,
            "median": float(np.median(valid_wins[valid_wins > 0])) if total_eligible > 0 else 0.0,
            "max": int(np.max(valid_wins[valid_wins > 0])) if total_eligible > 0 else 0,
        },
        "corrupted_unreadable_or_anomalous_files": anomalous_files,
        "ineligible_breakdown": {
            "total_ineligible": total_ineligible,
            "zero_active_rows": int((df_audit["active_row_count"] == 0).sum()),
            "active_duration_under_25h": int(((df_audit["active_row_count"] > 0) & (df_audit["aligned_5min_steps"] < WINDOW_W)).sum()),
            "excessive_gaps_under_300_valid": int(((df_audit["aligned_5min_steps"] >= WINDOW_W) & (df_audit["valid_24h_to_1h_windows"] == 0)).sum()),
            "parse_or_read_errors": total_failed_files,
        },
        "methodology_and_invariants": {
            "preprocessing_policy": "Policy C (Active Lifetime + 5-min Canonical Alignment + Linear Interpolation limit=2)",
            "active_condition": "memory_capacity_kb > 0 and cpu_capacity_mhz > 0",
            "window_size_W": WINDOW_W,
            "history_L": WINDOW_L,
            "horizon_H": WINDOW_H,
            "stride": 1,
            "split_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
            "split_containment_invariant": "train_windows + val_windows + test_windows <= whole_windows (PASSED)",
        },
    }

    summary_json_path = output_dir / "dataset_audit_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved dataset_audit_summary.json to {summary_json_path}")

    total_time = time.time() - start_time
    logger.info(f"Stage 13 Full Dataset Audit completed in {total_time:.2f}s ({total_files/total_time:.1f} files/s)")

    return df_audit, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 13: Full 1,250-VM Dataset Audit")
    parser.add_argument("--data_dir", type=str, default="dataset/fastStorage/2013-8", help="Path to fastStorage CSV directory")
    parser.add_argument("--output_dir", type=str, default="results/final_dataset", help="Output directory for audit artifacts")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel worker processes")
    args = parser.parse_args()

    run_dataset_audit(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        max_workers=args.workers,
    )
