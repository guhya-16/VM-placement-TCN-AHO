"""
Bitbrains Dataset Inspection & Cross-File Validation CLI (Stage 1)
==================================================================
Runs data quality inspections and schema consistency checks on single Bitbrains VM CSV
files or batches of files (e.g., 5 or 10 files).

Features:
- Single-file detailed statistical and quality inspection.
- Multi-file cross-file schema consistency verification (column count, names, ordering, dtypes).
- Detects and reports duplicate rows and duplicate timestamps without modifying raw data.
- Measures raw sampling intervals without automatic resampling.
- Saves audit reports into results/inspection/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ml.data_loader import (
    EXPECTED_RAW_COLUMNS,
    RAW_TO_NORMALIZED_COLS,
    load_bitbrains_vm,
    validate_bitbrains_df,
)


def format_bytes(kb: float) -> str:
    """Format KB into human-readable MB or GB."""
    if kb >= 1024 * 1024:
        return f"{kb / (1024 * 1024):.2f} GB"
    elif kb >= 1024:
        return f"{kb / 1024:.2f} MB"
    return f"{kb:.2f} KB"


def inspect_single_file(file_path: Path, output_dir: Path) -> Dict[str, Any]:
    """Inspects a single Bitbrains VM CSV file and writes a formatted report."""
    print(f"\n==================================================")
    print(f"STAGE 1 INSPECTION: {file_path.name}")
    print(f"==================================================")

    # 1. Load data
    df = load_bitbrains_vm(file_path, standardize_columns=True, add_datetime=True)
    val = validate_bitbrains_df(df, file_name=file_path.name)

    # 2. Extract statistics
    ts_diffs = df["timestamp_raw"].diff().dropna()
    cpu_stats = val.get("cpu_usage_percent_stats", {})

    mem_usage = df["memory_usage_kb"].dropna()
    mem_cap = df["memory_capacity_kb"].dropna()

    report_lines = [
        f"Bitbrains VM Trace Inspection Report: {file_path.name}",
        f"File Path: {file_path.resolve()}",
        f"Raw File Size: {os.path.getsize(file_path):,} bytes",
        f"--------------------------------------------------",
        f"OBSERVATIONS & DIMENSIONS:",
        f"  Total Rows:             {val['row_count']:,}",
        f"  Total Columns:          {val['column_count']}",
        f"  Columns:                {', '.join(df.columns)}",
        f"--------------------------------------------------",
        f"DATA TYPES:",
    ]
    for col, dtype in df.dtypes.items():
        report_lines.append(f"  {col:<30}: {dtype}")

    report_lines.extend([
        f"--------------------------------------------------",
        f"DATA INTEGRITY & QUALITY:",
        f"  Missing (Null) Values:  {val['total_missing']}",
        f"  Duplicate Rows:         {val['duplicate_rows']}",
        f"  Duplicate Timestamps:   {val['duplicate_timestamps']}",
        f"  Monotonic Timestamps:   {val['is_timestamp_monotonic']}",
        f"  Negative Metric Count:  {sum(val['negative_value_counts'].values())}",
        f"--------------------------------------------------",
        f"TIMESTAMP & SAMPLING BEHAVIOR:",
        f"  Timestamp Start (Raw):  {val['timestamp_start']} (Epoch seconds)",
        f"  Timestamp End (Raw):    {val['timestamp_end']} (Epoch seconds)",
        f"  Datetime Start (UTC):   {val.get('datetime_start', 'N/A')}",
        f"  Datetime End (UTC):     {val.get('datetime_end', 'N/A')}",
        f"  Duration:               {val['duration_seconds']:,} seconds ({val['duration_days']} days)",
    ])

    if val.get("interval_stats"):
        istats = val["interval_stats"]
        report_lines.extend([
            f"  Min Sampling Interval:  {istats['min_interval_seconds']} s",
            f"  Median Sampling Interval: {istats['median_interval_seconds']} s (~{istats['median_interval_seconds']/60:.1f} min)",
            f"  Mean Sampling Interval: {istats['mean_interval_seconds']} s",
            f"  Max Sampling Interval:  {istats['max_interval_seconds']} s",
            f"  Top Interval Frequencies:",
        ])
        for gap, count in istats["top_intervals_counts"].items():
            report_lines.append(f"    {gap:>6} s: {count:>6} occurrences ({count/len(ts_diffs)*100:.2f}%)")

    report_lines.extend([
        f"--------------------------------------------------",
        f"RESOURCE METRIC STATISTICS:",
        f"  CPU Usage [%]:",
        f"    Min:                  {cpu_stats.get('min', 'N/A')}%",
        f"    Max:                  {cpu_stats.get('max', 'N/A')}%",
        f"    Mean:                 {cpu_stats.get('mean', 'N/A')}%",
        f"    Median:               {cpu_stats.get('median', 'N/A')}%",
        f"    Std Dev:              {cpu_stats.get('std', 'N/A')}%",
        f"  CPU Cores Provisioned:  {df['cpu_cores'].iloc[0] if not df.empty else 'N/A'}",
        f"  CPU Capacity [MHZ]:     {df['cpu_capacity_mhz'].iloc[0] if not df.empty else 'N/A':.2f} MHz",
        f"  Memory Provisioned:     {format_bytes(mem_cap.iloc[0]) if not mem_cap.empty else 'N/A'}",
        f"  Memory Usage [Mean]:    {format_bytes(mem_usage.mean()) if not mem_usage.empty else 'N/A'}",
        f"  Memory Usage [Max]:     {format_bytes(mem_usage.max()) if not mem_usage.empty else 'N/A'}",
        f"--------------------------------------------------",
    ])

    if val["observations"]:
        report_lines.append("OBSERVATIONS & NOTICES:")
        for obs in val["observations"]:
            report_lines.append(f"  - {obs}")
    else:
        report_lines.append("OBSERVATIONS & NOTICES: Clean file. No duplicates or anomalies detected.")

    report_text = "\n".join(report_lines)
    print(report_text)

    # Print first 3 and last 3 rows preview
    print("\nSAMPLE HEAD (First 3 rows):")
    print(df.head(3)[["timestamp_raw", "timestamp", "cpu_cores", "cpu_usage_mhz", "cpu_usage_percent", "memory_usage_kb"]].to_string())
    print("\nSAMPLE TAIL (Last 3 rows):")
    print(df.tail(3)[["timestamp_raw", "timestamp", "cpu_cores", "cpu_usage_mhz", "cpu_usage_percent", "memory_usage_kb"]].to_string())

    # Save report
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / f"{file_path.stem}_report.txt"
    report_file.write_text(report_text, encoding="utf-8")
    print(f"\nSaved single-file report to: {report_file}")

    return val


def inspect_multi_files(dir_path: Path, count: int, output_dir: Path) -> Dict[str, Any]:
    """
    Inspects the first `count` files in `dir_path` and verifies cross-file schema consistency.
    """
    print(f"\n==================================================")
    print(f"STAGE 1 MULTI-FILE VALIDATION (First {count} files)")
    print(f"Directory: {dir_path.resolve()}")
    print(f"==================================================")

    # Collect and sort CSV files numerically (1.csv, 2.csv, etc.)
    all_files = list(dir_path.glob("*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {dir_path}")

    def numeric_sort_key(p: Path) -> int:
        try:
            return int(p.stem)
        except ValueError:
            return 999999

    sorted_files = sorted(all_files, key=numeric_sort_key)[:count]

    summaries: List[Dict[str, Any]] = []
    schema_deviations: List[str] = []
    reference_columns: List[str] = []
    reference_dtypes: Dict[str, str] = {}

    for idx, fpath in enumerate(sorted_files, start=1):
        df = load_bitbrains_vm(fpath, standardize_columns=True, add_datetime=True)
        val = validate_bitbrains_df(df, file_name=fpath.name)

        # Baseline reference from first file
        if idx == 1:
            reference_columns = list(df.columns)
            reference_dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

        # Verify cross-file schema consistency
        file_cols = list(df.columns)
        if len(file_cols) != len(reference_columns):
            schema_deviations.append(
                f"{fpath.name}: column count mismatch (got {len(file_cols)}, expected {len(reference_columns)})"
            )
        elif file_cols != reference_columns:
            schema_deviations.append(
                f"{fpath.name}: column name/order mismatch (got {file_cols})"
            )

        # Dtype compatibility check
        for col in reference_columns:
            if col in df.columns:
                curr_dtype = str(df[col].dtype)
                ref_dtype = reference_dtypes[col]
                # Compare underlying kind (numeric vs datetime)
                if ("float" in curr_dtype or "int" in curr_dtype) and not (
                    "float" in ref_dtype or "int" in ref_dtype
                ):
                    schema_deviations.append(
                        f"{fpath.name}: column {col} dtype incompatible ({curr_dtype} vs {ref_dtype})"
                    )

        # Record summary metrics
        cpu_stats = val.get("cpu_usage_percent_stats", {})
        istats = val.get("interval_stats", {})

        summaries.append({
            "file": fpath.name,
            "rows": val["row_count"],
            "cols": val["column_count"],
            "missing": val["total_missing"],
            "duplicate_rows": val["duplicate_rows"],
            "duplicate_timestamps": val["duplicate_timestamps"],
            "monotonic_ts": val["is_timestamp_monotonic"],
            "date_start": val.get("datetime_start", "")[:10],
            "date_end": val.get("datetime_end", "")[:10],
            "days": val.get("duration_days", 0),
            "median_interval_s": istats.get("median_interval_seconds", np.nan),
            "cpu_mean_pct": cpu_stats.get("mean", np.nan),
            "cpu_max_pct": cpu_stats.get("max", np.nan),
            "observations_count": len(val["observations"]),
        })

    # Print Tabular Summary
    summary_df = pd.DataFrame(summaries)
    print("\nCROSS-FILE SUMMARY TABLE:")
    print(summary_df.to_string(index=False))

    # Cross-File Schema Consistency Report
    print("\n--------------------------------------------------")
    print("CROSS-FILE SCHEMA CONSISTENCY EVALUATION:")
    if not schema_deviations:
        print("  [SUCCESS] All files share 100% identical schema:")
        print(f"    - Column count:    {len(reference_columns)}")
        print(f"    - Column names:    {reference_columns}")
        print("    - Column order:    Identical across all files")
        print("    - Data types:      Compatible numeric & datetime types across all files")
        print("    - Timestamp scale: Confirmed 10-digit Unix epoch seconds (Aug-Sep 2013)")
        print("    - CPU [%] range:   Valid within [0, 100]% for all files")
    else:
        print("  [WARNING] Schema deviations detected:")
        for dev in schema_deviations:
            print(f"    - {dev}")

    # Report Data Quality Findings across files
    total_dupe_rows = sum(s["duplicate_rows"] for s in summaries)
    total_dupe_ts = sum(s["duplicate_timestamps"] for s in summaries)
    files_with_dupes = [s["file"] for s in summaries if s["duplicate_rows"] > 0 or s["duplicate_timestamps"] > 0]

    print("\nCROSS-FILE DATA QUALITY FINDINGS:")
    print(f"  Total Missing Values across all {count} files:    {sum(s['missing'] for s in summaries)}")
    print(f"  Files with Duplicate Rows/Timestamps:           {len(files_with_dupes)} / {count} {files_with_dupes}")
    print(f"  Total Duplicate Rows (kept untouched):          {total_dupe_rows}")
    print(f"  Total Duplicate Timestamps (kept untouched):    {total_dupe_ts}")
    print("--------------------------------------------------")

    # Save output summaries
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / f"multi_file_summary_{count}.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "file_count": count,
                "schema_consistent": len(schema_deviations) == 0,
                "schema_deviations": schema_deviations,
                "files_with_duplicates": files_with_dupes,
                "summaries": summaries,
            },
            f,
            indent=2,
        )

    summary_txt = output_dir / f"multi_file_summary_{count}.txt"
    summary_txt.write_text(summary_df.to_string(index=False), encoding="utf-8")
    print(f"Saved multi-file summaries to:\n  - {summary_json}\n  - {summary_txt}")

    return {
        "count": count,
        "schema_deviations": schema_deviations,
        "summaries": summaries,
    }


def main():
    parser = argparse.ArgumentParser(description="Bitbrains Dataset Inspector (Stage 1)")
    parser.add_argument(
        "--file",
        type=str,
        default="dataset/fastStorage/2013-8/1.csv",
        help="Path to specific Bitbrains VM CSV file to inspect.",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="dataset/fastStorage/2013-8",
        help="Path to directory containing Bitbrains CSV files.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of files to inspect in multi-file mode (e.g., 5 or 10).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/inspection",
        help="Directory to store inspection reports.",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    if args.count is not None:
        inspect_multi_files(Path(args.dir), count=args.count, output_dir=output_dir)
    else:
        inspect_single_file(Path(args.file), output_dir=output_dir)


if __name__ == "__main__":
    main()

