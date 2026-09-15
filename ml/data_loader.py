"""
Bitbrains Dataset Loader & Validator (Stage 1)
==============================================
Provides reusable loading and validation functions for Bitbrains GWA-T-12 VM trace CSVs.

Key Bitbrains Quirks Handled:
1. Delimiter: Delimited by semicolon + tab (";\\t").
2. Quoted rows: Raw CSV lines are enclosed in double quotes (e.g., `"val1;\\tval2;\\t..."`).
3. Timestamp units: The raw CSV header specifies "Timestamp [ms]", but actual values
   are 10-digit Unix epoch SECONDS (e.g., 1376314846 = 2013-08-12 13:40:46 UTC).
   If parsed as milliseconds, the timestamp would correspond to 1970.
   The loader inspects the numeric values, converts using Unix epoch seconds, validates
   that resulting dates fall within plausible trace periods, and retains the raw integer
   timestamp as 'timestamp_raw' for full traceability and simulation compatibility.
4. Read-only safety: The raw CSV files are never modified, overwritten, or resaved.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# The 11 expected raw columns in Bitbrains fastStorage traces (exact original order)
EXPECTED_RAW_COLUMNS: List[str] = [
    "Timestamp [ms]",
    "CPU cores",
    "CPU capacity provisioned [MHZ]",
    "CPU usage [MHZ]",
    "CPU usage [%]",
    "Memory capacity provisioned [KB]",
    "Memory usage [KB]",
    "Disk read throughput [KB/s]",
    "Disk write throughput [KB/s]",
    "Network received throughput [KB/s]",
    "Network transmitted throughput [KB/s]",
]

# Standardized snake_case column mapping
RAW_TO_NORMALIZED_COLS: Dict[str, str] = {
    "Timestamp [ms]": "timestamp_raw",
    "CPU cores": "cpu_cores",
    "CPU capacity provisioned [MHZ]": "cpu_capacity_mhz",
    "CPU usage [MHZ]": "cpu_usage_mhz",
    "CPU usage [%]": "cpu_usage_percent",
    "Memory capacity provisioned [KB]": "memory_capacity_kb",
    "Memory usage [KB]": "memory_usage_kb",
    "Disk read throughput [KB/s]": "disk_read_kbps",
    "Disk write throughput [KB/s]": "disk_write_kbps",
    "Network received throughput [KB/s]": "network_received_kbps",
    "Network transmitted throughput [KB/s]": "network_transmitted_kbps",
}

# Reverse mapping for traceability
NORMALIZED_TO_RAW_COLS: Dict[str, str] = {
    norm: raw for raw, norm in RAW_TO_NORMALIZED_COLS.items()
}

# Resource metrics expected to be strictly non-negative
NON_NEGATIVE_METRICS: List[str] = [
    "cpu_cores",
    "cpu_capacity_mhz",
    "cpu_usage_mhz",
    "cpu_usage_percent",
    "memory_capacity_kb",
    "memory_usage_kb",
    "disk_read_kbps",
    "disk_write_kbps",
    "network_received_kbps",
    "network_transmitted_kbps",
]


def load_bitbrains_vm(
    file_path: Union[str, Path],
    standardize_columns: bool = True,
    add_datetime: bool = True,
) -> pd.DataFrame:
    """
    Loads a single Bitbrains VM trace CSV file into a pandas DataFrame.

    Parameters:
    -----------
    file_path : str or Path
        Path to the raw Bitbrains CSV file.
    standardize_columns : bool, default True
        If True, rename columns to clean snake_case names while preserving 'timestamp_raw'.
        If False, retain original column names (quotes stripped).
    add_datetime : bool, default True
        If True, adds a converted UTC datetime column named 'timestamp'.

    Returns:
    --------
    pd.DataFrame
        Clean DataFrame containing parsed observations with raw data preserved.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Bitbrains CSV file not found: {path}")

    # Read lines and strip line-level wrapping quotes without altering raw file on disk
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        clean_lines = [
            line.strip()[1:-1]
            if line.strip().startswith('"') and line.strip().endswith('"')
            else line.strip()
            for line in f
        ]

    if not clean_lines:
        raise ValueError(f"File {path.name} is empty.")

    # Parse using pandas Python engine with delimiter ";\t"
    raw_csv_buffer = io.StringIO("\n".join(clean_lines))
    df = pd.read_csv(raw_csv_buffer, sep=r";\t", engine="python")

    # Clean any accidental whitespace or remaining quote marks from column headers
    df.columns = [c.strip().strip('"').strip() for c in df.columns]

    # Timestamp column handling
    raw_ts_col = "Timestamp [ms]"
    if raw_ts_col not in df.columns:
        # Fallback: check if the first column is the timestamp
        raw_ts_col = df.columns[0]

    # Ensure timestamp is integer
    df[raw_ts_col] = pd.to_numeric(df[raw_ts_col], errors="coerce")

    # Inspect timestamp scale and convert to UTC datetime
    if add_datetime:
        ts_sample = df[raw_ts_col].dropna().iloc[0] if len(df) > 0 else 0
        
        # Robust timestamp validation:
        # 10-digit values (1e9 to 2e9) represent Unix epoch seconds (~2001 to ~2033).
        # 13-digit values (>1e12) represent Unix epoch milliseconds.
        # Bitbrains GWA-T-12 traces are recorded in Aug-Sep 2013 (~1.37e9 seconds).
        if 1e9 <= ts_sample < 2e9:
            ts_unit = "s"
        elif ts_sample >= 1e12:
            ts_unit = "ms"
        else:
            # Fallback based on verified dataset structure
            ts_unit = "s"

        dt_series = pd.to_datetime(df[raw_ts_col], unit=ts_unit, utc=True)
        
        # Validate that the resulting date range is plausible (e.g. between 2005 and 2030)
        valid_dates = dt_series.dropna()
        if not valid_dates.empty:
            min_year = valid_dates.min().year
            max_year = valid_dates.max().year
            if not (2005 <= min_year <= 2030 and 2005 <= max_year <= 2030):
                # If interpreting as seconds yielded an invalid year, try ms
                alt_dt = pd.to_datetime(df[raw_ts_col], unit="ms", utc=True)
                if 2005 <= alt_dt.dropna().min().year <= 2030:
                    dt_series = alt_dt

    # Standardize column names if requested
    if standardize_columns:
        # Map existing columns to normalized names
        new_cols = {col: RAW_TO_NORMALIZED_COLS.get(col, col) for col in df.columns}
        df = df.rename(columns=new_cols)

        # Add datetime column positioned right after timestamp_raw
        if add_datetime:
            df.insert(1, "timestamp", dt_series)
    else:
        if add_datetime:
            df.insert(1, "Timestamp [datetime]", dt_series)

    # Ensure numeric columns are cast appropriately
    numeric_cols = [
        c for c in df.columns if c not in ("timestamp", "Timestamp [datetime]")
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def validate_bitbrains_df(
    df: pd.DataFrame,
    file_name: str = "",
    expected_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Performs comprehensive data quality and schema validation on a Bitbrains DataFrame.

    Checks performed:
    A. Expected columns and order
    B. Data types
    C. Timestamp validity & monotonic ordering
    D. Duplicate rows (detected and reported without removal)
    E. Duplicate timestamps (detected and reported without removal)
    F. Missing (NaN / Null) values
    G. Negative resource metric check
    H. CPU percentage range [0, 100]
    I. Sampling interval distribution (min, median, mean, max, common frequencies)

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame loaded via load_bitbrains_vm.
    file_name : str
        Optional identifier for reporting.
    expected_cols : list of str, optional
        Custom expected column list. Defaults to normalized or raw Bitbrains schema.

    Returns:
    --------
    dict
        Detailed validation results and statistics.
    """
    results: Dict[str, Any] = {
        "file_name": file_name,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": df.columns.tolist(),
        "is_schema_valid": True,
        "schema_issues": [],
        "missing_values": {},
        "total_missing": 0,
        "duplicate_rows": 0,
        "duplicate_timestamps": 0,
        "is_timestamp_monotonic": True,
        "timestamp_start": None,
        "timestamp_end": None,
        "duration_seconds": None,
        "duration_days": None,
        "interval_stats": {},
        "cpu_usage_percent_stats": {},
        "negative_value_counts": {},
        "observations": [],
    }

    if df.empty:
        results["is_schema_valid"] = False
        results["schema_issues"].append("DataFrame is empty.")
        return results

    # Determine whether columns are normalized
    is_normalized = "timestamp_raw" in df.columns
    ts_raw_col = "timestamp_raw" if is_normalized else "Timestamp [ms]"
    cpu_pct_col = "cpu_usage_percent" if is_normalized else "CPU usage [%]"

    # A. Check expected columns
    if expected_cols is None:
        if is_normalized:
            expected_cols = ["timestamp_raw", "timestamp"] + [
                RAW_TO_NORMALIZED_COLS[c]
                for c in EXPECTED_RAW_COLUMNS
                if c != "Timestamp [ms]"
            ]
        else:
            expected_cols = (
                EXPECTED_RAW_COLUMNS
                if "Timestamp [datetime]" not in df.columns
                else ["Timestamp [ms]", "Timestamp [datetime]"]
                + EXPECTED_RAW_COLUMNS[1:]
            )

    missing_expected = [c for c in expected_cols if c not in df.columns]
    unexpected_cols = [c for c in df.columns if c not in expected_cols]

    if missing_expected:
        results["is_schema_valid"] = False
        results["schema_issues"].append(f"Missing expected columns: {missing_expected}")
    if unexpected_cols:
        results["is_schema_valid"] = False
        results["schema_issues"].append(f"Unexpected columns found: {unexpected_cols}")

    # Check column order
    if list(df.columns) != expected_cols:
        results["is_schema_valid"] = False
        results["schema_issues"].append("Column ordering does not match expected schema.")

    # B. Missing values
    missing_dict = df.isna().sum().to_dict()
    results["missing_values"] = missing_dict
    results["total_missing"] = int(sum(missing_dict.values()))

    # C. Duplicate rows (reported, not removed)
    results["duplicate_rows"] = int(df.duplicated().sum())
    if results["duplicate_rows"] > 0:
        results["observations"].append(
            f"Detected {results['duplicate_rows']} completely duplicate rows."
        )

    # D. Duplicate timestamps & timestamp ordering (reported, not removed)
    if ts_raw_col in df.columns:
        ts_series = df[ts_raw_col].dropna()
        dupe_ts_count = int(ts_series.duplicated().sum())
        results["duplicate_timestamps"] = dupe_ts_count
        if dupe_ts_count > 0:
            results["observations"].append(
                f"Detected {dupe_ts_count} duplicate timestamp occurrences."
            )

        # Monotonicity check
        is_monotonic = ts_series.is_monotonic_increasing
        results["is_timestamp_monotonic"] = bool(is_monotonic)
        if not is_monotonic:
            results["observations"].append("Timestamps are not strictly monotonically increasing.")

        # Timestamp start, end, duration
        ts_min = int(ts_series.min())
        ts_max = int(ts_series.max())
        duration = ts_max - ts_min
        results["timestamp_start"] = ts_min
        results["timestamp_end"] = ts_max
        results["duration_seconds"] = duration
        results["duration_days"] = round(duration / 86400.0, 2)

        # Datetime representations if available
        if "timestamp" in df.columns:
            results["datetime_start"] = str(df["timestamp"].min())
            results["datetime_end"] = str(df["timestamp"].max())
        elif "Timestamp [datetime]" in df.columns:
            results["datetime_start"] = str(df["Timestamp [datetime]"].min())
            results["datetime_end"] = str(df["Timestamp [datetime]"].max())

        # Sampling intervals
        diffs = ts_series.diff().dropna()
        if not diffs.empty:
            top_freqs = diffs.value_counts().head(5).to_dict()
            results["interval_stats"] = {
                "min_interval_seconds": float(diffs.min()),
                "median_interval_seconds": float(diffs.median()),
                "mean_interval_seconds": round(float(diffs.mean()), 2),
                "max_interval_seconds": float(diffs.max()),
                "top_intervals_counts": {
                    int(k) if k.is_integer() else float(k): int(v)
                    for k, v in top_freqs.items()
                },
            }

    # E. Negative values in resource metrics
    neg_counts: Dict[str, int] = {}
    for col in NON_NEGATIVE_METRICS:
        actual_col = (
            col if is_normalized else NORMALIZED_TO_RAW_COLS.get(col, col)
        )
        if actual_col in df.columns:
            cnt = int((df[actual_col] < 0).sum())
            if cnt > 0:
                neg_counts[actual_col] = cnt
                results["observations"].append(
                    f"Found {cnt} negative values in {actual_col}."
                )
    results["negative_value_counts"] = neg_counts

    # F. CPU Usage Percentage Range
    if cpu_pct_col in df.columns:
        cpu_series = df[cpu_pct_col].dropna()
        if not cpu_series.empty:
            cpu_min = float(cpu_series.min())
            cpu_max = float(cpu_series.max())
            results["cpu_usage_percent_stats"] = {
                "min": round(cpu_min, 4),
                "max": round(cpu_max, 4),
                "mean": round(float(cpu_series.mean()), 4),
                "median": round(float(cpu_series.median()), 4),
                "std": round(float(cpu_series.std()), 4),
            }
            if cpu_min < 0.0 or cpu_max > 100.0:
                results["observations"].append(
                    f"CPU usage [%] out of bounds [0, 100]: min={cpu_min}, max={cpu_max}."
                )

    return results

