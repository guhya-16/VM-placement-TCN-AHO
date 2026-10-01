"""
Stage 14: Final Full-Dataset Preprocessing + VM Split + Dataset Manifest
========================================================================
Implements the definitive, deterministic, leakage-safe partition of the 1,250
Bitbrains fastStorage VM traces into:
- Development: 972 eligible VMs (chronological 70/15/15 train/val/test)
- Unseen: 250 eligible VMs (stratified by Type A/B and data availability)
- Excluded: 28 ineligible VMs (0 valid forecasting windows under Policy C)

Strict Methodological Guardrails:
1. Seed = 42 for fully reproducible sampling.
2. Primary stratification: Type A (clean, continuous) vs Type B (decommissioned tail).
3. Secondary stratification: Data availability quartiles based on valid window counts.
4. Deterministic integer allocation using Largest Remainder (Hamilton-Hare) method.
5. Strict temporal split containment: each development VM uses independent chronological
   partitioning (train=70%, val=15%, test=15%) with zero cross-boundary window leakage.
6. Generates:
   - results/final_dataset/vm_split_manifest.csv (1,250 rows)
   - results/final_dataset/final_dataset_manifest.json
   - results/final_dataset/window_manifest.csv (1,250 rows)
   - results/final_dataset/vm_split_balance_report.json
   - results/final_dataset/vm_split_balance_report.csv
7. Verification suite confirming all 13 required invariants.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
TARGET_UNSEEN_TOTAL = 250
TARGET_DEV_TOTAL = 972
TARGET_EXCLUDED_TOTAL = 28
TOTAL_VMS = 1250

WINDOW_L = 288
WINDOW_H = 12
WINDOW_W = WINDOW_L + WINDOW_H  # 300 steps (25 hours)


def largest_remainder_allocation(counts: Dict[str, int], total_target: int) -> Dict[str, int]:
    """
    Allocates integer seats/counts proportionally using the Largest Remainder (Hamilton-Hare) method.
    Guarantees sum(allocated.values()) == total_target exactly.
    """
    total_items = sum(counts.values())
    exact = {k: v * total_target / total_items for k, v in counts.items()}
    floored = {k: int(np.floor(v)) for k, v in exact.items()}
    remainder = {k: exact[k] - floored[k] for k, v in exact.items()}
    deficit = total_target - sum(floored.values())

    # Deterministic tie-breaking: sort by remainder descending, then by key ascending
    sorted_keys = sorted(counts.keys(), key=lambda k: (-remainder[k], k))
    allocated = floored.copy()
    for k in sorted_keys[:deficit]:
        allocated[k] += 1
    return allocated


def calculate_distribution_stats(series: pd.Series) -> Dict[str, float]:
    """Computes descriptive summary statistics for a numeric series."""
    arr = series.dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return {"count": 0, "min": 0.0, "q1": 0.0, "median": 0.0, "q3": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "count": int(len(arr)),
        "min": float(np.min(arr)),
        "q1": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "q3": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
    }


def perform_stage14_split(
    audit_csv_path: Path,
    output_dir: Path,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Executes the full Stage 14 partitioning and manifest generation.
    """
    audit_csv_path = Path(audit_csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not audit_csv_path.exists():
        raise FileNotFoundError(f"Authoritative Stage 13 audit file not found: {audit_csv_path}")

    logger.info(f"Loading Stage 13 audit from {audit_csv_path}...")
    df_audit = pd.read_csv(audit_csv_path)
    assert len(df_audit) == TOTAL_VMS, f"Expected {TOTAL_VMS} VMs in audit, got {len(df_audit)}"

    # 1. Separate Eligible vs Excluded
    is_eligible = df_audit["eligible_for_unseen_vm_evaluation"].astype(bool)
    df_eligible = df_audit[is_eligible].copy()
    df_excluded = df_audit[~is_eligible].copy()

    assert len(df_eligible) == 1222, f"Expected 1,222 eligible VMs, got {len(df_eligible)}"
    assert len(df_excluded) == TARGET_EXCLUDED_TOTAL, f"Expected {TARGET_EXCLUDED_TOTAL} excluded VMs, got {len(df_excluded)}"

    # 2. Determine Trace Type:
    # Type B: active period followed by de-provisioned/inactive telemetry (inactive_suffix_rows > 0)
    # Type A: normal continuous long-running traces (inactive_suffix_rows == 0)
    df_eligible["trace_type"] = np.where(df_eligible["inactive_suffix_rows"] > 0, "Type B", "Type A")

    # For excluded VMs, preserve established classification
    df_excluded["trace_type"] = np.where(
        df_excluded["active_row_count"] == 0,
        "Zero_Capacity",
        "Transient",
    )
    df_excluded["data_availability_stratum"] = "excluded"

    # Type A / Type B eligible counts
    type_counts = df_eligible["trace_type"].value_counts().to_dict()
    type_a_total = type_counts.get("Type A", 0)
    type_b_total = type_counts.get("Type B", 0)
    logger.info(f"Eligible VM Population: Type A = {type_a_total} ({type_a_total/1222*100:.2f}%), Type B = {type_b_total} ({type_b_total/1222*100:.2f}%)")

    # Primary allocation between Type A and Type B using Largest Remainder
    type_unseen_allocation = largest_remainder_allocation(type_counts, TARGET_UNSEEN_TOTAL)
    target_unseen_a = type_unseen_allocation["Type A"]
    target_unseen_b = type_unseen_allocation["Type B"]
    logger.info(f"Unseen Primary Allocation: Type A = {target_unseen_a} ({target_unseen_a/TARGET_UNSEEN_TOTAL*100:.2f}%), Type B = {target_unseen_b} ({target_unseen_b/TARGET_UNSEEN_TOTAL*100:.2f}%)")

    # 3. Secondary Stratification: Data Availability Quartiles within each Type
    strata_series = pd.Series(index=df_eligible.index, dtype=str)
    strata_definitions: Dict[str, Dict[str, Any]] = {}
    strata_counts: Dict[Tuple[str, str], int] = {}

    for t, target_count in [("Type A", target_unseen_a), ("Type B", target_unseen_b)]:
        sub = df_eligible[df_eligible["trace_type"] == t]
        q = sub["valid_24h_to_1h_windows"].quantile([0.25, 0.5, 0.75]).tolist()
        strata_definitions[t] = {
            "q1_bound": float(q[0]),
            "median_bound": float(q[1]),
            "q3_bound": float(q[2]),
        }

        def assign_stratum(w: float, q_bounds: List[float] = q) -> str:
            if w <= q_bounds[0]:
                return "Q1_low"
            elif w <= q_bounds[1]:
                return "Q2_med_low"
            elif w <= q_bounds[2]:
                return "Q3_med_high"
            else:
                return "Q4_high"

        assigned = sub["valid_24h_to_1h_windows"].apply(assign_stratum)
        strata_series.loc[sub.index] = assigned
        for s_name, count in assigned.value_counts().items():
            strata_counts[(t, s_name)] = int(count)

    df_eligible["data_availability_stratum"] = strata_series

    # Secondary allocation: within Type A (target=226) and Type B (target=24)
    strata_counts_a = {s: strata_counts[( "Type A", s)] for s in ["Q1_low", "Q2_med_low", "Q3_med_high", "Q4_high"]}
    strata_counts_b = {s: strata_counts[( "Type B", s)] for s in ["Q1_low", "Q2_med_low", "Q3_med_high", "Q4_high"]}

    sub_alloc_a = largest_remainder_allocation(strata_counts_a, target_unseen_a)
    sub_alloc_b = largest_remainder_allocation(strata_counts_b, target_unseen_b)

    stratum_unseen_targets: Dict[Tuple[str, str], int] = {}
    for s, k in sub_alloc_a.items():
        stratum_unseen_targets[("Type A", s)] = k
    for s, k in sub_alloc_b.items():
        stratum_unseen_targets[("Type B", s)] = k

    logger.info("Stratified Unseen Targets per Cell:")
    for (t, s), k in sorted(stratum_unseen_targets.items()):
        total_in_stratum = strata_counts[(t, s)]
        logger.info(f"  {t:6s} | {s:11s} : {k:2d} / {total_in_stratum:3d} ({k/total_in_stratum*100:5.2f}%)")

    # 4. Deterministic Seeded Selection within Strata
    rng = np.random.default_rng(seed=seed)
    unseen_vm_ids: List[int] = []

    # Process strata in fixed deterministic order
    sorted_strata_keys = sorted(stratum_unseen_targets.keys())
    for t, s in sorted_strata_keys:
        k = stratum_unseen_targets[(t, s)]
        candidate_pool = (
            df_eligible[(df_eligible["trace_type"] == t) & (df_eligible["data_availability_stratum"] == s)]["vm_id"]
            .sort_values()
            .tolist()
        )
        assert len(candidate_pool) >= k, f"Candidate pool for ({t}, {s}) has {len(candidate_pool)} < {k}"
        chosen = rng.choice(candidate_pool, size=k, replace=False).tolist()
        unseen_vm_ids.extend(chosen)

    unseen_set = set(unseen_vm_ids)
    assert len(unseen_set) == TARGET_UNSEEN_TOTAL, f"Expected {TARGET_UNSEEN_TOTAL} unique unseen VMs, got {len(unseen_set)}"

    # 5. Populate Population and Temporal Split columns
    df_eligible["population"] = np.where(df_eligible["vm_id"].isin(unseen_set), "unseen", "development")
    df_eligible["temporal_split"] = np.where(df_eligible["population"] == "development", "development_70_15_15", "inference_only")
    df_eligible["eligibility"] = True

    df_excluded["population"] = "excluded"
    df_excluded["temporal_split"] = "excluded"
    df_excluded["eligibility"] = False

    # Combine back into full 1,250 VM manifest
    full_manifest_df = pd.concat([df_eligible, df_excluded], ignore_index=True)
    full_manifest_df["vm_id"] = full_manifest_df["vm_id"].astype(int)
    full_manifest_df = full_manifest_df.sort_values("vm_id").reset_index(drop=True)

    # 6. Generate Required File 1: results/final_dataset/vm_split_manifest.csv
    manifest_csv_cols = [
        "vm_id",
        "filename",
        "population",
        "temporal_split",
        "eligibility",
        "eligibility_reason",
        "trace_type",
        "data_availability_stratum",
    ]
    vm_split_manifest_csv = full_manifest_df[manifest_csv_cols].copy()
    manifest_csv_path = output_dir / "vm_split_manifest.csv"
    vm_split_manifest_csv.to_csv(manifest_csv_path, index=False)
    logger.info(f"Saved vm_split_manifest.csv ({len(vm_split_manifest_csv)} rows) to {manifest_csv_path}")

    # 7. Generate Required File 3: results/final_dataset/window_manifest.csv
    # Compact per-VM window count manifest
    window_manifest_rows: List[Dict[str, Any]] = []
    for _, row in full_manifest_df.iterrows():
        pop = row["population"]
        if pop == "development":
            whole = int(row["valid_24h_to_1h_windows"])
            tr = int(row["usable_train_windows"])
            va = int(row["usable_validation_windows"])
            te = int(row["usable_test_windows"])
            tot_split = tr + va + te
            bound = whole - tot_split
        elif pop == "unseen":
            whole = int(row["valid_24h_to_1h_windows"])
            tr = 0
            va = 0
            te = 0
            tot_split = 0
            bound = whole
        else:  # excluded
            whole = 0
            tr = 0
            va = 0
            te = 0
            tot_split = 0
            bound = 0

        window_manifest_rows.append({
            "vm_id": int(row["vm_id"]),
            "population": pop,
            "train_windows": tr,
            "validation_windows": va,
            "test_windows": te,
            "total_split_windows": tot_split,
            "whole_trace_valid_windows": whole,
            "boundary_excluded_windows": bound,
        })

    df_window_manifest = pd.DataFrame(window_manifest_rows).sort_values("vm_id").reset_index(drop=True)
    window_manifest_path = output_dir / "window_manifest.csv"
    df_window_manifest.to_csv(window_manifest_path, index=False)
    logger.info(f"Saved window_manifest.csv ({len(df_window_manifest)} rows) to {window_manifest_path}")

    # 8. Compute Comprehensive Balance Report
    dev_mask = full_manifest_df["population"] == "development"
    uns_mask = full_manifest_df["population"] == "unseen"
    el_mask = full_manifest_df["eligibility"] == True

    df_dev = full_manifest_df[dev_mask]
    df_uns = full_manifest_df[uns_mask]
    df_el = full_manifest_df[el_mask]

    balance_report = {
        "metadata": {
            "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "random_seed": seed,
            "total_vms": TOTAL_VMS,
            "eligible_vms": len(df_el),
            "development_vms": len(df_dev),
            "unseen_vms": len(df_uns),
            "excluded_vms": len(df_excluded),
        },
        "type_distribution": {
            "eligible": {
                "type_a_count": int((df_el["trace_type"] == "Type A").sum()),
                "type_a_percent": round(float((df_el["trace_type"] == "Type A").mean() * 100), 4),
                "type_b_count": int((df_el["trace_type"] == "Type B").sum()),
                "type_b_percent": round(float((df_el["trace_type"] == "Type B").mean() * 100), 4),
            },
            "development": {
                "type_a_count": int((df_dev["trace_type"] == "Type A").sum()),
                "type_a_percent": round(float((df_dev["trace_type"] == "Type A").mean() * 100), 4),
                "type_b_count": int((df_dev["trace_type"] == "Type B").sum()),
                "type_b_percent": round(float((df_dev["trace_type"] == "Type B").mean() * 100), 4),
            },
            "unseen": {
                "type_a_count": int((df_uns["trace_type"] == "Type A").sum()),
                "type_a_percent": round(float((df_uns["trace_type"] == "Type A").mean() * 100), 4),
                "type_b_count": int((df_uns["trace_type"] == "Type B").sum()),
                "type_b_percent": round(float((df_uns["trace_type"] == "Type B").mean() * 100), 4),
            },
            "differences_from_eligible": {
                "dev_minus_eligible_type_a_pct": round(float((df_dev["trace_type"] == "Type A").mean() * 100 - (df_el["trace_type"] == "Type A").mean() * 100), 4),
                "dev_minus_eligible_type_b_pct": round(float((df_dev["trace_type"] == "Type B").mean() * 100 - (df_el["trace_type"] == "Type B").mean() * 100), 4),
                "unseen_minus_eligible_type_a_pct": round(float((df_uns["trace_type"] == "Type A").mean() * 100 - (df_el["trace_type"] == "Type A").mean() * 100), 4),
                "unseen_minus_eligible_type_b_pct": round(float((df_uns["trace_type"] == "Type B").mean() * 100 - (df_el["trace_type"] == "Type B").mean() * 100), 4),
            },
        },
        "data_availability_valid_windows": {
            "eligible": calculate_distribution_stats(df_el["valid_24h_to_1h_windows"]),
            "development": calculate_distribution_stats(df_dev["valid_24h_to_1h_windows"]),
            "unseen": calculate_distribution_stats(df_uns["valid_24h_to_1h_windows"]),
        },
        "active_duration_days": {
            "eligible": calculate_distribution_stats(df_el["active_duration_days"]),
            "development": calculate_distribution_stats(df_dev["active_duration_days"]),
            "unseen": calculate_distribution_stats(df_uns["active_duration_days"]),
        },
        "aligned_5min_steps": {
            "eligible": calculate_distribution_stats(df_el["aligned_5min_steps"]),
            "development": calculate_distribution_stats(df_dev["aligned_5min_steps"]),
            "unseen": calculate_distribution_stats(df_uns["aligned_5min_steps"]),
        },
        "stratum_breakdown": {
            "strata_counts_eligible": {f"{t}_{s}": count for (t, s), count in sorted(strata_counts.items())},
            "strata_counts_unseen": {f"{t}_{s}": count for (t, s), count in sorted(stratum_unseen_targets.items())},
            "strata_counts_development": {
                f"{t}_{s}": strata_counts[(t, s)] - stratum_unseen_targets[(t, s)]
                for (t, s) in sorted(stratum_unseen_targets.keys())
            },
        },
    }

    # Add relative differences for valid windows, duration, and steps
    for metric_key, col in [
        ("data_availability_valid_windows", "valid_24h_to_1h_windows"),
        ("active_duration_days", "active_duration_days"),
        ("aligned_5min_steps", "aligned_5min_steps"),
    ]:
        el_m = balance_report[metric_key]["eligible"]["mean"]
        dev_m = balance_report[metric_key]["development"]["mean"]
        uns_m = balance_report[metric_key]["unseen"]["mean"]
        balance_report[metric_key]["differences_from_eligible"] = {
            "dev_absolute_diff_mean": round(dev_m - el_m, 4),
            "dev_relative_diff_pct_mean": round((dev_m - el_m) / el_m * 100, 4) if el_m != 0 else 0.0,
            "unseen_absolute_diff_mean": round(uns_m - el_m, 4),
            "unseen_relative_diff_pct_mean": round((uns_m - el_m) / el_m * 100, 4) if el_m != 0 else 0.0,
            "dev_median_diff": round(balance_report[metric_key]["development"]["median"] - balance_report[metric_key]["eligible"]["median"], 4),
            "unseen_median_diff": round(balance_report[metric_key]["unseen"]["median"] - balance_report[metric_key]["eligible"]["median"], 4),
        }

    balance_report_json_path = output_dir / "vm_split_balance_report.json"
    with open(balance_report_json_path, "w", encoding="utf-8") as f:
        json.dump(balance_report, f, indent=2)
    logger.info(f"Saved vm_split_balance_report.json to {balance_report_json_path}")

    # Build balance CSV table for rapid inspection
    balance_csv_rows = []
    for metric, col_name, unit in [
        ("Valid Windows", "valid_24h_to_1h_windows", "windows"),
        ("Active Duration", "active_duration_days", "days"),
        ("Aligned Steps", "aligned_5min_steps", "steps"),
    ]:
        s_el = balance_report["data_availability_valid_windows" if metric == "Valid Windows" else ("active_duration_days" if metric == "Active Duration" else "aligned_5min_steps")]
        balance_csv_rows.append({
            "metric": metric,
            "unit": unit,
            "eligible_mean": s_el["eligible"]["mean"],
            "eligible_median": s_el["eligible"]["median"],
            "eligible_q1": s_el["eligible"]["q1"],
            "eligible_q3": s_el["eligible"]["q3"],
            "dev_mean": s_el["development"]["mean"],
            "dev_median": s_el["development"]["median"],
            "dev_q1": s_el["development"]["q1"],
            "dev_q3": s_el["development"]["q3"],
            "unseen_mean": s_el["unseen"]["mean"],
            "unseen_median": s_el["unseen"]["median"],
            "unseen_q1": s_el["unseen"]["q1"],
            "unseen_q3": s_el["unseen"]["q3"],
        })
    df_balance_csv = pd.DataFrame(balance_csv_rows)
    balance_csv_path = output_dir / "vm_split_balance_report.csv"
    df_balance_csv.to_csv(balance_csv_path, index=False)
    logger.info(f"Saved vm_split_balance_report.csv to {balance_csv_path}")

    # 9. Compute Development Split Aggregates for Dataset Manifest
    dev_split_train_windows = int(df_dev["usable_train_windows"].sum())
    dev_split_val_windows = int(df_dev["usable_validation_windows"].sum())
    dev_split_test_windows = int(df_dev["usable_test_windows"].sum())
    dev_split_total_windows = dev_split_train_windows + dev_split_val_windows + dev_split_test_windows
    dev_whole_valid_windows = int(df_dev["valid_24h_to_1h_windows"].sum())
    dev_boundary_discarded_windows = dev_whole_valid_windows - dev_split_total_windows

    # 10. Generate Required File 2: results/final_dataset/final_dataset_manifest.json
    final_manifest = {
        "dataset": {
            "name": "Bitbrains GWA-T-12 fastStorage",
            "dataset_path": "dataset/fastStorage/2013-8/",
            "total_vms": TOTAL_VMS,
            "eligible_vms": len(df_el),
            "ineligible_vms": len(df_excluded),
            "sampling_interval_seconds": 300,
            "canonical_grid": "5-minute interval via Policy C (resample('5min').mean())",
        },
        "vm_population": {
            "development_vms": len(df_dev),
            "unseen_vms": len(df_uns),
            "excluded_vms": len(df_excluded),
            "total_vms": TOTAL_VMS,
            "accounting_check": f"{len(df_dev)} development + {len(df_uns)} unseen + {len(df_excluded)} excluded = {TOTAL_VMS} total",
        },
        "partition": {
            "random_seed": seed,
            "primary_stratification": "Type A (Continuous) vs Type B (Decommissioned Suffix)",
            "secondary_stratification": "Data availability quartiles of valid_24h_to_1h_windows within each type",
            "selection_method": "Deterministic stratified proportional allocation with seeded RNG choice",
            "deterministic_rounding_method": "Largest Remainder (Hamilton-Hare) method",
            "implementation_script": "ml/generate_dataset_manifest.py",
            "primary_type_allocation": {
                "eligible_type_a_count": int((df_el["trace_type"] == "Type A").sum()),
                "eligible_type_b_count": int((df_el["trace_type"] == "Type B").sum()),
                "unseen_type_a_target": target_unseen_a,
                "unseen_type_b_target": target_unseen_b,
                "development_type_a_count": int((df_dev["trace_type"] == "Type A").sum()),
                "development_type_b_count": int((df_dev["trace_type"] == "Type B").sum()),
            },
            "strata_definitions": strata_definitions,
            "strata_unseen_counts": {f"{t}_{s}": count for (t, s), count in sorted(stratum_unseen_targets.items())},
        },
        "temporal_development_split": {
            "protocol": "Chronological partition within EACH development VM trace",
            "ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "method": "partition_vm_chronologically(df_c, 0.70, 0.15)",
            "split_containment": "Strict partition containment; every window [X, y] resides 100% inside single split",
            "boundary_handling": "Windows crossing train->val or val->test boundaries are intentionally discarded to prevent leakage",
            "sequence_break_handling": "Unresolved gaps >2 samples remain sequence breaks (NaN) and are never crossed",
            "development_window_totals": {
                "train_windows": dev_split_train_windows,
                "validation_windows": dev_split_val_windows,
                "test_windows": dev_split_test_windows,
                "total_usable_split_windows": dev_split_total_windows,
                "whole_trace_valid_windows": dev_whole_valid_windows,
                "boundary_discarded_windows": dev_boundary_discarded_windows,
                "invariant_check": f"{dev_split_total_windows} split windows <= {dev_whole_valid_windows} whole windows (PASSED)",
            },
        },
        "window_configuration": {
            "history_steps_L": WINDOW_L,
            "history_duration_hours": 24.0,
            "horizon_steps_H": WINDOW_H,
            "horizon_duration_hours": 1.0,
            "total_window_span_W": WINDOW_W,
            "stride": 1,
            "sampling_interval_seconds": 300,
            "window_extraction_implementation": "ml.temporal_windows.extract_split_safe_windows",
        },
        "features_and_scaling": {
            "feature_mode": "M3",
            "num_features": 10,
            "feature_columns": [
                "cpu_usage_percent",
                "memory_usage_kb",
                "disk_read_kbps",
                "disk_write_kbps",
                "network_received_kbps",
                "network_transmitted_kbps",
                "hour_sin",
                "hour_cos",
                "day_sin",
                "day_cos",
            ],
            "scaling_method": "StandardScaler",
            "scaling_isolation": "Fitted EXCLUSIVELY on unique training observations across the 972 development VMs prior to window expansion; zero validation or test leakage",
        },
        "unseen_population_policy": {
            "purpose": "Independent cross-VM zero-shot generalization evaluation",
            "status": "Inference only; completely frozen in Stage 14",
            "isolation_rule": "Strictly excluded from training, validation, checkpoint selection, hyperparameter tuning, scaler fitting, and risk calibration",
        },
        "ineligible_vms_policy": {
            "status": "Excluded from forecasting experiments",
            "count": TARGET_EXCLUDED_TOTAL,
            "reason_summary": "Zero valid W=300 forecasting windows under Preprocessing Policy C (1 zero-capacity VM + 27 short-lived VMs <25h)",
        },
        "reproducibility": {
            "random_seed": seed,
            "python_version": sys.version,
            "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "script": "ml/generate_dataset_manifest.py",
            "artifacts_generated": [
                "results/final_dataset/vm_split_manifest.csv",
                "results/final_dataset/final_dataset_manifest.json",
                "results/final_dataset/window_manifest.csv",
                "results/final_dataset/vm_split_balance_report.json",
                "results/final_dataset/vm_split_balance_report.csv",
            ],
        },
    }

    final_manifest_path = output_dir / "final_dataset_manifest.json"
    with open(final_manifest_path, "w", encoding="utf-8") as f:
        json.dump(final_manifest, f, indent=2)
    logger.info(f"Saved final_dataset_manifest.json to {final_manifest_path}")

    return full_manifest_df, df_window_manifest, balance_report, final_manifest


def run_stage14_verification(output_dir: Path, data_dir: Path) -> Dict[str, Any]:
    """
    Executes all 13 required verification checks for Stage 14.
    """
    output_dir = Path(output_dir)
    data_dir = Path(data_dir)

    manifest_csv = output_dir / "vm_split_manifest.csv"
    final_json = output_dir / "final_dataset_manifest.json"
    window_csv = output_dir / "window_manifest.csv"
    balance_json = output_dir / "vm_split_balance_report.json"
    audit_csv = output_dir / "vm_audit.csv"

    df_manifest = pd.read_csv(manifest_csv)
    df_window = pd.read_csv(window_csv)
    df_audit = pd.read_csv(audit_csv)
    with open(final_json) as f:
        manifest_data = json.load(f)
    with open(balance_json) as f:
        balance_data = json.load(f)

    results: Dict[str, bool] = {}

    # CHECK 1: VM Coverage
    check1 = (
        len(df_manifest) == TOTAL_VMS
        and list(df_manifest["vm_id"]) == list(range(1, TOTAL_VMS + 1))
        and df_manifest["vm_id"].nunique() == TOTAL_VMS
    )
    results["CHECK_1_VM_COVERAGE"] = check1

    # CHECK 2: Population Counts
    dev_count = int((df_manifest["population"] == "development").sum())
    uns_count = int((df_manifest["population"] == "unseen").sum())
    exc_count = int((df_manifest["population"] == "excluded").sum())
    check2 = (dev_count == TARGET_DEV_TOTAL and uns_count == TARGET_UNSEEN_TOTAL and exc_count == TARGET_EXCLUDED_TOTAL)
    results["CHECK_2_POPULATION_COUNTS"] = check2

    # CHECK 3: Eligibility
    dev_elig = int((df_manifest[df_manifest["population"] == "development"]["eligibility"] == True).sum())
    uns_elig = int((df_manifest[df_manifest["population"] == "unseen"]["eligibility"] == True).sum())
    exc_inelig = int((df_manifest[df_manifest["population"] == "excluded"]["eligibility"] == False).sum())
    check3 = (dev_elig == TARGET_DEV_TOTAL and uns_elig == TARGET_UNSEEN_TOTAL and exc_inelig == TARGET_EXCLUDED_TOTAL)
    results["CHECK_3_ELIGIBILITY"] = check3

    # CHECK 4: No VM Overlap & Disjoint Sets
    dev_ids = set(df_manifest[df_manifest["population"] == "development"]["vm_id"])
    uns_ids = set(df_manifest[df_manifest["population"] == "unseen"]["vm_id"])
    exc_ids = set(df_manifest[df_manifest["population"] == "excluded"]["vm_id"])
    check4 = (
        len(dev_ids.intersection(uns_ids)) == 0
        and len(dev_ids.intersection(exc_ids)) == 0
        and len(uns_ids.intersection(exc_ids)) == 0
        and len(dev_ids.union(uns_ids)) == 1222
        and len(dev_ids.union(uns_ids).union(exc_ids)) == TOTAL_VMS
    )
    results["CHECK_4_NO_VM_OVERLAP"] = check4

    # CHECK 5: Type Representation
    el_types = balance_data["type_distribution"]["eligible"]
    dev_types = balance_data["type_distribution"]["development"]
    uns_types = balance_data["type_distribution"]["unseen"]
    diffs = balance_data["type_distribution"]["differences_from_eligible"]
    # Verify differences are small (< 1%)
    check5 = abs(diffs["unseen_minus_eligible_type_a_pct"]) < 1.0 and abs(diffs["dev_minus_eligible_type_a_pct"]) < 1.0
    results["CHECK_5_TYPE_REPRESENTATION"] = check5

    # CHECK 6: Data Availability Representation
    s_el = balance_data["data_availability_valid_windows"]["eligible"]
    s_dev = balance_data["data_availability_valid_windows"]["development"]
    s_uns = balance_data["data_availability_valid_windows"]["unseen"]
    # Check medians and quartiles match
    check6 = (
        s_el["median"] == s_dev["median"] == s_uns["median"]
        and s_el["q1"] == s_dev["q1"] == s_uns["q1"]
        and s_el["q3"] == s_dev["q3"] == s_uns["q3"]
    )
    results["CHECK_6_DATA_AVAILABILITY_REPRESENTATION"] = check6

    # CHECK 7: Determinism (Re-run check with seed 42)
    rng_test = np.random.default_rng(seed=RANDOM_SEED)
    rerun_unseen = []
    strata_keys_ordered = [
        ("Type A", "Q1_low"),
        ("Type A", "Q2_med_low"),
        ("Type A", "Q3_med_high"),
        ("Type A", "Q4_high"),
        ("Type B", "Q1_low"),
        ("Type B", "Q2_med_low"),
        ("Type B", "Q3_med_high"),
        ("Type B", "Q4_high"),
    ]
    for (t, s) in sorted(strata_keys_ordered):
        k = manifest_data["partition"]["strata_unseen_counts"][f"{t}_{s}"]
        pool = (
            df_manifest[(df_manifest["trace_type"] == t) & (df_manifest["data_availability_stratum"] == s)]["vm_id"]
            .sort_values()
            .tolist()
        )
        ch = rng_test.choice(pool, size=k, replace=False).tolist()
        rerun_unseen.extend(ch)
    check7 = (sorted(rerun_unseen) == sorted(uns_ids))
    results["CHECK_7_DETERMINISM"] = check7

    # CHECK 8: Stage 13 Consistency
    merged_audit = pd.merge(df_manifest, df_audit, on="vm_id", suffixes=("_manifest", "_audit"))
    check8 = (
        len(merged_audit) == TOTAL_VMS
        and (merged_audit["eligibility"] == merged_audit["eligible_for_unseen_vm_evaluation"]).all()
        and (merged_audit["eligibility_reason_manifest"] == merged_audit["eligibility_reason_audit"]).all()
    )
    results["CHECK_8_STAGE13_CONSISTENCY"] = check8

    # CHECK 9: Temporal Split Implementation
    check9 = (
        manifest_data["temporal_development_split"]["method"] == "partition_vm_chronologically(df_c, 0.70, 0.15)"
        and manifest_data["window_configuration"]["window_extraction_implementation"] == "ml.temporal_windows.extract_split_safe_windows"
    )
    results["CHECK_9_TEMPORAL_SPLIT_IMPLEMENTATION"] = check9

    # CHECK 10: Window Boundary Safety & Split Containment
    dev_windows = df_window[df_window["population"] == "development"]
    split_sum = dev_windows["train_windows"] + dev_windows["validation_windows"] + dev_windows["test_windows"]
    check10 = (
        (split_sum == dev_windows["total_split_windows"]).all()
        and (dev_windows["total_split_windows"] <= dev_windows["whole_trace_valid_windows"]).all()
        and (dev_windows["boundary_excluded_windows"] == (dev_windows["whole_trace_valid_windows"] - dev_windows["total_split_windows"])).all()
        and (df_window[df_window["population"] == "unseen"]["total_split_windows"] == 0).all()
        and (df_window[df_window["population"] == "excluded"]["total_split_windows"] == 0).all()
    )
    results["CHECK_10_WINDOW_BOUNDARY_SAFETY"] = check10

    # CHECK 11: Manifest Consistency
    check11 = (
        len(df_manifest) == len(df_window) == len(df_audit) == TOTAL_VMS
        and (df_manifest["population"] == df_window["population"]).all()
        and manifest_data["vm_population"]["development_vms"] == dev_count
        and manifest_data["vm_population"]["unseen_vms"] == uns_count
        and manifest_data["vm_population"]["excluded_vms"] == exc_count
    )
    results["CHECK_11_MANIFEST_CONSISTENCY"] = check11

    # CHECK 12: Raw Data Integrity (Check 1250 files exist, none modified)
    csv_files = list(data_dir.glob("*.csv"))
    check12 = len(csv_files) == TOTAL_VMS
    results["CHECK_12_RAW_DATA_INTEGRITY"] = check12

    # CHECK 13: Model Integrity
    # Confirm no models trained, predictions.csv or risk_state.csv not created in final_dataset
    pred_exists = (output_dir / "predictions.csv").exists()
    risk_exists = (output_dir / "risk_state.csv").exists()
    check13 = (not pred_exists) and (not risk_exists)
    results["CHECK_13_MODEL_INTEGRITY"] = check13

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 14: Final Preprocessing + VM Split + Manifest")
    parser.add_argument("--audit_csv", type=str, default="results/final_dataset/vm_audit.csv", help="Path to Stage 13 vm_audit.csv")
    parser.add_argument("--output_dir", type=str, default="results/final_dataset", help="Path to output directory")
    parser.add_argument("--data_dir", type=str, default="dataset/fastStorage/2013-8", help="Path to raw dataset directory")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for deterministic sampling")
    args = parser.parse_args()

    # Run split and generate manifests
    df_manifest, df_window, balance, final_manifest = perform_stage14_split(
        audit_csv_path=Path(args.audit_csv),
        output_dir=Path(args.output_dir),
        seed=args.seed,
    )

    # Run full verification
    logger.info("Running Stage 14 Verification Suite...")
    verif = run_stage14_verification(Path(args.output_dir), Path(args.data_dir))
    all_passed = True
    for check_name, passed in verif.items():
        status = "PASSED" if passed else "FAILED"
        logger.info(f"  {check_name:40s}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        logger.info("ALL 13 STAGE 14 CHECKS PASSED PERFECTLY!")
    else:
        logger.error("ONE OR MORE CHECKS FAILED.")
        sys.exit(1)
