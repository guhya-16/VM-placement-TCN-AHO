"""
Stage 4: Exploratory Data Analysis (EDA) & Temporal Workload Profiling
====================================================================
Performs rigorous exploratory data analysis on the preprocessed active Bitbrains
workloads (Policy C aligned 5-minute grid) across the controlled 14-VM cohort.

Key Responsibilities:
1. Quantitative profiling of CPU usage (mean, std, median, min, max, percentiles,
   zero-fraction, high-utilization fraction, coefficient of variation, skewness, kurtosis).
2. Dynamic resource metrics characterization (Memory, Disk I/O, Network I/O, CPU MHz).
3. Empirical correlation recomputation against cpu_usage_percent:
   - Cohort-level pooled correlations (descriptive macro analysis)
   - Within-VM correlation analysis (mean, median, min, max across individual VMs)
4. Temporal dynamics analysis: rolling mean, rolling volatility (1h, 6h), burst behavior.
5. Objective Autocorrelation Function (ACF) evaluation up to lag 288 (24 hours).
6. Generation of 6 publication-quality figures in results/stage4/figures/.
7. Export of eda_summary.csv, feature_analysis.csv, and within_vm_correlations.csv.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ml.data_loader import load_bitbrains_vm
from ml.preprocessing_evaluation import (
    REPRESENTATIVE_VMS,
    evaluate_candidate_active_rule,
    evaluate_policy_c_active_aligned,
)

def get_vm_id(filename: str) -> str:
    num_str = Path(filename).stem
    return f"VM_{int(num_str):03d}" if num_str.isdigit() else f"VM_{num_str}"


def compute_vm_eda_statistics(df_aligned: pd.DataFrame, vm_info: Dict[str, Any]) -> Dict[str, Any]:
    """Computes comprehensive descriptive and distributional statistics for a single VM."""
    vm_id = get_vm_id(vm_info["file"])
    cpu = df_aligned["cpu_usage_percent"].dropna()
    N = len(cpu)

    if N == 0:
        return {"vm_id": vm_id, "file": vm_info["file"], "category": vm_info["category"], "observations": 0}

    p25, p50, p75, p90, p95, p99 = np.percentile(cpu, [25, 50, 75, 90, 95, 99])

    mean_val = float(cpu.mean())
    std_val = float(cpu.std())
    min_val = float(cpu.min())
    max_val = float(cpu.max())
    skew_val = float(stats.skew(cpu)) if std_val > 0 else 0.0
    kurt_val = float(stats.kurtosis(cpu)) if std_val > 0 else 0.0
    cv_val = float(std_val / mean_val) if mean_val > 0 else 0.0

    zero_frac = float((cpu == 0.0).mean())
    high_util_frac = float((cpu > 80.0).mean())
    moderate_util_frac = float(((cpu >= 20.0) & (cpu <= 80.0)).mean())
    low_util_frac = float(((cpu > 0.0) & (cpu < 20.0)).mean())

    cpu_cap = float(df_aligned["cpu_capacity_mhz"].iloc[0]) if "cpu_capacity_mhz" in df_aligned.columns else None
    mem_cap_gb = (
        float(df_aligned["memory_capacity_kb"].iloc[0]) / (1024.0 * 1024.0)
        if "memory_capacity_kb" in df_aligned.columns
        else None
    )
    cores = float(df_aligned["cpu_cores"].iloc[0]) if "cpu_cores" in df_aligned.columns else None

    mem_mean_gb = (
        float(df_aligned["memory_usage_kb"].mean()) / (1024.0 * 1024.0)
        if "memory_usage_kb" in df_aligned.columns
        else None
    )
    disk_read_mean = float(df_aligned["disk_read_kbps"].mean()) if "disk_read_kbps" in df_aligned.columns else None
    disk_write_mean = float(df_aligned["disk_write_kbps"].mean()) if "disk_write_kbps" in df_aligned.columns else None
    net_rx_mean = float(df_aligned["network_received_kbps"].mean()) if "network_received_kbps" in df_aligned.columns else None
    net_tx_mean = float(df_aligned["network_transmitted_kbps"].mean()) if "network_transmitted_kbps" in df_aligned.columns else None

    span_seconds = int(df_aligned["timestamp_raw"].iloc[-1] - df_aligned["timestamp_raw"].iloc[0]) if N > 1 else 0
    span_days = round(span_seconds / 86400.0, 2)

    return {
        "vm_id": vm_id,
        "file": vm_info["file"],
        "category": vm_info["category"],
        "description": vm_info["desc"],
        "observations": N,
        "span_days": span_days,
        "cpu_cores": cores,
        "cpu_capacity_mhz": round(cpu_cap, 2) if cpu_cap else None,
        "memory_capacity_gb": round(mem_cap_gb, 2) if mem_cap_gb else None,
        "cpu_mean_pct": round(mean_val, 3),
        "cpu_std_pct": round(std_val, 3),
        "cpu_min_pct": round(min_val, 3),
        "cpu_p25_pct": round(p25, 3),
        "cpu_median_pct": round(p50, 3),
        "cpu_p75_pct": round(p75, 3),
        "cpu_p90_pct": round(p90, 3),
        "cpu_p95_pct": round(p95, 3),
        "cpu_p99_pct": round(p99, 3),
        "cpu_max_pct": round(max_val, 3),
        "cpu_cv": round(cv_val, 3),
        "cpu_skewness": round(skew_val, 3),
        "cpu_kurtosis": round(kurt_val, 3),
        "zero_fraction": round(zero_frac, 4),
        "low_util_fraction": round(low_util_frac, 4),
        "moderate_util_fraction": round(moderate_util_frac, 4),
        "high_util_fraction": round(high_util_frac, 4),
        "memory_usage_mean_gb": round(mem_mean_gb, 3) if mem_mean_gb is not None else None,
        "disk_read_mean_kbps": round(disk_read_mean, 2) if disk_read_mean is not None else None,
        "disk_write_mean_kbps": round(disk_write_mean, 2) if disk_write_mean is not None else None,
        "network_rx_mean_kbps": round(net_rx_mean, 2) if net_rx_mean is not None else None,
        "network_tx_mean_kbps": round(net_tx_mean, 2) if net_tx_mean is not None else None,
        "interpolated_steps": int(df_aligned["is_interpolated"].sum()) if "is_interpolated" in df_aligned.columns else 0,
    }


def compute_autocorrelation(series: pd.Series, max_lag: int = 288) -> np.ndarray:
    """Computes sample autocorrelation function up to max_lag."""
    s = series.dropna().to_numpy()
    n = len(s)
    if n <= max_lag:
        max_lag = n - 1
    if max_lag <= 0:
        return np.array([1.0])

    s_centered = s - np.mean(s)
    variance = np.var(s)
    if variance == 0:
        return np.ones(max_lag + 1)

    r = np.correlate(s_centered, s_centered, mode="full")
    r = r[n - 1 : n + max_lag] / (variance * np.arange(n, n - max_lag - 1, -1))
    return r


def analyze_features_and_correlations(aligned_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Recomputes empirical correlations, skewness, and zero-fractions across the pooled cohort
    to establish the descriptive macro-level feature candidate table.
    """
    pool_frames = []
    for vm_file, df in aligned_dfs.items():
        if len(df) > 0:
            df_copy = df.copy()
            df_copy["source_vm"] = vm_file
            pool_frames.append(df_copy)

    df_pooled = pd.concat(pool_frames, ignore_index=True)

    candidate_features = [
        ("cpu_usage_percent", "%", "Target & Autoregressive Input", "INCLUDED",
         "Primary prediction target (t+1..t+12) and principal autoregressive input (t-287..t)."),
        ("cpu_usage_mhz", "MHz", "Resource Telemetry", "EXCLUDED",
         "Within each VM, CPU usage MHz is deterministically proportional to CPU usage %, because CPU capacity is constant during the active lifetime."),
        ("cpu_capacity_mhz", "MHz", "Static Provisioning", "EXCLUDED (Metadata)",
         "Constant throughout active lifetime (std=0 within VM); causes division by zero in standard scalers."),
        ("cpu_cores", "count", "Static Provisioning", "EXCLUDED (Metadata)",
         "Static infrastructure configuration; zero temporal variance. Preserved as VM metadata."),
        ("memory_usage_kb", "KB", "Resource Telemetry", "INCLUDED",
         "Dynamic memory consumption; informative multi-resource context for memory-bound tasks."),
        ("memory_capacity_kb", "KB", "Static Provisioning", "EXCLUDED (Metadata)",
         "Static allocated RAM capacity (std=0 within VM); preserved as VM metadata."),
        ("disk_read_kbps", "KB/s", "I/O Throughput", "INCLUDED",
         "I/O read throughput; captures storage-intensive batch operations. Highly zero-inflated."),
        ("disk_write_kbps", "KB/s", "I/O Throughput", "INCLUDED",
         "I/O write throughput; captures computational log/file flushes correlated with CPU load."),
        ("network_received_kbps", "KB/s", "Network Traffic", "INCLUDED",
         "Ingress network traffic; leading indicator of incoming client application requests."),
        ("network_transmitted_kbps", "KB/s", "Network Traffic", "INCLUDED",
         "Egress network traffic; reflects server response volume."),
        ("hour_sin", "[-1, 1]", "Temporal Cyclical", "INCLUDED",
         "sin(2pi * hour_utc / 24); continuous smooth diurnal cycle encoding. Zero temporal leakage."),
        ("hour_cos", "[-1, 1]", "Temporal Cyclical", "INCLUDED",
         "cos(2pi * hour_utc / 24); cosine counterpart for continuous time-of-day representation."),
        ("day_sin", "[-1, 1]", "Temporal Cyclical", "INCLUDED",
         "sin(2pi * day_utc / 7); continuous encoding of weekly business vs weekend workload cycles."),
        ("day_cos", "[-1, 1]", "Temporal Cyclical", "INCLUDED",
         "cos(2pi * day_utc / 7); cosine counterpart for continuous weekly periodicity."),
    ]

    dt_utc = pd.to_datetime(df_pooled["timestamp_raw"], unit="s", utc=True)
    hour = dt_utc.dt.hour + dt_utc.dt.minute / 60.0
    day = dt_utc.dt.dayofweek
    df_pooled["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df_pooled["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df_pooled["day_sin"] = np.sin(2 * np.pi * day / 7.0)
    df_pooled["day_cos"] = np.cos(2 * np.pi * day / 7.0)

    rows = []
    target_series = df_pooled["cpu_usage_percent"].dropna()

    for col_name, unit, role, status, justification in candidate_features:
        if col_name not in df_pooled.columns:
            continue

        s = df_pooled[col_name].dropna()
        std_val = float(s.std())
        zero_pct = float((s == 0).mean() * 100)
        skew_val = float(stats.skew(s)) if std_val > 0 else 0.0
        kurt_val = float(stats.kurtosis(s)) if std_val > 0 else 0.0

        if std_val == 0 or target_series.std() == 0:
            pearson_r = np.nan
            spearman_r = np.nan
        else:
            valid_idx = s.index.intersection(target_series.index)
            pearson_r = float(stats.pearsonr(s.loc[valid_idx], target_series.loc[valid_idx])[0])
            spearman_r = float(stats.spearmanr(s.loc[valid_idx], target_series.loc[valid_idx])[0])

        rows.append({
            "feature_name": col_name,
            "unit": unit,
            "role": role,
            "inclusion_status": status,
            "pearson_r_with_cpu": round(pearson_r, 4) if not np.isnan(pearson_r) else "NaN (std=0)",
            "spearman_r_with_cpu": round(spearman_r, 4) if not np.isnan(spearman_r) else "NaN (std=0)",
            "std_dev": round(std_val, 2),
            "zero_percentage": round(zero_pct, 2),
            "skewness": round(skew_val, 2),
            "kurtosis": round(kurt_val, 2),
            "temporal_leakage_audit": "Zero Leakage: Derived strictly from historical t or UTC timestamp t.",
            "methodological_justification": justification,
        })

    return pd.DataFrame(rows)


def compute_within_vm_correlations(aligned_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Computes within-VM correlation analysis for dynamic candidate features across the 14 VMs.
    Prevents confusing pooled between-VM effects with individual within-VM workload relationships.
    """
    features = [
        "cpu_usage_mhz", "memory_usage_kb", "disk_read_kbps",
        "disk_write_kbps", "network_received_kbps", "network_transmitted_kbps"
    ]
    vm_corrs = {f: {"pearson": [], "spearman": []} for f in features}

    for fname, df_c in aligned_dfs.items():
        if len(df_c) < 10 or "cpu_usage_percent" not in df_c.columns:
            continue

        target = df_c["cpu_usage_percent"]
        if target.std() == 0:
            continue

        for f in features:
            if f not in df_c.columns:
                continue
            s = df_c[f]
            if s.std() == 0 or s.isna().all():
                continue
            valid = df_c[[f, "cpu_usage_percent"]].dropna()
            if len(valid) > 2 and valid[f].std() > 0 and valid["cpu_usage_percent"].std() > 0:
                p_r, _ = stats.pearsonr(valid[f], valid["cpu_usage_percent"])
                s_r, _ = stats.spearmanr(valid[f], valid["cpu_usage_percent"])
                vm_corrs[f]["pearson"].append(p_r)
                vm_corrs[f]["spearman"].append(s_r)

    rows = []
    for f in features:
        p_vals = [x for x in vm_corrs[f]["pearson"] if not np.isnan(x)]
        s_vals = [x for x in vm_corrs[f]["spearman"] if not np.isnan(x)]
        rows.append({
            "feature": f,
            "valid_vms": len(p_vals),
            "mean_within_pearson": round(float(np.mean(p_vals)), 4) if p_vals else np.nan,
            "median_within_pearson": round(float(np.median(p_vals)), 4) if p_vals else np.nan,
            "min_within_pearson": round(float(np.min(p_vals)), 4) if p_vals else np.nan,
            "max_within_pearson": round(float(np.max(p_vals)), 4) if p_vals else np.nan,
            "mean_within_spearman": round(float(np.mean(s_vals)), 4) if s_vals else np.nan,
            "median_within_spearman": round(float(np.median(s_vals)), 4) if s_vals else np.nan,
        })

    return pd.DataFrame(rows)


def generate_stage4_plots(
    aligned_dfs: Dict[str, pd.DataFrame],
    eda_df: pd.DataFrame,
    fig_dir: Path,
):
    """Generates the 6 required publication-quality figures."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})

    # Figure 1: CPU Distribution Boxplots & KDE across Regimes
    selected_vms = ["VM_001", "VM_003", "VM_011", "VM_012", "VM_013", "VM_023", "VM_025", "VM_002"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    box_data = []
    box_labels = []
    for vm_id in selected_vms:
        row = eda_df[eda_df["vm_id"] == vm_id]
        if not row.empty:
            f_name = row["file"].iloc[0]
            if f_name in aligned_dfs:
                box_data.append(aligned_dfs[f_name]["cpu_usage_percent"].dropna().values)
                box_labels.append(vm_id)

    axes[0].boxplot(box_data, tick_labels=box_labels, patch_artist=True,
                    boxprops=dict(facecolor="#cce5ff", color="#1f77b4"),
                    medianprops=dict(color="#d62728", lw=1.5),
                    flierprops=dict(marker=".", markersize=2, alpha=0.3, color="#7f7f7f"))
    axes[0].set_title("Figure 1A: CPU Usage [%] Distribution by Workload Regime", fontweight="bold", fontsize=11)
    axes[0].set_ylabel("CPU Usage [%]")
    axes[0].set_xlabel("VM Identifier")
    axes[0].grid(True, linestyle=":", alpha=0.5)

    kde_targets = [("1.csv", "VM_001 (Bursty/Peak)", "#d62728"),
                   ("3.csv", "VM_003 (Steady Moderate)", "#2ca02c"),
                   ("12.csv", "VM_012 (Low Load)", "#1f77b4"),
                   ("25.csv", "VM_025 (Heavy Sustained)", "#9467bd")]
    for fname, label, color in kde_targets:
        if fname in aligned_dfs:
            s = aligned_dfs[fname]["cpu_usage_percent"].dropna()
            kde = stats.gaussian_kde(s)
            x_eval = np.linspace(0, min(100, s.max() + 5), 300)
            axes[1].plot(x_eval, kde(x_eval), label=label, color=color, lw=1.8)
            axes[1].fill_between(x_eval, kde(x_eval), alpha=0.15, color=color)

    axes[1].set_title("Figure 1B: Kernel Density Estimates (KDE) of CPU Utilization", fontweight="bold", fontsize=11)
    axes[1].set_xlabel("CPU Usage [%]")
    axes[1].set_ylabel("Probability Density")
    axes[1].legend(loc="upper right", frameon=True, fontsize=9)
    axes[1].grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    fig.savefig(fig_dir / "cpu_distribution.png", dpi=200)
    plt.close(fig)

    # Figure 2: Workload Timelines across 5 Representative Regimes
    fig, axes = plt.subplots(5, 1, figsize=(14, 11), sharey=True)
    regime_examples = [
        ("1.csv", "VM_001 (Bursty / High-Peak)", "#1f77b4"),
        ("3.csv", "VM_003 (Steady Moderate Workload)", "#2ca02c"),
        ("13.csv", "VM_013 (Idle Provisioned Node)", "#7f7f7f"),
        ("2.csv", "VM_002 (Type B Active Lifetime — 1.89 Days)", "#ff7f0e"),
        ("1209.csv", "VM_1209 (Transient Ephemeral Task — 9.75 Hours)", "#9467bd"),
    ]

    for ax, (fname, title, color) in zip(axes, regime_examples):
        if fname in aligned_dfs:
            df_vm = aligned_dfs[fname]
            ax.plot(df_vm["timestamp"], df_vm["cpu_usage_percent"], color=color, lw=0.9, alpha=0.85)
            ax.set_title(title, fontweight="bold", fontsize=10, loc="left")
            ax.set_ylabel("CPU %")
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.set_ylim(-2, 105)

    axes[-1].set_xlabel("Timeline (UTC)")
    plt.suptitle("Figure 2: Empirical CPU Utilization Profiles Across Distinct Operational Regimes",
                 fontweight="bold", fontsize=12, y=0.995)
    plt.tight_layout()
    fig.savefig(fig_dir / "cpu_workload_examples.png", dpi=200)
    plt.close(fig)

    # Figure 3: Rolling Volatility & Mean Envelope
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    df_burst = aligned_dfs["1.csv"].copy()
    cpu_s = df_burst["cpu_usage_percent"]
    roll_mean_1h = cpu_s.rolling(window=12, min_periods=1).mean()
    roll_std_1h = cpu_s.rolling(window=12, min_periods=1).std().fillna(0)
    roll_std_6h = cpu_s.rolling(window=72, min_periods=1).std().fillna(0)

    axes[0].plot(df_burst["timestamp"], cpu_s, color="#aec7e8", lw=0.6, alpha=0.6, label="Raw Aligned CPU %")
    axes[0].plot(df_burst["timestamp"], roll_mean_1h, color="#1f77b4", lw=1.5, label="1-Hour Rolling Mean")
    axes[0].fill_between(df_burst["timestamp"],
                         np.clip(roll_mean_1h - roll_std_1h, 0, 100),
                         np.clip(roll_mean_1h + roll_std_1h, 0, 100),
                         color="#1f77b4", alpha=0.2, label="1-Hour Rolling Volatility (+/- 1 sigma)")
    axes[0].set_title("Figure 3A: VM_001 (Bursty) Workload Trajectory with 1-Hour Rolling Volatility Envelope",
                      fontweight="bold", fontsize=11)
    axes[0].set_ylabel("CPU Usage [%]")
    axes[0].legend(loc="upper right", frameon=True, fontsize=9)
    axes[0].grid(True, linestyle=":", alpha=0.5)

    axes[1].plot(df_burst["timestamp"], roll_std_1h, color="#d62728", lw=1.2, label="1-Hour Local Volatility (sigma_12)")
    axes[1].plot(df_burst["timestamp"], roll_std_6h, color="#ff7f0e", lw=1.5, label="6-Hour Macro Volatility (sigma_72)")
    axes[1].set_title("Figure 3B: Comparison of Short-Term (1h) vs Intermediate (6h) Workload Volatility",
                      fontweight="bold", fontsize=11)
    axes[1].set_ylabel("Rolling Std Dev [% CPU]")
    axes[1].set_xlabel("Date (UTC)")
    axes[1].legend(loc="upper right", frameon=True, fontsize=9)
    axes[1].grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    fig.savefig(fig_dir / "rolling_volatility.png", dpi=200)
    plt.close(fig)

    # Figure 4: Telemetry Cross-Metric Correlation Heatmap
    pool_frames = []
    telemetry_cols = [
        "cpu_usage_percent", "cpu_usage_mhz", "memory_usage_kb",
        "disk_read_kbps", "disk_write_kbps", "network_received_kbps", "network_transmitted_kbps"
    ]
    for df in aligned_dfs.values():
        if len(df) > 0:
            pool_frames.append(df[telemetry_cols])
    df_corr_pool = pd.concat(pool_frames, ignore_index=True)
    corr_mat = df_corr_pool.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(8.5, 7))
    cax = ax.matshow(corr_mat, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    clean_labels = [
        "CPU %", "CPU MHz", "RAM (KB)",
        "Disk Read", "Disk Write", "Net Rx", "Net Tx"
    ]
    ax.set_xticks(range(len(clean_labels)))
    ax.set_yticks(range(len(clean_labels)))
    ax.set_xticklabels(clean_labels, rotation=45, ha="left", fontsize=9)
    ax.set_yticklabels(clean_labels, fontsize=9)

    for i in range(len(clean_labels)):
        for j in range(len(clean_labels)):
            val = corr_mat.iloc[i, j]
            text_color = "white" if abs(val) > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontweight="bold", fontsize=9)

    ax.set_title("Figure 4: Empirical Pearson Cross-Correlation Heatmap Across Cohort Telemetry",
                 fontweight="bold", fontsize=11, pad=20)
    plt.tight_layout()
    fig.savefig(fig_dir / "correlation_matrix.png", dpi=200)
    plt.close(fig)

    # Figure 5: Objective Autocorrelation Function (ACF) up to Lag 288
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    acf_cohort = [
        ("1.csv", "VM_001 (Bursty Workload)", axes_flat[0], "#1f77b4"),
        ("3.csv", "VM_003 (Steady Moderate)", axes_flat[1], "#2ca02c"),
        ("11.csv", "VM_011 (Periodic Spikes)", axes_flat[2], "#d62728"),
        ("25.csv", "VM_025 (Heavy Sustained)", axes_flat[3], "#9467bd"),
    ]

    lags_hours = np.arange(289) * (5.0 / 60.0)

    for fname, title, ax, color in acf_cohort:
        if fname in aligned_dfs and len(aligned_dfs[fname]) >= 288:
            acf_vals = compute_autocorrelation(aligned_dfs[fname]["cpu_usage_percent"], max_lag=288)
            ax.plot(lags_hours[:len(acf_vals)], acf_vals, color=color, lw=1.6)
            ax.axhline(0, color="black", linestyle="--", lw=0.8, alpha=0.7)
            ax.axvline(1.0, color="#7f7f7f", linestyle=":", lw=0.8)
            ax.axvline(24.0, color="#d62728", linestyle=":", lw=1.0, alpha=0.8, label="24h Diurnal Boundary")
            ax.set_title(title, fontweight="bold", fontsize=10)
            ax.set_ylabel("Autocorrelation (ACF)")
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend(loc="upper right", frameon=True, fontsize=8)

    axes_flat[2].set_xlabel("Lag Horizon (Hours)")
    axes_flat[3].set_xlabel("Lag Horizon (Hours)")
    plt.suptitle("Figure 5: Empirical Autocorrelation Function (ACF) Curves up to Lag 288 (24 Hours)",
                 fontweight="bold", fontsize=12, y=0.995)
    plt.tight_layout()
    fig.savefig(fig_dir / "autocorrelation.png", dpi=200)
    plt.close(fig)

    # Figure 6: Visual Diagram of Chronological 70% / 15% / 15% Split
    fig, axes = plt.subplots(2, 1, figsize=(14, 6))

    split_examples = [
        ("1.csv", "VM_001 (Type A — Continuous 30-Day Operational Lifetime)", axes[0]),
        ("2.csv", "VM_002 (Type B — Decommissioned After 1.89 Days)", axes[1]),
    ]

    for fname, title, ax in split_examples:
        if fname in aligned_dfs:
            df_vm = aligned_dfs[fname]
            N = len(df_vm)
            n_train = int(0.70 * N)
            n_val = int(0.15 * N)
            n_test = N - n_train - n_val

            train_df = df_vm.iloc[:n_train]
            val_df = df_vm.iloc[n_train : n_train + n_val]
            test_df = df_vm.iloc[n_train + n_val :]

            ax.plot(train_df["timestamp"], train_df["cpu_usage_percent"], color="#1f77b4", lw=0.8, label=f"Train Split 70% (N={n_train})")
            ax.plot(val_df["timestamp"], val_df["cpu_usage_percent"], color="#2ca02c", lw=0.8, label=f"Val Split 15% (N={n_val})")
            ax.plot(test_df["timestamp"], test_df["cpu_usage_percent"], color="#d62728", lw=0.8, label=f"Test Split 15% (N={n_test})")

            if not val_df.empty:
                ax.axvline(val_df["timestamp"].iloc[0], color="black", linestyle="--", lw=1.2)
            if not test_df.empty:
                ax.axvline(test_df["timestamp"].iloc[0], color="black", linestyle="--", lw=1.2)

            ax.set_title(title, fontweight="bold", fontsize=10, loc="left")
            ax.set_ylabel("CPU %")
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend(loc="upper right", frameon=True, fontsize=8)
            ax.set_ylim(-2, 105)

    axes[-1].set_xlabel("Timeline (UTC)")
    plt.suptitle("Figure 6: Split-Safe Chronological Partitioning (70% Train / 15% Val / 15% Test)",
                 fontweight="bold", fontsize=12, y=0.995)
    plt.tight_layout()
    fig.savefig(fig_dir / "temporal_split.png", dpi=200)
    plt.close(fig)


def run_stage4_eda():
    """Main execution entry point for Stage 4 EDA."""
    data_dir = Path("dataset/fastStorage/2013-8")
    output_dir = Path("results/stage4")
    fig_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STAGE 4: EXPLORATORY DATA ANALYSIS & TEMPORAL WORKLOAD PROFILING")
    print("=" * 80)

    print(f"\n[1/5] Processing {len(REPRESENTATIVE_VMS)} representative VM traces...")
    aligned_dfs: Dict[str, pd.DataFrame] = {}
    eda_records: List[Dict[str, Any]] = []

    for vm_info in REPRESENTATIVE_VMS:
        fname = vm_info["file"]
        file_path = data_dir / fname
        if not file_path.is_file():
            print(f"  [WARNING] File {fname} not found in {data_dir}. Skipping.")
            continue

        df_raw = load_bitbrains_vm(file_path)
        rule_eval = evaluate_candidate_active_rule(df_raw, fname)
        stats_eval, df_c = evaluate_policy_c_active_aligned(df_raw, fname, rule_eval)

        aligned_dfs[fname] = df_c
        eda_stats = compute_vm_eda_statistics(df_c, vm_info)
        eda_records.append(eda_stats)
        print(f"  Processed {get_vm_id(fname)} ({fname}): {len(df_raw)} raw -> {rule_eval['active_rows']} active -> {len(df_c)} aligned steps ({eda_stats['span_days']} days)")

    eda_df = pd.DataFrame(eda_records)
    eda_summary_path = output_dir / "eda_summary.csv"
    eda_df.to_csv(eda_summary_path, index=False)
    print(f"\n[2/5] Exported EDA Summary to: {eda_summary_path}")

    print("\n[3/5] Recomputing empirical correlations across pooled cohort...")
    feature_df = analyze_features_and_correlations(aligned_dfs)
    feature_analysis_path = output_dir / "feature_analysis.csv"
    feature_df.to_csv(feature_analysis_path, index=False)
    print(f"  Exported Feature Analysis to: {feature_analysis_path}")

    print("\n[4/5] Computing separate within-VM correlation analysis for dynamic telemetry...")
    within_vm_df = compute_within_vm_correlations(aligned_dfs)
    within_vm_path = output_dir / "within_vm_correlations.csv"
    within_vm_df.to_csv(within_vm_path, index=False)
    print(f"  Exported Within-VM Correlations to: {within_vm_path}")
    print(within_vm_df.to_string(index=False))

    print("\n[5/5] Generating 6 publication-quality figures in results/stage4/figures/...")
    generate_stage4_plots(aligned_dfs, eda_df, fig_dir)
    print("  - Figure 1: cpu_distribution.png")
    print("  - Figure 2: cpu_workload_examples.png")
    print("  - Figure 3: rolling_volatility.png")
    print("  - Figure 4: correlation_matrix.png")
    print("  - Figure 5: autocorrelation.png")
    print("  - Figure 6: temporal_split.png")

    print("\n" + "=" * 80)
    print("STAGE 4 EDA EXECUTION COMPLETE SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    run_stage4_eda()
