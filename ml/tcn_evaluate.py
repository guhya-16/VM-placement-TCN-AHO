"""
Stage 5.4 / Phase 9: Final TCN Test Evaluation & Prediction Artifact Generator
==============================================================================
Evaluates the locked M3_FULL_standard TCN forecasting model on the untouched test
split (6,915 sliding temporal windows across 14 VMs) and generates the complete
prediction artifact suite for downstream workload-risk assessment and Adaptive
Hippopotamus Optimization (AHO).

Key Guardrails & Methodological Invariants:
1. Locked Model Verification:
   - Experiment ID: M3_FULL_standard
   - Input channels: F=10 (M3_FULL features)
   - Lookback: L=288 (24.0 hours)
   - Horizon: H=12 (1.0 hour at 5-minute sampling)
   - Receptive field: RF=511 >= 288
   - Scaler: StandardScalerWrapper fitted exclusively on 44,343 unique train steps
   - Target strategy: native (unscaled CPU%)
   - Strict checkpoint verification: fails immediately if any attribute mismatches.

2. Test Split Isolation:
   - Evaluated strictly on the 6,915 test sliding windows.
   - Zero training or validation windows evaluated.
   - Zero test data used for fitting scalers.

3. Prediction Persistence (82,980 rows):
   - 6,915 test windows * 12 forecast steps = 82,980 distinct forecast instances.
   - Preserves window cutoff, step (1..12), forecast timestamp (t_cutoff + h*300),
     actual CPU, predicted CPU, error, abs error, squared error, and point-level
     is_interpolated mask (from the exact Stage 4/5 interpolation flag).
   - No deduplication of overlapping forecast instances.

4. Consistent Evaluation Population:
   - Per-VM metrics and per-horizon metrics are calculated from the exact 82,980
     forecast population.

5. Sanity Checks & Publication-Quality Figures:
   - Zero NaN/Inf in predictions and targets.
   - Generates actual vs predicted trajectories, error distribution, and horizon degradation plots.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure project root is in sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from ml.tcn_dataset import (
    DEFAULT_H,
    DEFAULT_L,
    M3_FEATURES,
    BitbrainsWindowDataset,
    create_stage5_datasets,
    prepare_stage5_data,
)
from ml.tcn_model import (
    DEFAULT_DILATIONS,
    DEFAULT_DROPOUT,
    DEFAULT_HIDDEN_CHANNELS,
    DEFAULT_KERNEL_SIZE,
    TCNForecaster,
    calculate_receptive_field,
    count_parameters,
)
from ml.tcn_train import (
    DEFAULT_MAPE_THRESHOLD,
    compute_forecasting_metrics,
    get_device,
    set_seed,
)

# Locked configuration constants
LOCKED_EXPERIMENT_ID = "M3_FULL_standard"
LOCKED_FEATURE_MODE = "M3_FULL"
LOCKED_NUM_FEATURES = 10
LOCKED_SCALING_STRATEGY = "standard"
LOCKED_TARGET_STRATEGY = "native"
LOCKED_LOOKBACK_L = 288
LOCKED_HORIZON_H = 12
DEFAULT_CHECKPOINT_PATH = Path("results/stage5/checkpoints/M3_FULL_standard_best.pt")
DEFAULT_OUTPUT_DIR = Path("results/stage5/predictions")


def verify_checkpoint_compatibility(
    checkpoint_path: Path,
    expected_exp_id: str = LOCKED_EXPERIMENT_ID,
    expected_num_features: int = LOCKED_NUM_FEATURES,
    expected_lookback: int = LOCKED_LOOKBACK_L,
    expected_horizon: int = LOCKED_HORIZON_H,
) -> Dict[str, Any]:
    """
    Strictly verifies that the checkpoint file matches all architectural,
    feature, and methodological specifications.
    Fails immediately (raises ValueError) if ANY attribute mismatches.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    required_keys = {"model_state_dict", "best_epoch", "best_validation_loss", "config"}
    missing_keys = required_keys - set(ckpt.keys())
    if missing_keys:
        raise ValueError(f"Checkpoint missing required keys: {missing_keys}")

    cfg = ckpt["config"]
    exp_id = cfg.get("experiment_id")
    if exp_id != expected_exp_id:
        raise ValueError(
            f"Checkpoint experiment_id mismatch: expected '{expected_exp_id}', got '{exp_id}'"
        )

    num_feat = cfg.get("num_features")
    if num_feat != expected_num_features:
        raise ValueError(
            f"Checkpoint num_features mismatch: expected {expected_num_features}, got {num_feat}"
        )

    feat_mode = cfg.get("feature_mode")
    if feat_mode != LOCKED_FEATURE_MODE:
        raise ValueError(
            f"Checkpoint feature_mode mismatch: expected '{LOCKED_FEATURE_MODE}', got '{feat_mode}'"
        )

    scale_strat = cfg.get("scaling_strategy")
    if scale_strat != LOCKED_SCALING_STRATEGY:
        raise ValueError(
            f"Checkpoint scaling_strategy mismatch: expected '{LOCKED_SCALING_STRATEGY}', got '{scale_strat}'"
        )

    tgt_strat = cfg.get("target_strategy", "native")
    if tgt_strat != LOCKED_TARGET_STRATEGY:
        raise ValueError(
            f"Checkpoint target_strategy mismatch: expected '{LOCKED_TARGET_STRATEGY}', got '{tgt_strat}'"
        )

    best_epoch = int(ckpt.get("best_epoch", 0))
    if best_epoch <= 0:
        raise ValueError(f"Invalid best_epoch in checkpoint: {best_epoch}")

    best_val_loss = float(ckpt.get("best_validation_loss", float("inf")))
    if not np.isfinite(best_val_loss):
        raise ValueError(f"Invalid best_validation_loss in checkpoint: {best_val_loss}")

    rf = calculate_receptive_field(DEFAULT_KERNEL_SIZE, DEFAULT_DILATIONS)
    if rf < expected_lookback:
        raise ValueError(
            f"Architecture receptive field {rf} is less than required lookback {expected_lookback}"
        )

    print("=" * 80)
    print("CHECKPOINT ARCHITECTURE COMPATIBILITY VERIFICATION: [PASS]")
    print(f"  Checkpoint:           {checkpoint_path}")
    print(f"  Experiment ID:        {exp_id}")
    print(f"  Feature Mode:         {feat_mode} ({num_feat} features)")
    print(f"  Scaling Strategy:     {scale_strat}")
    print(f"  Target Strategy:      {tgt_strat}")
    print(f"  Best Epoch:           {best_epoch}")
    print(f"  Best Val Loss:        {best_val_loss:.6f}")
    print(f"  Receptive Field:      {rf} steps (>= {expected_lookback})")
    print("=" * 80)

    return ckpt


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Instantiates the locked TCNForecaster and restores checkpoint weights."""
    ckpt = verify_checkpoint_compatibility(checkpoint_path)

    model = TCNForecaster(
        in_channels=LOCKED_NUM_FEATURES,
        input_length=LOCKED_LOOKBACK_L,
        forecast_horizon=LOCKED_HORIZON_H,
        hidden_channels=DEFAULT_HIDDEN_CHANNELS,
        kernel_size=DEFAULT_KERNEL_SIZE,
        dilations=DEFAULT_DILATIONS,
        dropout=DEFAULT_DROPOUT,
    )

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    total_params = count_parameters(model)
    print(f"Model successfully loaded. Trainable parameters: {total_params:,}")
    return model, ckpt


def run_test_inference(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs batched inference over all test windows.
    Returns:
        y_true: ndarray of shape [N_windows, 12]
        y_pred: ndarray of shape [N_windows, 12]
    """
    all_y_true = []
    all_y_pred = []

    t_start = time.time()
    with torch.no_grad():
        for b_X, b_y, _ in test_loader:
            b_X = b_X.to(device, non_blocking=True)
            y_hat = model(b_X)

            all_y_true.append(b_y.numpy())
            all_y_pred.append(y_hat.cpu().numpy())

    y_true = np.vstack(all_y_true)
    y_pred = np.vstack(all_y_pred)
    duration = time.time() - t_start

    print(f"Inference completed over {len(y_true):,} test windows in {duration:.2f}s "
          f"({len(y_true) / duration:.1f} windows/s).")
    return y_true, y_pred


def build_predictions_dataframe(
    test_dataset: BitbrainsWindowDataset,
    vm_traces: Dict[str, pd.DataFrame],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    L: int = LOCKED_LOOKBACK_L,
    H: int = LOCKED_HORIZON_H,
) -> pd.DataFrame:
    """
    Constructs the final prediction table for downstream use.
    Generates exactly len(test_dataset) * H rows (6,915 * 12 = 82,980 rows).
    Does NOT deduplicate overlapping forecast instances.
    Reuses the exact Stage 4/5 point-level is_interpolated mask from vm_traces.
    """
    n_windows = len(test_dataset)
    assert n_windows == len(y_true) == len(y_pred), (
        f"Window count mismatch: dataset={n_windows}, y_true={len(y_true)}, y_pred={len(y_pred)}"
    )

    rows: List[Dict[str, Any]] = []

    for w_idx in range(n_windows):
        vm_id = str(test_dataset.vm_ids[w_idx])
        start_idx = int(test_dataset.start_indices[w_idx])
        t_cutoff = int(test_dataset.t_cutoffs[w_idx])
        trace_df = vm_traces[vm_id]

        for step in range(1, H + 1):
            h_idx = step - 1
            trace_pos = start_idx + L + h_idx
            row_trace = trace_df.iloc[trace_pos]

            fc_timestamp = int(row_trace["timestamp_raw"])
            # Exact 300-second step invariant check
            expected_fc_ts = t_cutoff + step * 300
            assert fc_timestamp == expected_fc_ts, (
                f"Timestamp mismatch for {vm_id} at window {w_idx} step {step}: "
                f"trace={fc_timestamp} vs expected={expected_fc_ts}"
            )

            actual_cpu = float(y_true[w_idx, h_idx])
            pred_cpu = float(y_pred[w_idx, h_idx])
            err = pred_cpu - actual_cpu
            abs_err = abs(err)
            sq_err = err ** 2

            # Exact Stage 4/5 interpolation mask
            is_interp = int(row_trace["is_interpolated"]) if "is_interpolated" in row_trace else 0

            rows.append({
                "window_idx": w_idx,
                "vm_id": vm_id,
                "cutoff_timestamp": t_cutoff,
                "forecast_step": step,
                "forecast_timestamp": fc_timestamp,
                "actual_cpu": round(actual_cpu, 4),
                "predicted_cpu": round(pred_cpu, 4),
                "error": round(err, 4),
                "absolute_error": round(abs_err, 4),
                "squared_error": round(sq_err, 4),
                "is_interpolated": is_interp,
            })

    df_preds = pd.DataFrame(rows)
    expected_rows = n_windows * H
    assert len(df_preds) == expected_rows, (
        f"Predictions row count mismatch: expected {expected_rows}, got {len(df_preds)}"
    )
    return df_preds


def compute_per_vm_metrics(df_preds: pd.DataFrame) -> pd.DataFrame:
    """
    Computes test evaluation metrics per VM across the full 82,980 forecast population.
    No deduplication.
    """
    records = []
    for vm_id, group in df_preds.groupby("vm_id", sort=True):
        y_t = group["actual_cpu"].to_numpy()
        y_p = group["predicted_cpu"].to_numpy()
        n_fc = len(group)
        n_win = group["window_idx"].nunique()

        m = compute_forecasting_metrics(y_t, y_p)
        records.append({
            "vm_id": vm_id,
            "number_of_windows": n_win,
            "number_of_forecasts": n_fc,
            "MAE": m["MAE"],
            "RMSE": m["RMSE"],
            "R2": m["R2"],
            "sMAPE": m["sMAPE"],
            "thresholded_MAPE": m["thresholded_MAPE"],
            "eligible_fraction_mape": m["eligible_fraction"],
        })

    return pd.DataFrame(records)


def compute_horizon_metrics(df_preds: pd.DataFrame) -> pd.DataFrame:
    """
    Computes test evaluation metrics per forecast step (h=1..12).
    Step 1 = +5 min, Step 12 = +60 min.
    """
    records = []
    for step in range(1, LOCKED_HORIZON_H + 1):
        group = df_preds[df_preds["forecast_step"] == step]
        y_t = group["actual_cpu"].to_numpy()
        y_p = group["predicted_cpu"].to_numpy()

        m = compute_forecasting_metrics(y_t, y_p)
        records.append({
            "forecast_step": step,
            "forecast_minutes": step * 5,
            "number_of_forecasts": len(group),
            "MAE": m["MAE"],
            "RMSE": m["RMSE"],
            "R2": m["R2"],
            "sMAPE": m["sMAPE"],
            "thresholded_MAPE": m["thresholded_MAPE"],
            "eligible_fraction_mape": m["eligible_fraction"],
        })

    return pd.DataFrame(records)


def generate_evaluation_figures(
    df_preds: pd.DataFrame,
    per_vm_df: pd.DataFrame,
    horizon_df: pd.DataFrame,
    figures_dir: Path,
) -> List[Path]:
    """Generates publication-quality figures under results/stage5/predictions/figures/."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated_plots = []

    # 1. Representative VM Forecasts: Actual vs Predicted
    # Pick 3 diverse representative VMs: high activity, moderate activity, cyclic activity
    rep_vms = ["VM_001", "VM_014", "VM_024"]
    # Filter to those available
    avail_vms = [v for v in rep_vms if v in df_preds["vm_id"].unique()]
    if not avail_vms:
        avail_vms = list(df_preds["vm_id"].unique()[:3])

    fig, axes = plt.subplots(len(avail_vms), 1, figsize=(14, 3.5 * len(avail_vms)), sharex=False)
    if len(avail_vms) == 1:
        axes = [axes]

    for ax, vid in zip(axes, avail_vms):
        vm_data = df_preds[df_preds["vm_id"] == vid]
        # To show a continuous timeline, select forecast_step == 1 (+5 min lead)
        step1 = vm_data[vm_data["forecast_step"] == 1].sort_values("forecast_timestamp").iloc[:288]
        # Also select step 12 (+60 min lead) for comparison
        step12 = vm_data[vm_data["forecast_step"] == 12].sort_values("forecast_timestamp").iloc[:288]

        x_time = np.arange(len(step1)) * 5 / 60  # Hours into test partition
        ax.plot(x_time, step1["actual_cpu"], label="Actual CPU %", color="#1f77b4", linewidth=1.5, alpha=0.9)
        ax.plot(x_time, step1["predicted_cpu"], label="TCN Forecast (+5m, Step 1)", color="#2ca02c", linestyle="--", linewidth=1.3)
        if len(step12) == len(step1):
            ax.plot(x_time, step12["predicted_cpu"], label="TCN Forecast (+60m, Step 12)", color="#ff7f0e", linestyle=":", linewidth=1.3)

        ax.set_title(f"{vid} Test Split Workload: Actual vs TCN Multi-Step Forecast (First 24 Hours)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Time Elapsed in Test Split (Hours)", fontsize=10)
        ax.set_ylabel("CPU Utilization (%)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    p1 = figures_dir / "actual_vs_predicted_representative_vms.png"
    plt.savefig(p1, dpi=300)
    plt.close()
    generated_plots.append(p1)

    # 2. Test Prediction Error Distribution
    errors = df_preds["error"].to_numpy()
    mean_err = float(np.mean(errors))
    std_err = float(np.std(errors))
    q01, q99 = np.percentile(errors, [1, 99])

    fig, ax = plt.subplots(figsize=(10, 6))
    n_bins = 100
    clipped_errors = np.clip(errors, q01, q99)
    ax.hist(clipped_errors, bins=n_bins, density=True, color="#3498db", edgecolor="#2980b9", alpha=0.7, label=f"Test Errors (N={len(errors):,})")
    ax.axvline(0, color="black", linestyle="--", linewidth=1.5, label="Zero Error (Unbiased)")
    ax.axvline(mean_err, color="red", linestyle="-", linewidth=1.5, label=f"Mean Error = {mean_err:+.3f}%")

    # Add Gaussian reference curve
    x_norm = np.linspace(q01, q99, 500)
    p_norm = (1 / (std_err * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_norm - mean_err) / std_err) ** 2)
    ax.plot(x_norm, p_norm, color="#e74c3c", linewidth=2.0, linestyle="-.", label=f"Gaussian Fit (σ={std_err:.3f})")

    ax.set_title("Test Prediction Error Distribution (M3_FULL_standard on 82,980 Test Points)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Prediction Error: (Predicted - Actual) % CPU", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    p2 = figures_dir / "error_distribution.png"
    plt.savefig(p2, dpi=300)
    plt.close()
    generated_plots.append(p2)

    # 3. Forecast Accuracy Degradation vs Horizon (h = 1..12)
    fig, ax1 = plt.subplots(figsize=(10, 5.5))

    color_mae = "#2b5c8f"
    color_rmse = "#d95f02"
    color_smape = "#7570b3"

    lead_mins = horizon_df["forecast_minutes"]
    ax1.plot(lead_mins, horizon_df["MAE"], marker="o", color=color_mae, linewidth=2.0, label="MAE (% CPU)")
    ax1.plot(lead_mins, horizon_df["RMSE"], marker="s", color=color_rmse, linewidth=2.0, label="RMSE (% CPU)")
    ax1.set_xlabel("Forecast Horizon Lead Time (Minutes)", fontsize=11)
    ax1.set_ylabel("Error Magnitude (% CPU)", fontsize=11)
    ax1.set_xticks(lead_mins)
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(lead_mins, horizon_df["sMAPE"], marker="^", color=color_smape, linewidth=2.0, linestyle="--", label="sMAPE (%)")
    ax2.set_ylabel("sMAPE (%)", color=color_smape, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color_smape)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.9)

    ax1.set_title("Forecast Accuracy Degradation Across Horizon (1 to 12 Steps: +5m to +60m)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    p3 = figures_dir / "metrics_vs_horizon.png"
    plt.savefig(p3, dpi=300)
    plt.close()
    generated_plots.append(p3)

    return generated_plots


def execute_phase9_evaluation(
    data_dir: Path = Path("dataset/fastStorage/2013-8"),
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    batch_size: int = 64,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Main execution pipeline for Phase 9 test evaluation and prediction artifact generation.
    """
    set_seed(seed)
    device = get_device()
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STAGE 5.4 / PHASE 9: FINAL TCN TEST EVALUATION & PREDICTION ARTIFACTS")
    print(f"Locked Model:        {LOCKED_EXPERIMENT_ID}")
    print(f"Checkpoint:          {checkpoint_path}")
    print(f"Device:              {device.type.upper()}")
    print("=" * 80)

    # 1. Load and strictly verify model checkpoint
    model, ckpt_meta = load_model_from_checkpoint(checkpoint_path, device=device)

    # 2. Ingest and align Stage 5 data (Policy C 5-minute canonical grid)
    print("\n[STEP 1/6] Ingesting Stage 5 data and extracting test split...")
    vm_traces, vm_partitions, index_table, stats = prepare_stage5_data(data_dir=data_dir)

    expected_test_windows = 6915
    actual_test_windows = stats["n_test_windows"]
    if actual_test_windows != expected_test_windows:
        raise ValueError(
            f"Test window count mismatch: expected {expected_test_windows}, got {actual_test_windows}"
        )

    # 3. Create datasets with train-only StandardScaler fitting
    print("\n[STEP 2/6] Fitting StandardScaler strictly on unique train observations (44,343 steps)...")
    datasets, feature_scaler, target_scaler = create_stage5_datasets(
        vm_traces=vm_traces,
        index_table=index_table,
        feature_mode=LOCKED_FEATURE_MODE,
        feature_strategy=LOCKED_SCALING_STRATEGY,
        target_strategy=LOCKED_TARGET_STRATEGY,
        vm_partitions=vm_partitions,
    )
    test_dataset = datasets["test"]
    assert len(test_dataset) == expected_test_windows, (
        f"Test dataset size mismatch: expected {expected_test_windows}, got {len(test_dataset)}"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Strict chronological ordering
        num_workers=0,
        pin_memory=False,
    )

    # 4. Run test inference
    print(f"\n[STEP 3/6] Running inference over {len(test_dataset):,} test windows...")
    y_true, y_pred = run_test_inference(model, test_loader, device=device)

    # Verify shapes and absence of NaNs/Infs
    assert y_true.shape == (expected_test_windows, LOCKED_HORIZON_H)
    assert y_pred.shape == (expected_test_windows, LOCKED_HORIZON_H)
    nan_pred_count = int(np.isnan(y_pred).sum())
    inf_pred_count = int(np.isinf(y_pred).sum())
    nan_true_count = int(np.isnan(y_true).sum())
    inf_true_count = int(np.isinf(y_true).sum())
    assert nan_pred_count == 0 and inf_pred_count == 0, "NaN or Inf found in predictions!"
    assert nan_true_count == 0 and inf_true_count == 0, "NaN or Inf found in actual targets!"

    # 5. Build predictions.csv (82,980 rows)
    print("\n[STEP 4/6] Constructing predictions DataFrame (82,980 forecast instances)...")
    df_preds = build_predictions_dataframe(
        test_dataset=test_dataset,
        vm_traces=vm_traces,
        y_true=y_true,
        y_pred=y_pred,
    )
    preds_csv_path = output_dir / "predictions.csv"
    df_preds.to_csv(preds_csv_path, index=False)
    print(f"  [SAVED] {preds_csv_path} ({preds_csv_path.stat().st_size / (1024 * 1024):.2f} MB, {len(df_preds):,} rows)")

    # 6. Compute metrics
    print("\n[STEP 5/6] Computing overall, per-VM, and horizon metrics...")
    overall_metrics = compute_forecasting_metrics(y_true, y_pred, mape_threshold=DEFAULT_MAPE_THRESHOLD)

    per_vm_df = compute_per_vm_metrics(df_preds)
    per_vm_csv_path = output_dir / "per_vm_test_metrics.csv"
    per_vm_df.to_csv(per_vm_csv_path, index=False)
    print(f"  [SAVED] {per_vm_csv_path} ({len(per_vm_df)} VMs)")

    horizon_df = compute_horizon_metrics(df_preds)
    horizon_csv_path = output_dir / "horizon_metrics.csv"
    horizon_df.to_csv(horizon_csv_path, index=False)
    print(f"  [SAVED] {horizon_csv_path} ({len(horizon_df)} steps)")

    # 7. Generate diagnostic figures
    print("\n[STEP 6/6] Generating diagnostic figures...")
    plots = generate_evaluation_figures(df_preds, per_vm_df, horizon_df, figures_dir)
    for p in plots:
        print(f"  [PLOT] {p}")

    # Build evaluation summary JSON
    summary_record = {
        "experiment_id": LOCKED_EXPERIMENT_ID,
        "checkpoint_path": checkpoint_path.as_posix(),
        "model_architecture": {
            "model_class": "TCNForecaster",
            "parameters": count_parameters(model),
            "input_channels": LOCKED_NUM_FEATURES,
            "forecast_horizon": LOCKED_HORIZON_H,
            "hidden_channels": DEFAULT_HIDDEN_CHANNELS,
            "kernel_size": DEFAULT_KERNEL_SIZE,
            "dilations": DEFAULT_DILATIONS,
            "receptive_field": calculate_receptive_field(DEFAULT_KERNEL_SIZE, DEFAULT_DILATIONS),
            "dropout": DEFAULT_DROPOUT,
        },
        "feature_mode": LOCKED_FEATURE_MODE,
        "feature_columns": M3_FEATURES,
        "scaler": LOCKED_SCALING_STRATEGY,
        "target_strategy": LOCKED_TARGET_STRATEGY,
        "lookback_L": LOCKED_LOOKBACK_L,
        "horizon_H": LOCKED_HORIZON_H,
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "test_window_count": expected_test_windows,
        "test_total_forecast_points": len(df_preds),
        "number_of_vms": len(per_vm_df),
        "nan_inf_counts": {
            "nan_predictions": nan_pred_count,
            "inf_predictions": inf_pred_count,
            "nan_targets": nan_true_count,
            "inf_targets": inf_true_count,
        },
        "distribution_stats": {
            "predicted_min": round(float(df_preds["predicted_cpu"].min()), 4),
            "predicted_max": round(float(df_preds["predicted_cpu"].max()), 4),
            "predicted_mean": round(float(df_preds["predicted_cpu"].mean()), 4),
            "predicted_std": round(float(df_preds["predicted_cpu"].std()), 4),
            "actual_min": round(float(df_preds["actual_cpu"].min()), 4),
            "actual_max": round(float(df_preds["actual_cpu"].max()), 4),
            "actual_mean": round(float(df_preds["actual_cpu"].mean()), 4),
            "actual_std": round(float(df_preds["actual_cpu"].std()), 4),
        },
        "test_metrics_overall": overall_metrics,
        "validation_metrics_reference": ckpt_meta.get("validation_metrics", {}),
        "validation_loss_reference": float(ckpt_meta.get("best_validation_loss", float("nan"))),
        "best_epoch_reference": int(ckpt_meta.get("best_epoch", 0)),
    }

    summary_json_path = output_dir / "evaluation_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_record, f, indent=2)
    print(f"  [SAVED] {summary_json_path}")

    # Display final summary table
    print("\n" + "=" * 80)
    print("FINAL TEST EVALUATION METRICS (OVERALL UNBIASED TEST BENCHMARK)")
    print("=" * 80)
    print(f"  Test Windows Evaluated:     {expected_test_windows:,}")
    print(f"  Total Forecast Points:      {len(df_preds):,} (6,915 windows x 12 steps)")
    print(f"  Test MAE:                   {overall_metrics['MAE']:.4f}% CPU")
    print(f"  Test RMSE:                  {overall_metrics['RMSE']:.4f}% CPU")
    print(f"  Test R2:                    {overall_metrics['R2']:.4f}")
    print(f"  Test sMAPE:                 {overall_metrics['sMAPE']:.4f}%")
    print(f"  Test Thresholded MAPE:      {overall_metrics['thresholded_MAPE']:.4f}% "
          f"({overall_metrics['eligible_fraction'] * 100:.1f}% eligible targets >= 1.0% CPU)")
    print(f"  Actual CPU Mean:            {summary_record['distribution_stats']['actual_mean']:.2f}% "
          f"(Min: {summary_record['distribution_stats']['actual_min']:.2f}%, Max: {summary_record['distribution_stats']['actual_max']:.2f}%)")
    print(f"  Predicted CPU Mean:         {summary_record['distribution_stats']['predicted_mean']:.2f}% "
          f"(Min: {summary_record['distribution_stats']['predicted_min']:.2f}%, Max: {summary_record['distribution_stats']['predicted_max']:.2f}%)")
    print("=" * 80)

    return summary_record


if __name__ == "__main__":
    execute_phase9_evaluation()
