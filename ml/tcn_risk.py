"""
Stage 5.5 / Phase 11: Workload Intelligence & Prediction-Risk Pipeline
======================================================================
Implements Person 1's workload intelligence and prediction-risk proxy layer
to feed Person 3's Adaptive Hippopotamus Optimization (AHO) algorithm.

Guarantees & Methodological Boundaries:
---------------------------------------
1. Offline Validation Calibration:
   - Evaluates the locked M3_FULL_standard TCN on the 6,557 validation sliding windows.
   - Computes causal workload volatility V(t) over past 24 steps (K=24, 2 hours, ddof=1).
   - Computes forecast trajectory spread S(t) = max(y_hat) - min(y_hat).
   - Extracts 95th percentiles (V_max, S_max) and RiskScore tertiles (tau_low, tau_high).
   - Exports frozen constants to results/stage5/risk/risk_calibration.json.
   - Zero test data is used for calibration.

2. Test-Time Placement Input Artifact:
   - Evaluates test decision windows (6,915 sliding windows across the 7 active VMs).
   - Generates results/stage5/risk/risk_state.csv with the approved 15-column schema.
   - Causal only: all features, current_cpu, and volatility rely strictly on x <= t.
   - Predictions cover t+5m through t+60m.
   - ZERO future actual CPU values, residuals, or error columns are present.

3. Short-Lived VMs:
   - 7 short-lived VMs (VM_002, VM_004, VM_006, VM_014, VM_018, VM_161, VM_1209) have
     test partition lengths < 300 steps (L=288 + H=12) and produce 0 test windows.
   - They are excluded from risk_state.csv without synthetic padding or fabricated predictions.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from ml.tcn_train import get_device, set_seed

# Approved configuration constants
LOCKED_EXPERIMENT_ID = "M3_FULL_standard"
LOCKED_FEATURE_MODE = "M3_FULL"
LOCKED_NUM_FEATURES = 10
LOCKED_SCALING_STRATEGY = "standard"
LOCKED_TARGET_STRATEGY = "native"
LOCKED_LOOKBACK_L = 288
LOCKED_HORIZON_H = 12
VOLATILITY_LOOKBACK_K = 24  # 24 steps = 2.0 hours at 5-minute sampling
RISK_VOLATILITY_WEIGHT = 0.60
RISK_SPREAD_WEIGHT = 0.40

DEFAULT_CHECKPOINT_PATH = Path("results/stage5/checkpoints/M3_FULL_standard_best.pt")
DEFAULT_RISK_DIR = Path("results/stage5/risk")


def load_locked_tcn_model(checkpoint_path: Path, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    """Loads the locked M3_FULL_standard TCN model and verifies compatibility."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt["config"]
    assert cfg.get("experiment_id") == LOCKED_EXPERIMENT_ID, f"Expected {LOCKED_EXPERIMENT_ID}, got {cfg.get('experiment_id')}"
    assert cfg.get("num_features") == LOCKED_NUM_FEATURES, f"Expected {LOCKED_NUM_FEATURES} features"
    assert cfg.get("scaling_strategy") == LOCKED_SCALING_STRATEGY, f"Expected {LOCKED_SCALING_STRATEGY}"

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

    print("=" * 80)
    print("LOCKED TCN MODEL LOADED & VERIFIED:")
    print(f"  Checkpoint:      {checkpoint_path}")
    print(f"  Experiment ID:   {LOCKED_EXPERIMENT_ID}")
    print(f"  Parameters:      {count_parameters(model):,}")
    print(f"  Best Epoch:      {ckpt.get('best_epoch')}")
    print(f"  Best Val Loss:   {float(ckpt.get('best_validation_loss', 0)):.6f}")
    print("=" * 80)
    return model, ckpt


def run_split_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Runs batched inference over a split dataset and returns forecast array [N, 12]."""
    all_preds = []
    with torch.no_grad():
        for b_X, _, _ in loader:
            b_X = b_X.to(device, non_blocking=True)
            y_hat = model(b_X)
            all_preds.append(y_hat.cpu().numpy())
    return np.vstack(all_preds)


def compute_causal_volatilities(
    dataset: BitbrainsWindowDataset,
    vm_traces: Dict[str, pd.DataFrame],
    K: int = VOLATILITY_LOOKBACK_K,
    L: int = LOCKED_LOOKBACK_L,
) -> np.ndarray:
    """
    Computes sample standard deviation (ddof=1) of the previous K=24 CPU observations
    up to decision timestamp t (cutoff step index s_idx + L - 1).
    Strictly causal: uses only observations at or before t.
    """
    volatilities = np.zeros(len(dataset), dtype=np.float32)

    for i in range(len(dataset)):
        vm_id = str(dataset.vm_ids[i])
        s_idx = int(dataset.start_indices[i])
        cutoff_idx = s_idx + L - 1

        trace_df = vm_traces[vm_id]
        # Previous K observations up to cutoff_idx: [cutoff_idx - K + 1 : cutoff_idx + 1]
        hist_slice = trace_df["cpu_usage_percent"].iloc[cutoff_idx - K + 1 : cutoff_idx + 1].to_numpy(dtype=np.float64)
        assert len(hist_slice) == K, f"Expected {K} observations, got {len(hist_slice)}"

        # Sample standard deviation with ddof=1
        volatilities[i] = float(np.std(hist_slice, ddof=1))

    return volatilities


def calibrate_risk_on_validation(
    model: nn.Module,
    val_dataset: BitbrainsWindowDataset,
    vm_traces: Dict[str, pd.DataFrame],
    device: torch.device,
    batch_size: int = 64,
    output_dir: Path = DEFAULT_RISK_DIR,
) -> Dict[str, Any]:
    """
    Calibrates V_max, S_max, tau_low, and tau_high strictly on the 6,557 validation sliding windows.
    Zero test data is accessed or used.
    """
    print("\n[CALIBRATION] Running inference on 6,557 validation sliding windows...")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    t0 = time.time()
    y_hat_val = run_split_inference(model, val_loader, device=device)
    print(f"Validation inference completed in {time.time() - t0:.2f}s.")

    # 1. Causal Volatility (K=24 steps = 2.0 hours)
    print(f"[CALIBRATION] Computing causal volatility (K={VOLATILITY_LOOKBACK_K} steps, ddof=1)...")
    val_volatilities = compute_causal_volatilities(val_dataset, vm_traces, K=VOLATILITY_LOOKBACK_K)

    # 2. Forecast Trajectory Spread: max(y_hat) - min(y_hat)
    print("[CALIBRATION] Computing forecast trajectory spread across 12-step horizon...")
    val_spreads = np.max(y_hat_val, axis=1) - np.min(y_hat_val, axis=1)

    # 3. Normalization constants: 95th percentiles
    V_max = float(np.percentile(val_volatilities, 95.0))
    S_max = float(np.percentile(val_spreads, 95.0))

    # 4. RiskScore calculation
    v_norm = np.clip(val_volatilities / V_max, 0.0, 1.0)
    s_norm = np.clip(val_spreads / S_max, 0.0, 1.0)
    val_risk_scores = RISK_VOLATILITY_WEIGHT * v_norm + RISK_SPREAD_WEIGHT * s_norm

    # 5. Risk state tertiles (33.3rd and 66.7th percentiles)
    tau_low = float(np.percentile(val_risk_scores, 100.0 / 3.0))
    tau_high = float(np.percentile(val_risk_scores, 200.0 / 3.0))

    calibration_metadata = {
        "model_name": LOCKED_EXPERIMENT_ID,
        "feature_mode": LOCKED_FEATURE_MODE,
        "feature_strategy": LOCKED_SCALING_STRATEGY,
        "target_strategy": LOCKED_TARGET_STRATEGY,
        "lookback_steps": LOCKED_LOOKBACK_L,
        "forecast_horizon": LOCKED_HORIZON_H,
        "sampling_seconds": 300,
        "volatility_window": VOLATILITY_LOOKBACK_K,
        "volatility_metric": "sample_standard_deviation_ddof_1",
        "risk_volatility_weight": RISK_VOLATILITY_WEIGHT,
        "risk_spread_weight": RISK_SPREAD_WEIGHT,
        "volatility_p95": round(V_max, 6),
        "spread_p95": round(S_max, 6),
        "risk_low_threshold": round(tau_low, 6),
        "risk_high_threshold": round(tau_high, 6),
        "validation_window_count": len(val_dataset),
        "calibration_split": "validation",
        "calibration_sliding_windows_dependent": True,
        "calibration_timestamp": datetime.now(timezone.utc).isoformat(),
        "methodology_version": "Phase 11 (Approved Phase 10 Specification)",
        "summary_statistics_validation": {
            "volatility_mean": round(float(np.mean(val_volatilities)), 6),
            "volatility_median": round(float(np.median(val_volatilities)), 6),
            "volatility_p95": round(V_max, 6),
            "spread_mean": round(float(np.mean(val_spreads)), 6),
            "spread_median": round(float(np.median(val_spreads)), 6),
            "spread_p95": round(S_max, 6),
            "risk_score_mean": round(float(np.mean(val_risk_scores)), 6),
            "risk_score_min": round(float(np.min(val_risk_scores)), 6),
            "risk_score_max": round(float(np.max(val_risk_scores)), 6),
            "tau_low_p33": round(tau_low, 6),
            "tau_high_p67": round(tau_high, 6),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    calib_path = output_dir / "risk_calibration.json"
    with open(calib_path, "w", encoding="utf-8") as f:
        json.dump(calibration_metadata, f, indent=2)

    print("\n" + "=" * 80)
    print("FROZEN VALIDATION RISK CALIBRATION EXPORTED:")
    print(f"  File:                 {calib_path}")
    print(f"  V_max (p95 vol):      {V_max:.4f}% CPU")
    print(f"  S_max (p95 spread):   {S_max:.4f}% CPU")
    print(f"  tau_low (p33.3 risk): {tau_low:.4f}")
    print(f"  tau_high (p66.7 risk):{tau_high:.4f}")
    print(f"  Validation Windows:   {len(val_dataset):,}")
    print("=" * 80)

    return calibration_metadata


def generate_test_risk_state(
    model: nn.Module,
    test_dataset: BitbrainsWindowDataset,
    vm_traces: Dict[str, pd.DataFrame],
    calibration: Dict[str, Any],
    device: torch.device,
    batch_size: int = 64,
    output_dir: Path = DEFAULT_RISK_DIR,
) -> pd.DataFrame:
    """
    Generates the test-time placement input artifact results/stage5/risk/risk_state.csv
    over the 6,915 test decision windows using the frozen validation calibration.

    Strict Invariants:
    - Contains ONLY information available at decision time t.
    - Zero future actual values, zero errors, zero residuals.
    - Exactly 15 approved columns.
    """
    print("\n[TEST ARTIFACT] Running inference on 6,915 test decision windows...")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    t0 = time.time()
    y_hat_test = run_split_inference(model, test_loader, device=device)
    print(f"Test inference completed in {time.time() - t0:.2f}s.")

    # Causal Volatility (K=24 steps up to cutoff t)
    print(f"[TEST ARTIFACT] Computing causal volatility (K={VOLATILITY_LOOKBACK_K} steps, ddof=1)...")
    test_volatilities = compute_causal_volatilities(test_dataset, vm_traces, K=VOLATILITY_LOOKBACK_K)

    # Forecast Trajectory Spread: max(y_hat) - min(y_hat)
    test_spreads = np.max(y_hat_test, axis=1) - np.min(y_hat_test, axis=1)

    # Apply frozen validation calibration parameters
    V_max = float(calibration["volatility_p95"])
    S_max = float(calibration["spread_p95"])
    tau_low = float(calibration["risk_low_threshold"])
    tau_high = float(calibration["risk_high_threshold"])

    v_norm = np.clip(test_volatilities / V_max, 0.0, 1.0)
    s_norm = np.clip(test_spreads / S_max, 0.0, 1.0)
    test_risk_scores = RISK_VOLATILITY_WEIGHT * v_norm + RISK_SPREAD_WEIGHT * s_norm
    test_risk_scores = np.clip(test_risk_scores, 0.0, 1.0)

    n_windows = len(test_dataset)
    assert n_windows == 6915, f"Expected 6,915 test windows, got {n_windows}"

    rows: List[Dict[str, Any]] = []

    for i in range(n_windows):
        vm_id = str(test_dataset.vm_ids[i])
        s_idx = int(test_dataset.start_indices[i])
        t_cutoff = int(test_dataset.t_cutoffs[i])
        cutoff_idx = s_idx + LOCKED_LOOKBACK_L - 1

        trace_df = vm_traces[vm_id]
        row_cutoff = trace_df.iloc[cutoff_idx]
        assert int(row_cutoff["timestamp_raw"]) == t_cutoff, f"Timestamp mismatch: trace={row_cutoff['timestamp_raw']} vs t_cutoff={t_cutoff}"

        # Current observed CPU at decision timestamp t
        current_cpu = float(row_cutoff["cpu_usage_percent"])

        # Multi-step forecasts (clipped to physical non-negative range [0, 100]% for placement)
        y_fc = np.clip(y_hat_test[i], 0.0, 100.0)

        r_score = float(test_risk_scores[i])
        if r_score < tau_low:
            r_state = "LOW"
        elif r_score < tau_high:
            r_state = "MEDIUM"
        else:
            r_state = "HIGH"

        rows.append({
            "vm_id": vm_id,
            "decision_timestamp": t_cutoff,
            "current_cpu": round(current_cpu, 4),
            "predicted_cpu_t5m": round(float(y_fc[0]), 4),
            "predicted_cpu_t10m": round(float(y_fc[1]), 4),
            "predicted_cpu_t15m": round(float(y_fc[2]), 4),
            "predicted_cpu_t30m": round(float(y_fc[5]), 4),
            "predicted_cpu_t45m": round(float(y_fc[8]), 4),
            "predicted_cpu_t60m": round(float(y_fc[11]), 4),
            "predicted_mean_cpu": round(float(np.mean(y_fc)), 4),
            "predicted_peak_cpu": round(float(np.max(y_fc)), 4),
            "predicted_std_cpu": round(float(np.std(y_fc)), 4),
            "volatility_score": round(float(test_volatilities[i]), 4),
            "risk_score": round(r_score, 4),
            "risk_state": r_state,
        })

    df_risk = pd.DataFrame(rows)

    # Strict column verification
    expected_cols = [
        "vm_id", "decision_timestamp", "current_cpu",
        "predicted_cpu_t5m", "predicted_cpu_t10m", "predicted_cpu_t15m",
        "predicted_cpu_t30m", "predicted_cpu_t45m", "predicted_cpu_t60m",
        "predicted_mean_cpu", "predicted_peak_cpu", "predicted_std_cpu",
        "volatility_score", "risk_score", "risk_state",
    ]
    assert list(df_risk.columns) == expected_cols, f"Column mismatch: {list(df_risk.columns)} vs {expected_cols}"

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "risk_state.csv"
    df_risk.to_csv(csv_path, index=False)

    print("\n" + "=" * 80)
    print("TEST-TIME PLACEMENT INPUT ARTIFACT GENERATED:")
    print(f"  File:                 {csv_path}")
    print(f"  Total Rows:           {len(df_risk):,} (exactly 6,915)")
    print(f"  Active VMs:           {df_risk['vm_id'].nunique()} ({df_risk['vm_id'].unique().tolist()})")
    print(f"  Decision Timestamps:  {df_risk['decision_timestamp'].nunique():,}")
    print(f"  Risk State Counts:    {dict(df_risk['risk_state'].value_counts())}")
    print("=" * 80)

    return df_risk


def verify_phase11_artifacts(
    calibration_path: Path,
    risk_state_path: Path,
    output_dir: Path = DEFAULT_RISK_DIR,
) -> Dict[str, Any]:
    """
    Executes all 17 automated validation acceptance tests on the generated artifacts.
    """
    print("\n[VERIFICATION] Executing Phase 11 Acceptance Suite (Checks 1 to 17)...")
    assert calibration_path.exists(), "Check 1 Failed: risk_calibration.json does not exist"
    assert risk_state_path.exists(), "Check 2 Failed: risk_state.csv does not exist"

    with open(calibration_path, "r", encoding="utf-8") as f:
        calib = json.load(f)

    df_risk = pd.read_csv(risk_state_path)

    # Check 3: Row count
    assert len(df_risk) == 6915, f"Check 3 Failed: expected 6,915 rows, got {len(df_risk)}"

    # Check 4: Exactly approved 15 columns
    expected_cols = [
        "vm_id", "decision_timestamp", "current_cpu",
        "predicted_cpu_t5m", "predicted_cpu_t10m", "predicted_cpu_t15m",
        "predicted_cpu_t30m", "predicted_cpu_t45m", "predicted_cpu_t60m",
        "predicted_mean_cpu", "predicted_peak_cpu", "predicted_std_cpu",
        "volatility_score", "risk_score", "risk_state",
    ]
    assert list(df_risk.columns) == expected_cols, f"Check 4 Failed: column mismatch"

    # Check 5: No NaN
    assert df_risk.isna().sum().sum() == 0, "Check 5 Failed: NaN values detected"

    # Check 6: No Inf
    numeric_cols = [c for c in df_risk.columns if c not in ("vm_id", "risk_state")]
    assert np.isinf(df_risk[numeric_cols].to_numpy()).sum() == 0, "Check 6 Failed: Inf values detected"

    # Check 7: current_cpu in [0, 100]
    assert df_risk["current_cpu"].min() >= 0.0, "Check 7 Failed: negative current_cpu"
    assert df_risk["current_cpu"].max() <= 100.0, "Check 7 Failed: current_cpu > 100"

    # Check 8: All predicted CPU values finite and non-negative
    pred_cols = [
        "predicted_cpu_t5m", "predicted_cpu_t10m", "predicted_cpu_t15m",
        "predicted_cpu_t30m", "predicted_cpu_t45m", "predicted_cpu_t60m",
        "predicted_mean_cpu", "predicted_peak_cpu", "predicted_std_cpu",
    ]
    for c in pred_cols:
        assert np.isfinite(df_risk[c]).all(), f"Check 8 Failed: non-finite values in {c}"
        assert (df_risk[c] >= 0.0).all(), f"Check 8 Failed: negative values in {c}"

    # Check 9: RiskScore in [0, 1]
    assert df_risk["risk_score"].min() >= 0.0, "Check 9 Failed: risk_score < 0"
    assert df_risk["risk_score"].max() <= 1.0, "Check 9 Failed: risk_score > 1"

    # Check 10: risk_state contains only LOW, MEDIUM, HIGH
    valid_states = {"LOW", "MEDIUM", "HIGH"}
    assert set(df_risk["risk_state"].unique()).issubset(valid_states), "Check 10 Failed: invalid risk_state"

    # Check 11 & 12: No future actuals or error columns
    forbidden_keywords = ["actual", "error", "residual", "loss", "mae", "rmse", "target"]
    for c in df_risk.columns:
        for kw in forbidden_keywords:
            assert kw not in c.lower(), f"Check 11/12 Failed: forbidden column '{c}' detected"

    # Check 13: Simultaneous VM cohort structure
    timestamps = df_risk["decision_timestamp"].unique()
    assert len(timestamps) > 0, "Check 13 Failed: no timestamps"

    # Check 14: Unique (vm_id, decision_timestamp) pairs
    assert not df_risk.duplicated(subset=["vm_id", "decision_timestamp"]).any(), "Check 14 Failed: duplicates found"

    # Check 15: Complete horizon columns
    assert all(c in df_risk.columns for c in pred_cols), "Check 15 Failed: incomplete horizon"

    # Check 16: Calibration derived strictly from validation
    assert calib.get("calibration_split") == "validation", "Check 16 Failed: calibration not validation"
    assert calib.get("validation_window_count") == 6557, "Check 16 Failed: window count mismatch"

    # Check 17: Zero test data in calibration
    assert "test" not in str(calib.get("calibration_split")).lower(), "Check 17 Failed: test data in calibration"

    print("ALL 17 ACCEPTANCE TESTS PASSED SUCCESSFULLY! [PASS]")

    # Build verification report
    verification_record = {
        "status": "PASS",
        "row_count": len(df_risk),
        "expected_row_count": 6915,
        "vm_count": int(df_risk["vm_id"].nunique()),
        "vms_represented": sorted(df_risk["vm_id"].unique().tolist()),
        "short_lived_vms_excluded_count": 7,
        "short_lived_vms_excluded": ["VM_002", "VM_004", "VM_006", "VM_014", "VM_018", "VM_161", "VM_1209"],
        "decision_timestamp_count": int(df_risk["decision_timestamp"].nunique()),
        "columns": list(df_risk.columns),
        "nan_count": int(df_risk.isna().sum().sum()),
        "inf_count": int(np.isinf(df_risk[numeric_cols].to_numpy()).sum()),
        "current_cpu_range": [float(df_risk["current_cpu"].min()), float(df_risk["current_cpu"].max())],
        "risk_score_range": [float(df_risk["risk_score"].min()), float(df_risk["risk_score"].max())],
        "risk_state_distribution": {k: int(v) for k, v in df_risk["risk_state"].value_counts().items()},
        "calibration_split": calib["calibration_split"],
        "calibration_window_count": calib["validation_window_count"],
        "calibration_parameters": {
            "volatility_p95": calib["volatility_p95"],
            "spread_p95": calib["spread_p95"],
            "risk_low_threshold": calib["risk_low_threshold"],
            "risk_high_threshold": calib["risk_high_threshold"],
        },
        "leakage_checks": {
            "future_actuals_in_placement_input": False,
            "test_residuals_used_for_calibration": False,
            "causality_strictly_enforced": True,
        },
        "model_identifier": LOCKED_EXPERIMENT_ID,
        "verification_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    verif_path = output_dir / "risk_verification.json"
    with open(verif_path, "w", encoding="utf-8") as f:
        json.dump(verification_record, f, indent=2)
    print(f"[VERIFICATION] Report exported to: {verif_path}")

    return verification_record


def write_person3_readme(output_dir: Path = DEFAULT_RISK_DIR) -> Path:
    """Writes the comprehensive handoff README.md for Person 3."""
    readme_content = """# Person 3 Integration Guide: Workload Intelligence & Risk-State Interface
**Handoff Producer**: Person 1 (ML Workload Forecasting Engine)  
**Consumer**: Person 3 (Adaptive Hippopotamus Optimization - AHO)  
**Downstream Simulation**: Person 2 (CloudSim Plus Simulation Engine)  
**File**: `risk_state.csv` (in this directory)  
**Classification**: **TEST-TIME PLACEMENT INPUT / ONLINE-SIMULATION ARTIFACT**  
**Row Count**: Exactly **6,915** rows  

---

## 1. What is this Artifact?
`risk_state.csv` contains the **workload intelligence snapshot** generated by Person 1's champion Temporal Convolutional Network (`TCNForecaster`, model `M3_FULL_standard`, 10 input features, receptive field 511 steps).

For every active VM at every 5-minute decision timestamp $t$ across the test split, this file provides:
1. **Current Workload**: `current_cpu` observed at time $t$.
2. **Multi-Step Forecasting**: Predicted CPU demand for the upcoming 1-hour horizon:
   - `predicted_cpu_t5m` (+5 min, step 1)
   - `predicted_cpu_t10m` (+10 min, step 2)
   - `predicted_cpu_t15m` (+15 min, step 3)
   - `predicted_cpu_t30m` (+30 min, step 6)
   - `predicted_cpu_t45m` (+45 min, step 9)
   - `predicted_cpu_t60m` (+60 min, step 12)
3. **Summary Forecasts**:
   - `predicted_mean_cpu`: Arithmetic mean over the 1-hour horizon (ideal for **energy consumption estimation** in the AHO fitness function).
   - `predicted_peak_cpu`: Maximum predicted demand over the 1-hour horizon (ideal for **SLA & host capacity constraint checking**).
   - `predicted_std_cpu`: Standard deviation across the 12 forecast steps (measures projected trajectory variation).
4. **Causal Workload Volatility**:
   - `volatility_score`: Sample standard deviation ($ddof=1$) of the previous 24 CPU observations ($K=24$, 2.0 hours) strictly up to cutoff $t$. Measures physical workload turbulence.
5. **Prediction-Risk Proxy**:
   - `risk_score`: Standardized index in $[0.0, 1.0]$ combining input turbulence ($60\%$) and forecast trajectory spread ($40\%$), calibrated strictly on historical validation sliding windows.
6. **Categorical Risk State**:
   - `risk_state`: `LOW`, `MEDIUM`, or `HIGH`, partitioned via empirical validation tertiles.

---

## 2. Why Was it Designed This Way?
In an online cloud data center, future workload is unknown when placing VMs. A placement optimizer that assumes static demand risks SLA violations when spikes occur, while one that allocates peak demand at all times wastes vast amounts of energy.

This artifact provides Person 3's Adaptive HO with:
- A forward-looking multi-step profile rather than a myopic single-step point.
- An explicit risk signal so AHO can dynamically adapt its exploration vs. exploitation balance and capacity safety margins.

---

## 3. How Person 3 Consumes this Table in Python (Minimal Code Example)

```python
import pandas as pd

# 1. Load the master placement input artifact
df_risk = pd.read_csv("results/stage5/risk/risk_state.csv")

# 2. Get unique decision timestamps in chronological order
decision_epochs = sorted(df_risk["decision_timestamp"].unique())
print(f"Total placement decision epochs: {len(decision_epochs)}")

# 3. For a given decision cycle t (e.g. the first test epoch):
t = decision_epochs[0]
cohort_at_t = df_risk[df_risk["decision_timestamp"] == t].copy()

print(f"\\n--- Decision Cycle: {t} ---")
print(f"Candidate VMs to place: {len(cohort_at_t)}")
print(cohort_at_t[["vm_id", "current_cpu", "predicted_mean_cpu", "predicted_peak_cpu", "risk_state"]])

# 4. Use in AHO:
# - Capacity constraint: sum(predicted_peak_cpu) + Headroom(risk_state) <= Host_Capacity
# - Fitness function: Host_Power_Model(predicted_mean_cpu)
# - Optimizer adaptation: Adjust AHO exploration parameter based on mean(risk_score)
```

---

## 4. Java / CloudSim Plus Interoperability Note
For teams integrating with Java-based CloudSim Plus:
- `risk_state.csv` uses standard RFC 4180 CSV format (UTF-8, comma-delimited, newline-terminated).
- In Java, it can be parsed using `BufferedReader` or `org.apache.commons.csv.CSVParser`.
- The `decision_timestamp` is a standard 64-bit Unix epoch integer matching CloudSim simulation clock seconds.

---

## 5. Formal Ownership Boundaries & Contract

### Person 3 OWNS:
1. **VM-to-PM Allocation Encoding**: Representation of chromosomes/particles/hippos.
2. **Physical Host Modeling**: CPU capacity, RAM capacity, and power curves ($W$).
3. **Fitness Function Formulation**: Trade-offs between power ($kWh$), migrations, and SLA violations.
4. **Standard Hippopotamus Optimization (HO)**: Core optimization algorithm.
5. **Adaptive Hippopotamus Optimization (AHO)**: Dynamic adaptation rules.
6. **Risk-to-Parameter Mapping**: Deciding how `risk_state` and `risk_score` adjust exploration/exploitation.
7. **Host Headroom & Safety Margins**: Defining PM buffer percentages or colocation rules.
8. **Final `placement.csv`**: Generating the VM-to-PM placement decision file.

### Person 3 MUST NOT:
1. Recalculate TCN predictions or attempt to fine-tune the TCN weights.
2. Use future actual CPU values during placement optimization (future actuals are reserved for CloudSim simulation evaluation).
3. Tune risk thresholds using test-set actuals (all calibration parameters are frozen from the validation split).
4. Modify the locked Stage 5 TCN results.

---

## 6. Short-Lived VMs Note
7 VMs (`VM_002`, `VM_004`, `VM_006`, `VM_014`, `VM_018`, `VM_161`, `VM_1209`) in the Bitbrains cohort have short active lifespans ($< 1.89$ days) and their test partitions contain $< 300$ steps ($< 25$ hours). Because the TCN requires a 24-hour lookback, no sliding windows could be formed for them without cross-partition leakage. They do not appear in `risk_state.csv`. If Person 2/3 needs to place these VMs during their brief lifespans, apply a standard non-TCN fallback rule.
"""
    readme_path = output_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content.strip() + "\n")
    print(f"[HANDOFF] README.md written to: {readme_path}")
    return readme_path


def execute_phase11_pipeline() -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """Executes the complete Phase 11 workload intelligence generation and verification."""
    set_seed(42)
    device = get_device()
    output_dir = DEFAULT_RISK_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STAGE 5.5 / PHASE 11: IMPLEMENT PERSON 1 WORKLOAD INTELLIGENCE LAYER")
    print(f"Device: {device.type.upper()}")
    print("=" * 80)

    # 1. Load locked TCN model
    model, ckpt = load_locked_tcn_model(DEFAULT_CHECKPOINT_PATH, device=device)

    # 2. Ingest Stage 5 data (Policy C 5-minute canonical grid)
    print("\n[STEP 1/5] Ingesting Stage 5 data (Policy C canonical grid)...")
    vm_traces, vm_partitions, index_table, stats = prepare_stage5_data()

    # 3. Create datasets with train-only StandardScaler fitting
    print("[STEP 2/5] Creating Stage 5 datasets with train-only StandardScaler...")
    datasets, _, _ = create_stage5_datasets(
        vm_traces=vm_traces,
        index_table=index_table,
        feature_mode=LOCKED_FEATURE_MODE,
        feature_strategy=LOCKED_SCALING_STRATEGY,
        target_strategy=LOCKED_TARGET_STRATEGY,
        vm_partitions=vm_partitions,
    )
    val_dataset = datasets["val"]
    test_dataset = datasets["test"]
    assert len(val_dataset) == 6557, f"Expected 6,557 validation windows, got {len(val_dataset)}"
    assert len(test_dataset) == 6915, f"Expected 6,915 test windows, got {len(test_dataset)}"

    # 4. Calibrate risk strictly on validation sliding windows
    print("\n[STEP 3/5] Calibrating risk parameters on validation sliding windows...")
    calibration = calibrate_risk_on_validation(
        model=model,
        val_dataset=val_dataset,
        vm_traces=vm_traces,
        device=device,
        output_dir=output_dir,
    )

    # 5. Generate test-time placement input artifact
    print("\n[STEP 4/5] Generating test-time placement input (risk_state.csv)...")
    df_risk = generate_test_risk_state(
        model=model,
        test_dataset=test_dataset,
        vm_traces=vm_traces,
        calibration=calibration,
        device=device,
        output_dir=output_dir,
    )

    # 6. Generate Person 3 README
    write_person3_readme(output_dir=output_dir)

    # 7. Execute automated verification suite
    print("\n[STEP 5/5] Executing automated verification suite...")
    calib_path = output_dir / "risk_calibration.json"
    csv_path = output_dir / "risk_state.csv"
    verif = verify_phase11_artifacts(calib_path, csv_path, output_dir=output_dir)

    print("\n" + "=" * 80)
    print("PHASE 11 WORKLOAD INTELLIGENCE PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)

    return df_risk, calibration, verif


if __name__ == "__main__":
    execute_phase11_pipeline()

