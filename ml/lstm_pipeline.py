"""
Gen_LSTM: End-to-End Execution Pipeline (14 Bitbrains VMs)
============================================================
Project: Predictive Energy-Efficient VM Placement in Cloud Using ML and Adaptive Hippopotamus Optimization
Scope: Complete Gen_LSTM Pipeline Orchestrator for Controlled TCN vs. LSTM Comparison

Pipeline Execution Stages:
1. Ingests the exact same 14 representative VMs as Gen 1 via Policy C.
2. Fits StandardScaler strictly on unique training observations (44,343 steps).
3. Instantiates and audits LSTMForecaster (exactly 53,516 parameters).
4. Executes AdamW training loop with ReduceLROnPlateau and EarlyStopping.
5. Evaluates best validation model on Test partition -> predictions.csv (82,980 rows).
6. Calibrates causal risk parameters on Validation partition -> risk_calibration.json.
7. Evaluates Test partition -> risk_state.csv (6,915 rows, 15 columns, zero future actuals).
8. Exports complete metrics and metadata under results/gen_lstm/.
"""

from __future__ import annotations

import argparse
import json
import os
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

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from ml.lstm_model import (
    DEFAULT_DROPOUT,
    DEFAULT_FORECAST_HORIZON,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_INPUT_LENGTH,
    DEFAULT_INPUT_SIZE,
    DEFAULT_NUM_LAYERS,
    EXPECTED_PARAM_COUNT,
    LSTMForecaster,
    count_trainable_parameters,
)
from ml.lstm_train import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EARLY_STOP_PATIENCE,
    DEFAULT_GRAD_CLIP,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_MIN_LR,
    DEFAULT_SCHEDULER_FACTOR,
    DEFAULT_SCHEDULER_PATIENCE,
    DEFAULT_SEED,
    DEFAULT_WEIGHT_DECAY,
    get_device,
    set_seed,
    train_lstm_experiment,
)
from ml.tcn_dataset import (
    BitbrainsWindowDataset,
    create_stage5_dataloaders,
    create_stage5_datasets,
    fit_scalers_on_unique_train,
    prepare_stage5_data,
)
from ml.tcn_evaluate import build_predictions_dataframe, compute_per_vm_metrics
from ml.tcn_risk import (
    RISK_SPREAD_WEIGHT,
    RISK_VOLATILITY_WEIGHT,
    VOLATILITY_LOOKBACK_K,
    compute_causal_volatilities,
)
from ml.tcn_train import compute_forecasting_metrics

DEFAULT_OUTPUT_DIR = Path("results/gen_lstm")
DEFAULT_DATA_DIR = Path("dataset/fastStorage/2013-8")


def run_lstm_test_inference(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Runs batched inference on test windows returning y_true and y_pred."""
    model.eval()
    all_y_true = []
    all_y_pred = []

    with torch.no_grad():
        for b_X, b_y, _ in test_loader:
            b_X = b_X.to(device, non_blocking=True)
            y_hat = model(b_X)

            all_y_true.append(b_y.numpy())
            all_y_pred.append(y_hat.cpu().numpy())

    return np.vstack(all_y_true), np.vstack(all_y_pred)


def calibrate_and_generate_lstm_risk(
    model: nn.Module,
    val_dataset: BitbrainsWindowDataset,
    test_dataset: BitbrainsWindowDataset,
    vm_traces: Dict[str, pd.DataFrame],
    device: torch.device,
    risk_dir: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Calibrates V_max, S_max, tau_low, tau_high strictly on Validation split,
    then generates test-time risk_state.csv.
    """
    risk_dir.mkdir(parents=True, exist_ok=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 1. Validation Inference
    model.eval()
    with torch.no_grad():
        val_preds_list = []
        for b_X, _, _ in val_loader:
            b_X = b_X.to(device, non_blocking=True)
            val_preds_list.append(model(b_X).cpu().numpy())
    y_hat_val = np.vstack(val_preds_list)

    val_volatilities = compute_causal_volatilities(val_dataset, vm_traces, K=VOLATILITY_LOOKBACK_K)
    val_spreads = np.max(y_hat_val, axis=1) - np.min(y_hat_val, axis=1)

    V_max = float(np.percentile(val_volatilities, 95.0))
    S_max = float(np.percentile(val_spreads, 95.0))

    v_norm_val = np.clip(val_volatilities / V_max, 0.0, 1.0)
    s_norm_val = np.clip(val_spreads / S_max, 0.0, 1.0)
    val_risk_scores = RISK_VOLATILITY_WEIGHT * v_norm_val + RISK_SPREAD_WEIGHT * s_norm_val

    tau_low = float(np.percentile(val_risk_scores, 100.0 / 3.0))
    tau_high = float(np.percentile(val_risk_scores, 200.0 / 3.0))

    calib_meta = {
        "model_name": "gen_lstm",
        "model_architecture": "LSTMForecaster (2 layers, hidden=64, 53,516 params)",
        "feature_mode": "M3_FULL",
        "feature_strategy": "standard",
        "target_strategy": "native",
        "lookback_steps": DEFAULT_INPUT_LENGTH,
        "forecast_horizon": DEFAULT_FORECAST_HORIZON,
        "sampling_seconds": 300,
        "volatility_window": VOLATILITY_LOOKBACK_K,
        "risk_volatility_weight": RISK_VOLATILITY_WEIGHT,
        "risk_spread_weight": RISK_SPREAD_WEIGHT,
        "volatility_p95": round(V_max, 6),
        "spread_p95": round(S_max, 6),
        "risk_low_threshold": round(tau_low, 6),
        "risk_high_threshold": round(tau_high, 6),
        "validation_window_count": len(val_dataset),
        "calibration_timestamp": datetime.now(timezone.utc).isoformat(),
        "summary_statistics_validation": {
            "volatility_mean": round(float(np.mean(val_volatilities)), 6),
            "volatility_p95": round(V_max, 6),
            "spread_mean": round(float(np.mean(val_spreads)), 6),
            "spread_p95": round(S_max, 6),
            "risk_score_mean": round(float(np.mean(val_risk_scores)), 6),
            "tau_low_p33": round(tau_low, 6),
            "tau_high_p67": round(tau_high, 6),
        },
    }

    calib_path = risk_dir / "risk_calibration.json"
    with open(calib_path, "w", encoding="utf-8") as f:
        json.dump(calib_meta, f, indent=2)

    # 2. Test Inference & Risk State Generation
    with torch.no_grad():
        test_preds_list = []
        for b_X, _, _ in test_loader:
            b_X = b_X.to(device, non_blocking=True)
            test_preds_list.append(model(b_X).cpu().numpy())
    y_hat_test = np.vstack(test_preds_list)

    test_volatilities = compute_causal_volatilities(test_dataset, vm_traces, K=VOLATILITY_LOOKBACK_K)
    test_spreads = np.max(y_hat_test, axis=1) - np.min(y_hat_test, axis=1)

    v_norm_test = np.clip(test_volatilities / V_max, 0.0, 1.0)
    s_norm_test = np.clip(test_spreads / S_max, 0.0, 1.0)
    test_risk_scores = np.clip(RISK_VOLATILITY_WEIGHT * v_norm_test + RISK_SPREAD_WEIGHT * s_norm_test, 0.0, 1.0)

    rows = []
    for i in range(len(test_dataset)):
        vm_id = str(test_dataset.vm_ids[i])
        s_idx = int(test_dataset.start_indices[i])
        t_cutoff = int(test_dataset.t_cutoffs[i])
        cutoff_idx = s_idx + DEFAULT_INPUT_LENGTH - 1

        trace_df = vm_traces[vm_id]
        row_cutoff = trace_df.iloc[cutoff_idx]
        current_cpu = float(row_cutoff["cpu_usage_percent"])

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
    risk_csv_path = risk_dir / "risk_state.csv"
    df_risk.to_csv(risk_csv_path, index=False)

    return calib_meta, df_risk


def run_gen_lstm_pipeline(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    epochs: int = DEFAULT_MAX_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    seed: int = DEFAULT_SEED,
    device_str: Optional[str] = None,
    smoke_test: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Orchestrates the complete Gen_LSTM experiment.
    """
    t_start_total = time.time()
    set_seed(seed)
    device = get_device(device_str)

    output_dir = Path(output_dir)
    ckpt_dir = output_dir / "checkpoints"
    train_dir = output_dir / "training"
    preds_dir = output_dir / "predictions"
    risk_dir = output_dir / "risk"

    for d in [ckpt_dir, train_dir, preds_dir, risk_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 80)
        print("GEN_LSTM EXPERIMENT ORCHESTRATION")
        print(f"Device:           {device}")
        print(f"Output Directory: {output_dir}")
        print(f"Smoke Test Mode:  {smoke_test}")
        print("=" * 80)

    # Step 1: Ingest 14 VMs and build windows
    if verbose:
        print("\n[STEP 1/6] Ingesting and aligning 14 Gen 1 representative VMs...")
    vm_traces, vm_partitions, index_table, stats = prepare_stage5_data(data_dir=data_dir)

    # Step 2: Fit scaler strictly on unique training observations
    if verbose:
        print("\n[STEP 2/6] Fitting StandardScaler on unique training observations...")
    feature_cols = [
        "cpu_usage_percent", "memory_usage_kb", "disk_read_kbps",
        "disk_write_kbps", "network_received_kbps", "network_transmitted_kbps",
        "hour_sin", "hour_cos", "day_sin", "day_cos"
    ]
    feature_scaler, target_scaler = fit_scalers_on_unique_train(
        vm_partitions=vm_partitions,
        feature_cols=feature_cols,
        feature_strategy="standard",
        target_strategy="native",
        expected_steps=44343,
    )

    datasets = create_stage5_datasets(
        vm_traces=vm_traces,
        index_table=index_table,
        feature_mode="M3_FULL",
        feature_strategy="standard",
        target_strategy="native",
        vm_partitions=vm_partitions,
    )
    loaders = create_stage5_dataloaders(datasets, batch_size=batch_size, num_workers=0)

    # Step 3: Instantiating and verifying LSTM model
    if verbose:
        print("\n[STEP 3/6] Instantiating and auditing LSTMForecaster architecture...")
    model = LSTMForecaster(
        in_features=DEFAULT_INPUT_SIZE,
        input_length=DEFAULT_INPUT_LENGTH,
        hidden_size=DEFAULT_HIDDEN_SIZE,
        num_layers=DEFAULT_NUM_LAYERS,
        forecast_horizon=DEFAULT_FORECAST_HORIZON,
        dropout=DEFAULT_DROPOUT,
    )
    model.to(device)
    param_count = count_trainable_parameters(model)
    assert param_count == EXPECTED_PARAM_COUNT, f"Param count mismatch: {param_count} != {EXPECTED_PARAM_COUNT}"

    if verbose:
        print(f"  Trainable Parameters: {param_count:,} (Expected: {EXPECTED_PARAM_COUNT:,}) [PASS]")

    # Step 4: Training execution
    if verbose:
        print("\n[STEP 4/6] Executing training loop...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=DEFAULT_SCHEDULER_FACTOR,
        patience=DEFAULT_SCHEDULER_PATIENCE,
        min_lr=DEFAULT_MIN_LR,
    )
    criterion = nn.MSELoss()

    config = {
        "experiment_id": "gen_lstm",
        "model_architecture": "LSTMForecaster",
        "parameters": param_count,
        "input_features": DEFAULT_INPUT_SIZE,
        "hidden_size": DEFAULT_HIDDEN_SIZE,
        "num_layers": DEFAULT_NUM_LAYERS,
        "dropout": DEFAULT_DROPOUT,
        "forecast_horizon": DEFAULT_FORECAST_HORIZON,
        "epochs": 1 if smoke_test else epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "gradient_clip": DEFAULT_GRAD_CLIP,
        "early_stop_patience": DEFAULT_EARLY_STOP_PATIENCE,
        "seed": seed,
    }

    model, df_hist, train_summary = train_lstm_experiment(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        config=config,
        target_scaler=target_scaler,
        checkpoint_dir=ckpt_dir,
        history_dir=train_dir,
        max_train_batches=10 if smoke_test else None,
        max_val_batches=5 if smoke_test else None,
        verbose=verbose,
    )

    # Step 5: Test evaluation & predictions.csv
    if verbose:
        print("\n[STEP 5/6] Evaluating on Test split (6,915 windows)...")
    y_true_test, y_pred_test = run_lstm_test_inference(model, loaders["test"], device=device)

    df_preds = build_predictions_dataframe(
        test_dataset=datasets["test"],
        vm_traces=vm_traces,
        y_true=y_true_test,
        y_pred=y_pred_test,
        L=DEFAULT_INPUT_LENGTH,
        H=DEFAULT_FORECAST_HORIZON,
    )
    preds_csv_path = preds_dir / "predictions.csv"
    df_preds.to_csv(preds_csv_path, index=False)

    test_metrics = compute_forecasting_metrics(y_true_test, y_pred_test, mape_threshold=1.0)
    df_per_vm = compute_per_vm_metrics(df_preds)
    df_per_vm.to_csv(preds_dir / "per_vm_test_metrics.csv", index=False)

    eval_summary = {
        "experiment_id": "gen_lstm",
        "checkpoint_path": train_summary["checkpoint_path"],
        "model_architecture": {
            "model_class": "LSTMForecaster",
            "parameters": param_count,
            "input_features": DEFAULT_INPUT_SIZE,
            "hidden_size": DEFAULT_HIDDEN_SIZE,
            "num_layers": DEFAULT_NUM_LAYERS,
            "dropout": DEFAULT_DROPOUT,
            "forecast_horizon": DEFAULT_FORECAST_HORIZON,
        },
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "test_window_count": len(datasets["test"]),
        "test_total_forecast_points": len(df_preds),
        "test_metrics_overall": test_metrics,
        "best_epoch": train_summary["best_epoch"],
        "best_validation_loss": train_summary["best_validation_loss"],
        "total_training_time_sec": train_summary["total_training_time_sec"],
    }
    with open(preds_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    # Step 6: Risk calibration & risk_state.csv
    if verbose:
        print("\n[STEP 6/6] Calibrating risk on validation and generating risk_state.csv...")
    calib_meta, df_risk = calibrate_and_generate_lstm_risk(
        model=model,
        val_dataset=datasets["val"],
        test_dataset=datasets["test"],
        vm_traces=vm_traces,
        device=device,
        risk_dir=risk_dir,
        batch_size=batch_size,
    )

    total_pipeline_time = time.time() - t_start_total

    exp_config_path = output_dir / "experiment_config.json"
    with open(exp_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    if verbose:
        print("\n" + "=" * 80)
        print("GEN_LSTM PIPELINE EXECUTION COMPLETE")
        print(f"Total Pipeline Time: {total_pipeline_time:.1f}s")
        print(f"Predictions File:    {preds_csv_path} ({len(df_preds):,} rows)")
        print(f"Risk State File:     {risk_dir / 'risk_state.csv'} ({len(df_risk):,} rows)")
        print(f"Test MAE:            {test_metrics['MAE']:.4f}% CPU")
        print(f"Test RMSE:           {test_metrics['RMSE']:.4f}% CPU")
        print(f"Test R2:             {test_metrics['R2']:.4f}")
        print("=" * 80 + "\n")

    return {
        "status": "SUCCESS",
        "model_parameters": param_count,
        "test_metrics": test_metrics,
        "total_pipeline_time_sec": round(total_pipeline_time, 2),
        "predictions_path": str(preds_csv_path),
        "risk_state_path": str(risk_dir / "risk_state.csv"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gen_LSTM Pipeline Orchestrator")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-epoch lightweight smoke test")
    args = parser.parse_args()

    run_gen_lstm_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device_str=args.device,
        smoke_test=args.smoke_test,
        verbose=True,
    )
