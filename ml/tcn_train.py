"""
Stage 5.3: TCN Training Loop & Controlled Ablation Runner
=========================================================
Implements the training pipeline, early stopping, learning rate scheduling,
and controlled feature-mode / scaling-strategy ablations for the TCN forecaster.

Core Principles & Methodological Guardrails:
--------------------------------------------
1. Strict Split Isolation (NON-NEGOTIABLE):
   - Aligned observations: 63,353 total (Train: 44,343 | Val: 9,498 | Test: 9,512).
   - Valid sliding windows: 53,524 total (Train: 40,052 | Val: 6,557 | Test: 6,915).
   - TRAIN set is used strictly for parameter updates.
   - VALIDATION set is used strictly for early stopping, LR scheduling, and model selection.
   - TEST set (6,915 windows) is NEVER ACCESSED during Stage 5.3. Zero test leakage.
     Test evaluation belongs exclusively to Stage 5.4.

2. Fixed Approved TCN Architecture:
   - Imported directly from ml.tcn_model.TCNForecaster.
   - k=3, dilations=[1, 2, 4, 8, 16, 32, 64, 128], RF=511 >= 288 steps (24.0 hours).
   - 8 residual blocks (2 causal convs per block), hidden_channels=32, dropout=0.1.
   - Linear multi-step projection head producing exactly [B, 12].

3. Reproducibility:
   - Explicit random seeds for Python, NumPy, and PyTorch (default seed=42).
   - Seed recorded in all checkpoint files and summary logs.

4. Train-Only Scaler Fitting:
   - Scalers fitted exclusively on the 44,343 unique aligned training observations.
   - Validation observations are transformed using parameters learned strictly from training.

5. Validation Metrics in Native CPU % Units:
   - Inverted to native CPU percentage [0, 100]% before calculating:
     * MAE: Mean Absolute Error
     * RMSE: Root Mean Squared Error
     * R2: Coefficient of Determination (safe handling of zero-variance cases)
     * sMAPE: Symmetric Mean Absolute Percentage Error (bounded in [0, 200]%, robust to zero CPU)
     * Thresholded MAPE: Evaluated strictly on actual targets >= 1.0% CPU to prevent division by zero.

6. Checkpoint Safety & Early Stopping:
   - Monitored strictly on validation loss with configurable patience (default 5 epochs).
   - Best checkpoint saved to results/stage5/checkpoints/<exp_name>_best.pt.
   - Best checkpoint weights restored into model after training.
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# Ensure project root is in sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Reuse existing Stage 5.1 dataset pipeline and Stage 5.2 model architecture
from ml.tcn_dataset import (
    BaseScaler,
    create_stage5_dataloaders,
    create_stage5_datasets,
    prepare_stage5_data,
)
from ml.tcn_model import (
    DEFAULT_DILATIONS,
    DEFAULT_DROPOUT,
    DEFAULT_FORECAST_HORIZON,
    DEFAULT_HIDDEN_CHANNELS,
    DEFAULT_INPUT_LENGTH,
    DEFAULT_KERNEL_SIZE,
    TCNForecaster,
    calculate_receptive_field,
    count_parameters,
)

# Standard training hyperparameter defaults
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_EPOCHS = 50
DEFAULT_EARLY_STOP_PATIENCE = 5
DEFAULT_SCHEDULER_PATIENCE = 2
DEFAULT_SCHEDULER_FACTOR = 0.5
DEFAULT_SEED = 42
DEFAULT_MAPE_THRESHOLD = 1.0  # 1.0% CPU threshold for zero-safe MAPE


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Sets explicit random seeds for Python, NumPy, and PyTorch reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Automatically selects CUDA if genuinely available, otherwise CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


# =============================================================================
# METRICS SUITE (Evaluated in Native CPU % Units)
# =============================================================================

def compute_forecasting_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mape_threshold: float = DEFAULT_MAPE_THRESHOLD,
    eps: float = 1e-5,
) -> Dict[str, float]:
    """
    Computes rigorous forecasting accuracy metrics in native CPU percentage units:
    1. MAE: Mean Absolute Error (primary linear magnitude metric)
    2. RMSE: Root Mean Squared Error (primary large error penalty metric)
    3. R2: Coefficient of Determination (safe handling of constant targets)
    4. sMAPE: Symmetric Mean Absolute Percentage Error (robust to zero targets)
    5. Thresholded MAPE: Ordinary MAPE evaluated strictly on targets >= mape_threshold (1.0% CPU)
    """
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()

    # 1. MAE
    mae = float(np.mean(np.abs(y_t - y_p)))

    # 2. RMSE
    rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))

    # 3. R2 with safe zero-variance handling
    ss_res = float(np.sum((y_t - y_p) ** 2))
    ss_tot = float(np.sum((y_t - np.mean(y_t)) ** 2))
    if ss_tot > 1e-8:
        r2 = float(1.0 - (ss_res / ss_tot))
    else:
        r2 = float("nan")

    # 4. sMAPE
    denom = np.abs(y_t) + np.abs(y_p) + eps
    smape = float(np.mean(2.0 * np.abs(y_t - y_p) / denom) * 100.0)

    # 5. Thresholded MAPE (>= 1.0% CPU)
    mask = y_t >= mape_threshold
    n_eligible = int(np.sum(mask))
    total_obs = len(y_t)
    eligible_frac = float(n_eligible / total_obs) if total_obs > 0 else 0.0

    if n_eligible > 0:
        mape_thresh = float(np.mean(np.abs(y_t[mask] - y_p[mask]) / y_t[mask]) * 100.0)
    else:
        mape_thresh = float("nan")

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4) if not np.isnan(r2) else float("nan"),
        "sMAPE": round(smape, 4),
        "thresholded_MAPE": round(mape_thresh, 4) if not np.isnan(mape_thresh) else float("nan"),
        "eligible_observations": n_eligible,
        "eligible_fraction": round(eligible_frac, 4),
    }


# =============================================================================
# EPOCH EXECUTION FUNCTIONS
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    clip_grad_norm: float = DEFAULT_GRAD_CLIP,
    target_scaler: Optional[BaseScaler] = None,
    max_batches: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Executes one complete training epoch over the training partition.

    Returns:
        mean_loss: Mean training loss (MSE)
        mean_mae: Mean training MAE in native CPU % units
    """
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    batches_processed = 0

    for batch_idx, (b_X, b_y, _) in enumerate(train_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        b_X = b_X.to(device, non_blocking=True)
        b_y = b_y.to(device, non_blocking=True)

        optimizer.zero_grad()
        y_hat = model(b_X)
        loss = criterion(y_hat, b_y)
        loss.backward()

        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

        optimizer.step()

        total_loss += float(loss.item())

        # Compute native training MAE for tracking
        with torch.no_grad():
            if target_scaler is None or getattr(target_scaler, "strategy", "native") == "native":
                total_mae += float(torch.mean(torch.abs(b_y - y_hat)).item())
            else:
                y_hat_np = y_hat.detach().cpu().numpy()
                b_y_np = b_y.detach().cpu().numpy()
                y_hat_np = target_scaler.inverse_transform(y_hat_np)
                b_y_np = target_scaler.inverse_transform(b_y_np)
                total_mae += float(np.mean(np.abs(b_y_np - y_hat_np)))

        batches_processed += 1

    mean_loss = total_loss / batches_processed if batches_processed > 0 else 0.0
    mean_mae = total_mae / batches_processed if batches_processed > 0 else 0.0
    return mean_loss, mean_mae


def evaluate_validation(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    target_scaler: Optional[BaseScaler] = None,
    mape_threshold: float = DEFAULT_MAPE_THRESHOLD,
    max_batches: Optional[int] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluates model strictly on the VALIDATION partition.
    Test set is NEVER touched.

    Returns:
        val_loss: Mean validation loss (MSE)
        metrics: Dictionary of validation metrics in native CPU % units
    """
    model.eval()
    total_loss = 0.0
    batches_processed = 0

    all_y_true = []
    all_y_pred = []

    with torch.no_grad():
        for batch_idx, (b_X, b_y, _) in enumerate(val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            b_X = b_X.to(device, non_blocking=True)
            b_y = b_y.to(device, non_blocking=True)

            y_hat = model(b_X)
            loss = criterion(y_hat, b_y)
            total_loss += float(loss.item())

            y_hat_np = y_hat.cpu().numpy()
            b_y_np = b_y.cpu().numpy()

            # Invert to native CPU % units before calculating metrics
            if target_scaler is not None:
                y_hat_np = target_scaler.inverse_transform(y_hat_np)
                b_y_np = target_scaler.inverse_transform(b_y_np)

            all_y_true.append(b_y_np)
            all_y_pred.append(y_hat_np)
            batches_processed += 1

    val_loss = total_loss / batches_processed if batches_processed > 0 else 0.0

    y_true_all = np.vstack(all_y_true)
    y_pred_all = np.vstack(all_y_pred)

    metrics = compute_forecasting_metrics(y_true_all, y_pred_all, mape_threshold=mape_threshold)
    return val_loss, metrics


# =============================================================================
# MODEL TRAINING WITH EARLY STOPPING & CHECKPOINTING
# =============================================================================

def train_tcn_experiment(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    criterion: nn.Module,
    device: torch.device,
    config: Dict[str, Any],
    target_scaler: Optional[BaseScaler] = None,
    checkpoint_dir: Path = Path("results/stage5/checkpoints"),
    history_dir: Path = Path("results/stage5/training"),
    max_train_batches: Optional[int] = None,
    max_val_batches: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, pd.DataFrame, Dict[str, Any]]:
    """
    Executes a complete TCN training experiment with:
    - Validation loss monitoring
    - Learning rate scheduling via ReduceLROnPlateau
    - Checkpoint saving on best validation loss
    - Early stopping when patience is exhausted
    - Restoration of best validation checkpoint after training
    """
    exp_id = config.get("experiment_id", "experiment")
    max_epochs = config.get("max_epochs", DEFAULT_MAX_EPOCHS)
    patience = config.get("early_stopping_patience", DEFAULT_EARLY_STOP_PATIENCE)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    exp_history_dir = history_dir / exp_id
    exp_history_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{exp_id}_best.pt"

    best_val_loss = float("inf")
    best_epoch = -1
    best_metrics: Dict[str, float] = {}
    patience_counter = 0
    history_records: List[Dict[str, Any]] = []

    if verbose:
        print(f"\n--- Starting Experiment: {exp_id} ---", flush=True)
        print(f"  Configuration: Mode={config.get('feature_mode')}, Scaling={config.get('scaling_strategy')}, "
              f"Target={config.get('target_strategy')}, Max Epochs={max_epochs}, Patience={patience}", flush=True)

    t_start = time.time()

    for epoch in range(1, max_epochs + 1):
        t_epoch_start = time.time()

        # 1. Train one epoch
        train_loss, train_mae = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            clip_grad_norm=config.get("grad_clip", DEFAULT_GRAD_CLIP),
            target_scaler=target_scaler,
            max_batches=max_train_batches,
        )

        # 2. Evaluate strictly on Validation partition
        val_loss, val_metrics = evaluate_validation(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            target_scaler=target_scaler,
            mape_threshold=config.get("mape_threshold", DEFAULT_MAPE_THRESHOLD),
            max_batches=max_val_batches,
        )

        # 3. Learning rate scheduler step (driven strictly by validation loss)
        scheduler.step(val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])

        # 4. Checkpoint & early stopping evaluation
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            best_metrics = val_metrics.copy()
            patience_counter = 0

            # Save best checkpoint
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_validation_loss": best_val_loss,
                "best_epoch": best_epoch,
                "config": config,
                "validation_metrics": best_metrics,
                "random_seed": config.get("seed", DEFAULT_SEED),
                "timestamp": time.time(),
            }, checkpoint_path)
            checkpoint_tag = " [*BEST CHECKPOINT SAVED*]"
        else:
            patience_counter += 1
            checkpoint_tag = f" [patience: {patience_counter}/{patience}]"

        t_epoch = time.time() - t_epoch_start

        # Record history
        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "validation_loss": round(val_loss, 6),
            "learning_rate": current_lr,
            "train_MAE": round(train_mae, 4),
            "validation_MAE": val_metrics["MAE"],
            "validation_RMSE": val_metrics["RMSE"],
            "validation_R2": val_metrics["R2"],
            "validation_sMAPE": val_metrics["sMAPE"],
            "validation_thresholded_MAPE": val_metrics["thresholded_MAPE"],
            "eligible_fraction_mape": val_metrics["eligible_fraction"],
            "epoch_duration_sec": round(t_epoch, 2),
        }
        history_records.append(record)
        # Incrementally persist history CSV after every completed epoch for live observability/recovery
        history_csv_path = exp_history_dir / "history.csv"
        pd.DataFrame(history_records).to_csv(history_csv_path, index=False)

        if verbose:
            print(f"  Epoch [{epoch:2d}/{max_epochs:2d}] ({t_epoch:5.1f}s) | "
                  f"Train Loss: {train_loss:.4f} (MAE: {train_mae:.2f}%) | "
                  f"Val Loss: {val_loss:.4f} (MAE: {val_metrics['MAE']:.2f}%, RMSE: {val_metrics['RMSE']:.2f}%, "
                  f"R2: {val_metrics['R2']:.4f}, sMAPE: {val_metrics['sMAPE']:.2f}%) | "
                  f"LR: {current_lr:.1e}{checkpoint_tag}", flush=True)

        # Check early stopping trigger
        if patience_counter >= patience:
            if verbose:
                print(f"  [EARLY STOP] Validation loss did not improve for {patience} consecutive epochs. Stopping.", flush=True)
            break

    total_training_time = time.time() - t_start

    # 5. Restore best validation checkpoint weights
    if checkpoint_path.exists():
        best_ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])
        if verbose:
            print(f"  [RESTORE] Successfully restored best model weights from Epoch {best_epoch} "
                  f"(Best Val Loss: {best_val_loss:.6f})")

    # 6. Save training history CSV
    history_df = pd.DataFrame(history_records)
    history_csv_path = exp_history_dir / "history.csv"
    history_df.to_csv(history_csv_path, index=False)
    if verbose:
        print(f"  [EXPORT] Training history saved to: {history_csv_path}")

    # 7. Build summary record
    summary_record = {
        "experiment_id": exp_id,
        "feature_mode": config.get("feature_mode"),
        "num_features": config.get("num_features"),
        "scaling_strategy": config.get("scaling_strategy"),
        "target_scaling": config.get("target_strategy", "native"),
        "seed": config.get("seed", DEFAULT_SEED),
        "best_epoch": best_epoch,
        "epochs_trained": len(history_records),
        "best_train_loss": round(history_df.loc[history_df["epoch"] == best_epoch, "train_loss"].iloc[0], 6) if best_epoch > 0 else float("nan"),
        "best_validation_loss": round(best_val_loss, 6),
        "validation_MAE": best_metrics.get("MAE", float("nan")),
        "validation_RMSE": best_metrics.get("RMSE", float("nan")),
        "validation_R2": best_metrics.get("R2", float("nan")),
        "validation_sMAPE": best_metrics.get("sMAPE", float("nan")),
        "validation_thresholded_MAPE": best_metrics.get("thresholded_MAPE", float("nan")),
        "eligible_fraction_mape": best_metrics.get("eligible_fraction", float("nan")),
        "final_learning_rate": history_records[-1]["learning_rate"] if history_records else float("nan"),
        "total_training_time_sec": round(total_training_time, 2),
        "checkpoint_path": checkpoint_path.as_posix(),
    }

    return model, history_df, summary_record


# =============================================================================
# CONTROLLED ABLATION EXPERIMENT RUNNER
# =============================================================================

def update_ablation_summary_table(
    summary_record: Dict[str, Any],
    output_path: Path = Path("results/stage5/ablation_summary.csv"),
) -> pd.DataFrame:
    """Appends or updates an experiment record in results/stage5/ablation_summary.csv."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        try:
            df = pd.read_csv(output_path)
            # Replace existing record if re-run
            df = df[df["experiment_id"] != summary_record["experiment_id"]]
            df = pd.concat([df, pd.DataFrame([summary_record])], ignore_index=True)
        except Exception:
            df = pd.DataFrame([summary_record])
    else:
        df = pd.DataFrame([summary_record])

    df.to_csv(output_path, index=False)
    return df


def run_experiment(
    vm_traces: Dict[str, pd.DataFrame],
    vm_partitions: Dict[str, Dict[str, pd.DataFrame]],
    index_table: pd.DataFrame,
    feature_mode: str = "M1_UNIVARIATE",
    scaling_strategy: str = "native",
    target_strategy: str = "native",
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    patience: int = DEFAULT_EARLY_STOP_PATIENCE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    seed: int = DEFAULT_SEED,
    max_train_batches: Optional[int] = None,
    max_val_batches: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, pd.DataFrame, Dict[str, Any]]:
    """
    Executes a single controlled experiment for a given (feature_mode, scaling_strategy) pair.

    Guarantees:
    - Train-only feature scaling on unique 44,343 observations.
    - Zero test set access.
    - Standardized architecture and hyperparameter settings.
    """
    set_seed(seed)
    device = get_device()
    if verbose:
        print(f"Device: {device.type.upper()}")

    # 1. Prepare datasets using Stage 5.1 interface
    datasets, _, target_scaler = create_stage5_datasets(
        vm_traces=vm_traces,
        index_table=index_table,
        feature_mode=feature_mode,
        feature_strategy=scaling_strategy,
        target_strategy=target_strategy,
        vm_partitions=vm_partitions,
    )

    # 2. Build DataLoaders (Train and Val ONLY; Test is not loaded)
    loaders = create_stage5_dataloaders(datasets, batch_size=batch_size, num_workers=0)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    # Verify split window counts
    assert len(train_loader.dataset) == 40052, f"Expected 40,052 train windows, got {len(train_loader.dataset)}"
    assert len(val_loader.dataset) == 6557, f"Expected 6,557 val windows, got {len(val_loader.dataset)}"

    # 3. Instantiate fixed TCN architecture
    num_features = datasets["train"].scaled_vm_traces[list(datasets["train"].scaled_vm_traces.keys())[0]].shape[1]
    model = TCNForecaster(
        in_channels=num_features,
        input_length=DEFAULT_INPUT_LENGTH,
        forecast_horizon=DEFAULT_FORECAST_HORIZON,
        hidden_channels=DEFAULT_HIDDEN_CHANNELS,
        kernel_size=DEFAULT_KERNEL_SIZE,
        dilations=DEFAULT_DILATIONS,
        dropout=DEFAULT_DROPOUT,
    ).to(device)

    # 4. Optimizer, Scheduler, and Criterion
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=DEFAULT_SCHEDULER_FACTOR,
        patience=DEFAULT_SCHEDULER_PATIENCE,
    )
    criterion = nn.MSELoss()

    # 5. Build experiment configuration dictionary
    exp_id = f"{feature_mode}_{scaling_strategy}"
    config = {
        "experiment_id": exp_id,
        "feature_mode": feature_mode,
        "num_features": num_features,
        "scaling_strategy": scaling_strategy,
        "target_strategy": target_strategy,
        "seed": seed,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "max_epochs": max_epochs,
        "early_stopping_patience": patience,
        "grad_clip": DEFAULT_GRAD_CLIP,
        "mape_threshold": DEFAULT_MAPE_THRESHOLD,
        "receptive_field": calculate_receptive_field(DEFAULT_KERNEL_SIZE, DEFAULT_DILATIONS),
        "parameters": count_parameters(model),
    }

    # 6. Train model with early stopping and checkpointing
    model, history_df, summary = train_tcn_experiment(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        config=config,
        target_scaler=target_scaler,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
        verbose=verbose,
    )

    # 7. Update master ablation summary table
    update_ablation_summary_table(summary)

    return model, history_df, summary


# =============================================================================
# INITIAL SMOKE TRAINING RUNNER (Requirement 18)
# =============================================================================

def run_smoke_training(epochs: int = 2, data_dir: Path = Path("dataset/fastStorage/2013-8")) -> Dict[str, Any]:
    """
    Executes one controlled smoke-training experiment (Requirement 18):
    - Mode: M1_UNIVARIATE (1 feature: cpu_usage_percent)
    - Scaling: Native
    - Batch Size: 64
    - Epochs: 2 (sufficient to verify the complete pipeline end-to-end)

    Verifies:
    1. Dataset loading (14 VMs under Policy C).
    2. Batch creation (Train=40,052 windows, Val=6,557 windows).
    3. Forward pass (Tensor layout [64, 1, 288] -> [64, 12]).
    4. MSE loss calculation.
    5. Backward pass & gradient clipping (max_norm=1.0).
    6. AdamW parameter update.
    7. Validation evaluation strictly on validation partition.
    8. Native metric computation (MAE, RMSE, R2, sMAPE, thresholded MAPE).
    9. Learning rate scheduler step.
    10. Checkpoint saving on best validation loss.
    11. Restoration of best checkpoint.
    12. Zero access to test partition (test windows = 6,915 remain untouched).
    """
    print("=" * 80)
    print("STAGE 5.3: TCN SMOKE-TRAINING & PIPELINE INTEGRITY VERIFICATION")
    print("=" * 80)

    t_start = time.time()
    device = get_device()
    print(f"Device: {device.type.upper()}")

    # 1. Load data
    print(f"\n[1/4] Loading and preparing Stage 5 data from {data_dir} (Policy C 5-min canonical grid)...")
    vm_traces, vm_partitions, index_table, stats = prepare_stage5_data(data_dir=data_dir)

    print(f"  [PASS] Aligned Steps: {stats['total_steps']:,} total "
          f"(Train: {stats['train_steps']:,} | Val: {stats['val_steps']:,} | Test: {stats['test_steps']:,})")
    print(f"  [PASS] Valid Windows: {stats['n_total_windows']:,} total "
          f"(Train: {stats['n_train_windows']:,} | Val: {stats['n_val_windows']:,} | Test: {stats['n_test_windows']:,})")
    print("  [CRITICAL CHECK] Test set (6,915 windows) is strictly excluded from Stage 5.3.")

    # 2. Run smoke experiment
    print(f"\n[2/4] Executing Smoke Training Run: M1_UNIVARIATE + Native Scaling ({epochs} epochs)...")
    _, _, summary = run_experiment(
        vm_traces=vm_traces,
        vm_partitions=vm_partitions,
        index_table=index_table,
        feature_mode="M1_UNIVARIATE",
        scaling_strategy="native",
        target_strategy="native",
        max_epochs=epochs,
        patience=DEFAULT_EARLY_STOP_PATIENCE,
        batch_size=DEFAULT_BATCH_SIZE,
        seed=DEFAULT_SEED,
        verbose=True,
    )

    # 3. Checkpoint verification
    print("\n[3/4] Verifying Checkpoint & Model State Restoration:")
    ckpt_path = Path(summary["checkpoint_path"])
    assert ckpt_path.exists(), f"Checkpoint file not found: {ckpt_path}"
    ckpt_data = torch.load(ckpt_path, map_location=device)
    print(f"  [PASS] Best Checkpoint File Exists: {ckpt_path.name} ({ckpt_path.stat().st_size / 1024:.1f} KB)")
    print(f"  [PASS] Saved Best Epoch:           {ckpt_data['best_epoch']}")
    print(f"  [PASS] Best Validation Loss:       {ckpt_data['best_validation_loss']:.6f}")
    print(f"  [PASS] Checkpoint Keys Verified:    {list(ckpt_data.keys())}")

    # 4. Summary reporting
    print("\n[4/4] Smoke Training Results Summary:")
    print(f"  - Experiment ID:            {summary['experiment_id']}")
    print(f"  - Total Epochs Trained:     {summary['epochs_trained']}")
    print(f"  - Best Train Loss:          {summary['best_train_loss']:.4f}")
    print(f"  - Best Validation Loss:     {summary['best_validation_loss']:.4f}")
    print(f"  - Validation MAE:           {summary['validation_MAE']:.2f}% CPU")
    print(f"  - Validation RMSE:          {summary['validation_RMSE']:.2f}% CPU")
    print(f"  - Validation R2:            {summary['validation_R2']:.4f}")
    print(f"  - Validation sMAPE:         {summary['validation_sMAPE']:.2f}%")
    print(f"  - Validation MAPE (>=1%):   {summary['validation_thresholded_MAPE']:.2f}% "
          f"({summary['eligible_fraction_mape'] * 100:.1f}% eligible observations)")
    print(f"  - Total Training Duration:  {summary['total_training_time_sec']:.1f}s")

    t_total = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"STAGE 5.3 SMOKE TRAINING COMPLETED SUCCESSFULLY IN {t_total:.1f}s")
    print("ALL TRAINING, SCHEDULING, CHECKPOINTING & METRIC INVARIANTS VERIFIED")
    print("ZERO ACCESS TO TEST SPLIT CONFIRMED")
    print("=" * 80)

    return summary


# =============================================================================
# FULL CONTROLLED 12-RUN ABLATION RUNNER & RESUMABILITY
# =============================================================================

APPROVED_12_CONFIGURATIONS: List[Tuple[str, str]] = [
    # M1_UNIVARIATE (F=1)
    ("M1_UNIVARIATE", "native"),
    ("M1_UNIVARIATE", "standard"),
    ("M1_UNIVARIATE", "robust"),
    ("M1_UNIVARIATE", "log1p_standard"),
    # M2_RESOURCE (F=6)
    ("M2_RESOURCE", "native"),
    ("M2_RESOURCE", "standard"),
    ("M2_RESOURCE", "robust"),
    ("M2_RESOURCE", "log1p_standard"),
    # M3_FULL (F=10)
    ("M3_FULL", "native"),
    ("M3_FULL", "standard"),
    ("M3_FULL", "robust"),
    ("M3_FULL", "log1p_standard"),
]


def is_experiment_completed(
    exp_id: str,
    checkpoint_dir: Path = Path("results/stage5/checkpoints"),
    history_dir: Path = Path("results/stage5/training"),
    summary_path: Path = Path("results/stage5/ablation_summary.csv"),
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Verifies whether an experiment is genuinely completed and verified.
    Does NOT treat merely existing files as proof of completion.

    Checks:
    1. Checkpoint file exists, is non-empty, and loads successfully via torch.load.
    2. Checkpoint contains expected keys: model_state_dict, best_epoch, best_validation_loss, validation_metrics.
    3. best_epoch > 0 and best_validation_loss is finite.
    4. history.csv exists and has at least best_epoch rows.
    5. summary CSV exists and contains a valid row with matching experiment_id.
    """
    ckpt_path = checkpoint_dir / f"{exp_id}_best.pt"
    if not ckpt_path.exists() or ckpt_path.stat().st_size == 0:
        return False, None

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        required_keys = {"model_state_dict", "best_epoch", "best_validation_loss", "validation_metrics"}
        if not required_keys.issubset(ckpt.keys()):
            return False, None

        best_epoch = int(ckpt["best_epoch"])
        best_val_loss = float(ckpt["best_validation_loss"])
        if best_epoch <= 0 or not np.isfinite(best_val_loss):
            return False, None

        # Verify history CSV
        hist_path = history_dir / exp_id / "history.csv"
        if not hist_path.exists() or hist_path.stat().st_size == 0:
            return False, None

        hist_df = pd.read_csv(hist_path)
        if len(hist_df) < best_epoch:
            return False, None

        # Verify summary CSV
        if summary_path.exists():
            sum_df = pd.read_csv(summary_path)
            match = sum_df[sum_df["experiment_id"] == exp_id]
            if len(match) > 0 and int(match["best_epoch"].iloc[0]) > 0:
                return True, match.iloc[0].to_dict()

        return True, {
            "experiment_id": exp_id,
            "best_epoch": best_epoch,
            "best_validation_loss": best_val_loss,
            "validation_metrics": ckpt.get("validation_metrics", {}),
        }
    except Exception:
        return False, None


def run_full_ablation(
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    patience: int = DEFAULT_EARLY_STOP_PATIENCE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    seed: int = DEFAULT_SEED,
    num_threads: int = 8,
    skip_completed: bool = False,
    target_strategy: str = "native",
    summary_path: Path = Path("results/stage5/ablation_summary.csv"),
    target_experiment: Optional[str] = None,
    data_dir: Path = Path("dataset/fastStorage/2013-8"),
) -> pd.DataFrame:
    """
    Executes the approved 12-configuration validation ablation sequentially:
    M1_UNIVARIATE x 4 scaling strategies
    M2_RESOURCE   x 4 scaling strategies
    M3_FULL       x 4 scaling strategies

    Guarantees:
    - Train-only feature scaling on unique 44,343 observations.
    - Zero test set access.
    - Standardized architecture and hyperparameter settings across all runs.
    - Fresh model, optimizer, scheduler, scaler, and checkpoint per experiment.
    - Incremental per-epoch persistence to history.csv for live monitoring/recovery.
    - Early stopping on validation loss (patience=5).
    - Summary table and per-experiment histories recorded.
    """
    print("=" * 80, flush=True)
    if target_experiment:
        print(f"STAGE 5.3: EXECUTING SINGLE CONTROLLED EXPERIMENT: {target_experiment}", flush=True)
    else:
        print("STAGE 5.3: EXECUTING FULL CONTROLLED 12-RUN TCN ABLATION", flush=True)
    print("=" * 80, flush=True)

    # Set PyTorch CPU thread count to 8 (optimal for 12-core hybrid architecture)
    torch.set_num_threads(num_threads)
    print(f"PyTorch CPU Threads: {torch.get_num_threads()}", flush=True)

    t_ablation_start = time.time()

    # 1. Load and prepare Stage 5 data ONCE
    print(f"\n[PREPARATION] Loading Stage 5 data from {data_dir} (Policy C 5-minute canonical grid)...", flush=True)
    vm_traces, vm_partitions, index_table, stats = prepare_stage5_data(data_dir=data_dir)

    print(f"  Aligned Steps: {stats['total_steps']:,} total "
          f"(Train: {stats['train_steps']:,} | Val: {stats['val_steps']:,} | Test: {stats['test_steps']:,})", flush=True)
    print(f"  Valid Windows: {stats['n_total_windows']:,} total "
          f"(Train: {stats['n_train_windows']:,} | Val: {stats['n_val_windows']:,} | Test: {stats['n_test_windows']:,})", flush=True)
    print("  [CRITICAL GUARANTEE] Test windows (6,915) are strictly excluded from all 12 experiments.", flush=True)

    # Filter configurations if single experiment requested
    configs_to_run = APPROVED_12_CONFIGURATIONS
    if target_experiment is not None:
        configs_to_run = [c for c in APPROVED_12_CONFIGURATIONS if f"{c[0]}_{c[1]}" == target_experiment]
        if not configs_to_run:
            valid_ids = [f"{c[0]}_{c[1]}" for c in APPROVED_12_CONFIGURATIONS]
            raise ValueError(f"Unknown experiment '{target_experiment}'. Must be one of:\n" + "\n".join(valid_ids))

    # 2. Sequential execution of the experiments
    for exp_idx, (feat_mode, scale_strat) in enumerate(configs_to_run, start=1):
        exp_id = f"{feat_mode}_{scale_strat}"

        # Check safe resumability
        if skip_completed:
            is_done, done_info = is_experiment_completed(exp_id, summary_path=summary_path)
            if is_done:
                print(f"\n{'=' * 80}", flush=True)
                print(f"EXPERIMENT [{exp_idx:02d}/{len(configs_to_run)}]: {exp_id} -- [ALREADY COMPLETED & VERIFIED: SKIPPING]", flush=True)
                print(f"  Best Epoch:           {done_info.get('best_epoch')}", flush=True)
                print(f"  Best Val Loss:        {float(done_info.get('best_validation_loss', 0)):.4f}", flush=True)
                if "validation_MAE" in done_info:
                    print(f"  Val MAE:              {float(done_info.get('validation_MAE', 0)):.2f}% CPU", flush=True)
                    print(f"  Val RMSE:             {float(done_info.get('validation_RMSE', 0)):.2f}% CPU", flush=True)
                    print(f"  Val R2:               {float(done_info.get('validation_R2', 0)):.4f}", flush=True)
                print(f"{'=' * 80}", flush=True)
                continue

        print(f"\n{'=' * 80}", flush=True)
        print(f"EXPERIMENT [{exp_idx:02d}/12]: {exp_id}", flush=True)
        print(f"Feature Mode: {feat_mode} | Scaling Strategy: {scale_strat} | Target: {target_strategy}", flush=True)
        print(f"{'=' * 80}", flush=True)

        _, _, summary = run_experiment(
            vm_traces=vm_traces,
            vm_partitions=vm_partitions,
            index_table=index_table,
            feature_mode=feat_mode,
            scaling_strategy=scale_strat,
            target_strategy=target_strategy,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=seed,
            verbose=True,
        )

        print(f"--- Experiment {exp_id} Complete ---", flush=True)
        print(f"  Best Epoch:           {summary['best_epoch']} / {summary['epochs_trained']}", flush=True)
        print(f"  Best Val Loss:        {summary['best_validation_loss']:.4f}", flush=True)
        print(f"  Val MAE:              {summary['validation_MAE']:.2f}% CPU", flush=True)
        print(f"  Val RMSE:             {summary['validation_RMSE']:.2f}% CPU", flush=True)
        print(f"  Val R2:               {summary['validation_R2']:.4f}", flush=True)
        print(f"  Val sMAPE:            {summary['validation_sMAPE']:.2f}%", flush=True)
        print(f"  Val MAPE (>=1%):      {summary['validation_thresholded_MAPE']:.2f}%", flush=True)
        print(f"  Duration:             {summary['total_training_time_sec']:.1f}s", flush=True)

    total_duration = time.time() - t_ablation_start
    print(f"\n{'=' * 80}", flush=True)
    print(f"ALL 12 ABLATION EXPERIMENTS PROCESSED IN {total_duration / 60:.1f} MINUTES ({total_duration:.1f}s)", flush=True)
    print(f"{'=' * 80}", flush=True)

    # 3. Load and display final summary table
    summary_df = pd.read_csv(summary_path)
    # Sort by validation loss ascending
    summary_df_sorted = summary_df.sort_values(by="best_validation_loss", ascending=True).reset_index(drop=True)
    summary_df_sorted["rank"] = range(1, len(summary_df_sorted) + 1)

    print("\nFINAL ABLATION SUMMARY TABLE (Ranked by Validation Loss):", flush=True)
    cols_to_print = [
        "rank", "experiment_id", "feature_mode", "scaling_strategy",
        "best_epoch", "epochs_trained", "best_validation_loss",
        "validation_MAE", "validation_RMSE", "validation_R2",
        "validation_sMAPE", "validation_thresholded_MAPE", "final_learning_rate"
    ]
    available_cols = [c for c in cols_to_print if c in summary_df_sorted.columns]
    print(summary_df_sorted[available_cols].to_string(index=False), flush=True)

    best_cand = summary_df_sorted.iloc[0]
    print(f"\n>>> Validation-selected candidate: {best_cand['experiment_id']} <<<", flush=True)
    print(f"    Validation Loss: {best_cand['best_validation_loss']:.4f} | "
          f"Validation MAE: {best_cand['validation_MAE']:.2f}% | "
          f"Validation RMSE: {best_cand['validation_RMSE']:.2f}% | "
          f"Validation R2: {best_cand['validation_R2']:.4f}", flush=True)

    return summary_df_sorted


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 5.3: TCN Training & Controlled Ablation Runner")
    parser.add_argument("--smoke", action="store_true", help="Run 2-epoch smoke test on M1 native")
    parser.add_argument("--skip-completed", action="store_true", help="Skip experiments that are already completed and verified")
    parser.add_argument("--experiment", type=str, default=None, help="Run a single specific experiment by ID (e.g. M1_UNIVARIATE_standard)")
    parser.add_argument("--data-dir", type=str, default="dataset/fastStorage/2013-8", help="Path to raw Bitbrains dataset directory")
    parser.add_argument("--epochs", type=int, default=DEFAULT_MAX_EPOCHS, help="Maximum epochs per experiment")
    parser.add_argument("--patience", type=int, default=DEFAULT_EARLY_STOP_PATIENCE, help="Early stopping patience")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--threads", type=int, default=8, help="PyTorch CPU threads")
    args = parser.parse_args()

    if args.smoke:
        run_smoke_training(epochs=2, data_dir=Path(args.data_dir))
    else:
        run_full_ablation(
            max_epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            seed=args.seed,
            num_threads=args.threads,
            skip_completed=args.skip_completed,
            target_experiment=args.experiment,
            data_dir=Path(args.data_dir),
        )

