"""
Gen_LSTM: Isolated Training Pipeline for Controlled Model Comparison
======================================================================
Project: Predictive Energy-Efficient VM Placement in Cloud Using ML and Adaptive Hippopotamus Optimization
Scope: Dedicated LSTM Training Routine preserving exact TCN optimization protocol

Protocols & Invariants (Identical to Gen 1 TCN):
- Deterministic seed = 42
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
- Loss: MSELoss
- Gradient clipping: max_norm = 1.0
- Scheduler: ReduceLROnPlateau(mode='min', factor=0.5, patience=2, min_lr=1e-5)
- Early Stopping: patience = 5 epochs on validation loss
- Validation evaluation in native CPU% units
- Checkpoint: Saves best validation model to checkpoints/gen_lstm_best.pt
"""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from ml.lstm_model import LSTMForecaster, count_trainable_parameters
from ml.tcn_dataset import BaseScaler, TargetScaler
from ml.tcn_train import compute_forecasting_metrics

# Default hyper-parameters locked to Gen 1 TCN parity
DEFAULT_LEARNING_RATE: float = 1e-3
DEFAULT_WEIGHT_DECAY: float = 1e-4
DEFAULT_GRAD_CLIP: float = 1.0
DEFAULT_EARLY_STOP_PATIENCE: int = 5
DEFAULT_SCHEDULER_PATIENCE: int = 2
DEFAULT_SCHEDULER_FACTOR: float = 0.5
DEFAULT_MIN_LR: float = 1e-5
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_MAX_EPOCHS: int = 50
DEFAULT_SEED: int = 42


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Sets deterministic random seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(requested_device: Optional[str] = None) -> torch.device:
    """Selects target torch device."""
    if requested_device is not None:
        return torch.device(requested_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    mape_threshold: float = 1.0,
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

            if target_scaler is not None and getattr(target_scaler, "strategy", "native") != "native":
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


def train_lstm_experiment(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    criterion: nn.Module,
    device: torch.device,
    config: Dict[str, Any],
    target_scaler: Optional[BaseScaler] = None,
    checkpoint_dir: Path = Path("results/gen_lstm/checkpoints"),
    history_dir: Path = Path("results/gen_lstm/training"),
    max_train_batches: Optional[int] = None,
    max_val_batches: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, pd.DataFrame, Dict[str, Any]]:
    """
    Executes a complete Gen_LSTM training experiment with EarlyStopping and ReduceLROnPlateau.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    max_epochs = config.get("epochs", DEFAULT_MAX_EPOCHS)
    patience = config.get("early_stop_patience", DEFAULT_EARLY_STOP_PATIENCE)
    clip_grad_norm = config.get("gradient_clip", DEFAULT_GRAD_CLIP)
    exp_name = config.get("experiment_id", "gen_lstm")

    checkpoint_path = checkpoint_dir / f"{exp_name}_best.pt"
    history_csv = history_dir / "history.csv"

    best_val_loss = float("inf")
    best_epoch = 0
    best_metrics: Dict[str, float] = {}
    epochs_no_improve = 0

    history_records = []
    total_train_start = time.time()

    if verbose:
        print("=" * 80)
        print(f"STARTING GEN_LSTM TRAINING: {exp_name}")
        print(f"Device:           {device}")
        print(f"Parameters:       {count_trainable_parameters(model):,}")
        print(f"Max Epochs:       {max_epochs} (Early Stopping Patience: {patience})")
        print(f"Initial LR:       {optimizer.param_groups[0]['lr']}")
        print(f"Checkpoint Path:  {checkpoint_path}")
        print("=" * 80)

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        train_loss, train_mae = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            clip_grad_norm=clip_grad_norm,
            target_scaler=target_scaler,
            max_batches=max_train_batches,
        )

        val_loss, val_metrics = evaluate_validation(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            target_scaler=target_scaler,
            max_batches=max_val_batches,
        )

        current_lr = float(optimizer.param_groups[0]["lr"])
        if scheduler is not None:
            scheduler.step(val_loss)

        epoch_duration = time.time() - t0

        history_records.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_mae": train_mae,
            "val_mae": val_metrics["MAE"],
            "learning_rate": current_lr,
            "epoch_time_sec": round(epoch_duration, 2),
        })

        if verbose:
            print(
                f"Epoch {epoch:2d}/{max_epochs:2d} ({epoch_duration:5.1f}s) | "
                f"Train MSE: {train_loss:8.4f} | Val MSE: {val_loss:8.4f} | "
                f"Val MAE: {val_metrics['MAE']:5.2f}% | Val R2: {val_metrics['R2']:6.3f} | "
                f"LR: {current_lr:.1e}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_metrics = val_metrics
            epochs_no_improve = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_val_loss,
                    "best_metrics": best_metrics,
                    "config": config,
                },
                checkpoint_path,
            )
            if verbose:
                print(f"  --> Checkpoint saved (New best validation loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"\n[EARLY STOPPING TRIGGERED] No improvement for {patience} consecutive epochs.")
                    print(f"Restoring best checkpoint from Epoch {best_epoch} (Val Loss: {best_val_loss:.4f}).")
                break

    df_history = pd.DataFrame(history_records)
    df_history.to_csv(history_csv, index=False)

    # Restore best weights
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    total_duration = time.time() - total_train_start

    summary = {
        "experiment_id": exp_name,
        "best_epoch": best_epoch,
        "epochs_trained": len(df_history),
        "best_validation_loss": best_val_loss,
        "best_validation_metrics": best_metrics,
        "total_training_time_sec": round(total_duration, 2),
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(history_csv),
    }

    if verbose:
        print("=" * 80)
        print("TRAINING RUN COMPLETE")
        print(f"Best Epoch:            {best_epoch}")
        print(f"Best Validation Loss:  {best_val_loss:.4f}")
        print(f"Best Validation MAE:   {best_metrics.get('MAE', float('nan')):.4f}% CPU")
        print(f"Total Training Time:   {total_duration:.1f}s")
        print("=" * 80)

    return model, df_history, summary
