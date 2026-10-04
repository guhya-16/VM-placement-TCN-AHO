"""
Gen_LSTM: Isolated LSTM Forecasting Architecture for Controlled Model Comparison
==================================================================================
Project: Predictive Energy-Efficient VM Placement in Cloud Using ML and Adaptive Hippopotamus Optimization
Scope: Controlled Model Comparison (Gen 1 TCN vs. Gen LSTM on exact same 14 Bitbrains VMs)

Model Specification:
- Class: LSTMForecaster
- Input size (features): 10 (M3_FULL: 6 telemetry resource features + 4 cyclical calendar features)
- Input sequence length: 288 historical steps (24 hours at 5-minute sampling)
- Hidden size: 64
- Number of stacked LSTM layers: 2
- Dropout: 0.10 (applied inter-layer and before output linear head)
- Output horizon: 12 future steps (1 hour ahead at 5-minute sampling)
- Output head: nn.Linear(64, 12)
- Forecasting strategy: Direct Multi-Horizon (Sequence-to-Vector)
- Expected trainable parameters: 53,516

Academic Note:
The LSTM has 53,516 trainable parameters, approximately 10.8% more than the 48,300-parameter TCN.
Both models therefore operate at a comparable parameter-count scale, while their temporal
modeling mechanisms remain fundamentally different.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

# Default frozen architecture hyper-parameters
DEFAULT_INPUT_SIZE: int = 10
DEFAULT_INPUT_LENGTH: int = 288
DEFAULT_HIDDEN_SIZE: int = 64
DEFAULT_NUM_LAYERS: int = 2
DEFAULT_DROPOUT: float = 0.10
DEFAULT_FORECAST_HORIZON: int = 12
EXPECTED_PARAM_COUNT: int = 53516


def count_trainable_parameters(model: nn.Module) -> int:
    """Returns the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class LSTMForecaster(nn.Module):
    """
    2-Layer Stacked Direct Multi-Horizon LSTM Forecaster.

    Processes 288 historical steps of 10-feature telemetry and directly predicts
    the 12 future CPU utilization steps without recursive autoregressive compounding.

    Input tensor shapes supported:
    - [Batch, 288, 10]: standard sequence layout (Length=288, Features=10)
    - [Batch, 10, 288]: channel-first layout (from BitbrainsWindowDataset)

    Output tensor shape:
    - [Batch, 12]: predicted CPU usage percent for steps t+1 ... t+12
    """

    def __init__(
        self,
        in_features: int = DEFAULT_INPUT_SIZE,
        input_length: int = DEFAULT_INPUT_LENGTH,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        num_layers: int = DEFAULT_NUM_LAYERS,
        forecast_horizon: int = DEFAULT_FORECAST_HORIZON,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.input_length = int(input_length)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.forecast_horizon = int(forecast_horizon)
        self.dropout_rate = float(dropout)

        # 2-layer stacked LSTM
        # Inter-layer dropout is applied between layer 1 and layer 2 when num_layers > 1
        self.lstm = nn.LSTM(
            input_size=self.in_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0.0,
        )

        # Regularization before the output projection head
        self.dropout = nn.Dropout(p=self.dropout_rate)

        # Direct multi-horizon linear projection head mapping final hidden state to H=12
        self.fc_out = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.forecast_horizon,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initializes weights using standard Xavier/Glorot uniform initialization
        for projection and orthogonal initialization for LSTM recurrent weights.
        """
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
                # Initialize forget gate bias to 1.0 for improved gradient flow
                # Forget gate bias is at offset [hidden_size : 2 * hidden_size]
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.0)

        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor. Either [Batch, 288, 10] or [Batch, 10, 288].

        Returns:
            predictions: [Batch, 12] in native CPU usage percentage units.
        """
        # Shape normalization: accommodate both [B, L, F] and [B, F, L]
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input tensor [Batch, Length, Features], got shape {x.shape}")

        if x.size(1) == self.in_features and x.size(2) == self.input_length:
            # Transpose from [B, F=10, L=288] to [B, L=288, F=10]
            x = x.transpose(1, 2)
        elif x.size(1) == self.input_length and x.size(2) == self.in_features:
            # Already [B, L=288, F=10]
            pass
        else:
            raise ValueError(
                f"Input tensor dimensions mismatch: expected ({self.input_length}, {self.in_features}) "
                f"or ({self.in_features}, {self.input_length}), got {tuple(x.shape[1:])}"
            )

        # Run LSTM: out has shape [Batch, 288, 64]
        # h_n has shape [num_layers=2, Batch, 64]
        out, (h_n, c_n) = self.lstm(x)

        # Extract final layer hidden state at the last time step
        # h_n[-1] corresponds to the top LSTM layer's hidden state for all batch items
        final_hidden = h_n[-1]  # [Batch, 64]

        # Regularization & direct multi-horizon projection
        h_drop = self.dropout(final_hidden)
        predictions = self.fc_out(h_drop)  # [Batch, 12]

        return predictions


def verify_lstm_architecture(verbose: bool = True) -> Dict[str, Any]:
    """
    Validates parameter counts, layer specifications, and input/output contracts.
    """
    model = LSTMForecaster()
    param_count = count_trainable_parameters(model)

    # Detailed layer parameter audit
    layer1_ih = model.lstm.weight_ih_l0.numel()
    layer1_hh = model.lstm.weight_hh_l0.numel()
    layer1_bih = model.lstm.bias_ih_l0.numel()
    layer1_bhh = model.lstm.bias_hh_l0.numel()
    layer1_total = layer1_ih + layer1_hh + layer1_bih + layer1_bhh

    layer2_ih = model.lstm.weight_ih_l1.numel()
    layer2_hh = model.lstm.weight_hh_l1.numel()
    layer2_bih = model.lstm.bias_ih_l1.numel()
    layer2_bhh = model.lstm.bias_hh_l1.numel()
    layer2_total = layer2_ih + layer2_hh + layer2_bih + layer2_bhh

    fc_w = model.fc_out.weight.numel()
    fc_b = model.fc_out.bias.numel()
    fc_total = fc_w + fc_b

    assert layer1_total == 19456, f"Layer 1 param mismatch: {layer1_total}"
    assert layer2_total == 33280, f"Layer 2 param mismatch: {layer2_total}"
    assert fc_total == 780, f"FC head param mismatch: {fc_total}"
    assert param_count == EXPECTED_PARAM_COUNT, f"Total param mismatch: {param_count} != {EXPECTED_PARAM_COUNT}"

    # Forward pass test with both layouts
    dummy_blf = torch.randn(4, 288, 10)
    dummy_bfl = torch.randn(4, 10, 288)

    model.eval()
    with torch.no_grad():
        out_blf = model(dummy_blf)
        out_bfl = model(dummy_bfl)

    assert out_blf.shape == (4, 12), f"Shape mismatch: {out_blf.shape}"
    assert out_bfl.shape == (4, 12), f"Shape mismatch: {out_bfl.shape}"
    assert torch.isfinite(out_blf).all(), "NaN or Inf in output"

    audit_result = {
        "model_class": "LSTMForecaster",
        "in_features": model.in_features,
        "input_length": model.input_length,
        "hidden_size": model.hidden_size,
        "num_layers": model.num_layers,
        "forecast_horizon": model.forecast_horizon,
        "dropout": model.dropout_rate,
        "layer1_params": layer1_total,
        "layer2_params": layer2_total,
        "output_head_params": fc_total,
        "total_trainable_parameters": param_count,
        "expected_parameters": EXPECTED_PARAM_COUNT,
        "parameter_parity_with_tcn": {
            "tcn_parameters": 48300,
            "lstm_parameters": param_count,
            "absolute_diff": param_count - 48300,
            "relative_diff_percent": round((param_count - 48300) / 48300 * 100.0, 2),
            "statement": (
                "The LSTM has 53,516 trainable parameters, approximately 10.8% more than "
                "the 48,300-parameter TCN. Both models therefore operate at a comparable "
                "parameter-count scale, while their temporal modeling mechanisms remain "
                "fundamentally different."
            ),
        },
        "forward_test_blf_shape": list(out_blf.shape),
        "forward_test_bfl_shape": list(out_bfl.shape),
        "forward_test_finite": True,
    }

    if verbose:
        print("=" * 80)
        print("GEN_LSTM ARCHITECTURE VERIFICATION AUDIT")
        print("=" * 80)
        print(f"Model Class:        {audit_result['model_class']}")
        print(f"Layer 1 Params:     {layer1_total:,} (Input=10 -> Hidden=64)")
        print(f"Layer 2 Params:     {layer2_total:,} (Hidden=64 -> Hidden=64)")
        print(f"Output Head Params: {fc_total:,} (Linear 64 -> 12)")
        print(f"Total Parameters:   {param_count:,} (Expected: {EXPECTED_PARAM_COUNT:,}) [PASS]")
        print(f"TCN Parameters:     48,300 (+{param_count - 48300:,} / +10.80%)")
        print(f"Forward Pass Test:  [4, 288, 10] -> {list(out_blf.shape)} [PASS]")
        print(f"Channel-First Test: [4, 10, 288] -> {list(out_bfl.shape)} [PASS]")
        print("=" * 80)

    return audit_result


if __name__ == "__main__":
    verify_lstm_architecture(verbose=True)

