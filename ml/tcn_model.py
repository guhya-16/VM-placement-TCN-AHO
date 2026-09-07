"""
Stage 5.2: Temporal Convolutional Network (TCN) Model Architecture
===================================================================
Implements a causal dilated 1D convolutional network with residual blocks
for multi-step cloud workload forecasting (L=288 history steps -> H=12 forecast steps).

Core Architectural Principles:
------------------------------
1. Strict Temporal Causality:
   - Output representation at time t depends exclusively on inputs at or before time t:
     x[t], x[t-1], ..., x[0].
   - Causal padding is applied exclusively on the left (past):
     pad_left = (kernel_size - 1) * dilation, pad_right = 0.
   - Future information leakage is mathematically and computationally impossible.

2. Receptive Field Design:
   - Approved dilation schedule: [1, 2, 4, 8, 16, 32, 64, 128] (8 residual blocks).
   - Kernel size: k = 3.
   - Theoretical Receptive Field:
       RF = 1 + (k - 1) * sum(dilations)
          = 1 + (3 - 1) * (1 + 2 + 4 + 8 + 16 + 32 + 64 + 128)
          = 1 + 2 * 255 = 511 steps.
   - Since RF = 511 >= 288 steps (24.0 hours), the convolutional backbone has full
     causal receptive access across the entire 24-hour input history.

3. Modular Residual Blocks:
   - Each block consists of:
       CausalConv1d(in, out, k, d) -> ReLU -> Dropout
       -> CausalConv1d(out, out, k, d) -> ReLU -> Dropout
       -> Residual addition (with 1x1 conv if in_channels != out_channels)
       -> ReLU activation.

4. Decoupled Multi-Step Output Head:
   - The TCN feature state at cutoff step t (the final historical position index -1)
     encapsulates the causally aggregated 24-hour context: h_t in R^[B, hidden_channels].
   - A simple linear projection head (nn.Linear) maps h_t directly to y_hat in R^[B, 12].

5. Universal Feature Mode Support:
   - The exact same architecture class supports all Stage 4 feature modes by simply
     configuring in_channels:
       * M1_UNIVARIATE: in_channels = 1  (cpu_usage_percent)
       * M2_RESOURCE:   in_channels = 6  (cpu, ram, disk_r, disk_w, net_rx, net_tx)
       * M3_FULL:       in_channels = 10 (M2 + hour_sin, hour_cos, day_sin, day_cos)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Default architectural hyperparameters approved for Stage 5
DEFAULT_INPUT_LENGTH = 288
DEFAULT_FORECAST_HORIZON = 12
DEFAULT_HIDDEN_CHANNELS = 32
DEFAULT_KERNEL_SIZE = 3
DEFAULT_DILATIONS = [1, 2, 4, 8, 16, 32, 64, 128]
DEFAULT_DROPOUT = 0.1


def calculate_receptive_field(
    kernel_size: int = DEFAULT_KERNEL_SIZE,
    dilations: Optional[List[int]] = None,
) -> int:
    """
    Calculates the theoretical receptive field of the causal dilated TCN.

    Formula:
        RF = 1 + (kernel_size - 1) * sum(dilations)

    For k=3 and dilations=[1, 2, 4, 8, 16, 32, 64, 128]:
        sum(dilations) = 1 + 2 + 4 + 8 + 16 + 32 + 64 + 128 = 255
        RF = 1 + (3 - 1) * 255 = 1 + 2 * 255 = 511 steps.

    Verification:
        RF = 511 >= 288 steps (24.0 hours), ensuring every output prediction
        has causal coverage over the entire 24-hour historical window.
    """
    d_list = dilations if dilations is not None else DEFAULT_DILATIONS
    rf = 1 + (kernel_size - 1) * sum(d_list)
    return int(rf)


class CausalConv1d(nn.Module):
    """
    1D Causal Convolution layer.

    Enforces temporal causality by prepending padding strictly to the left (past)
    of the sequence, with zero padding on the right (future).

    For kernel size k and dilation d:
        pad_left = (k - 1) * d
        pad_right = 0
    Output sequence length is strictly invariant: T_out == T_in.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.padding = (self.kernel_size - 1) * self.dilation

        self.conv = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            dilation=self.dilation,
            padding=0,  # Explicit manual padding via F.pad
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, in_channels, T]
        Returns:
            Tensor of shape [B, out_channels, T]
        """
        # Prepend causal padding to the temporal dimension (last axis)
        x_padded = F.pad(x, (self.padding, 0))
        return self.conv(x_padded)


class TemporalBlock(nn.Module):
    """
    Modular Residual Block for Dilated Causal Convolution Networks.

    Structure:
        x
        ├──> CausalConv1d(in, out, k, d) -> ReLU -> Dropout -> CausalConv1d(out, out, k, d) -> ReLU -> Dropout ──(+)──> ReLU -> out
        └──> Residual connection: 1x1 Conv(in, out) if in != out else Identity ──────────────────────────────────┘

    Key Properties:
    - 2 causal convolutional layers per block.
    - Identity shortcut when in_channels == out_channels; 1x1 Conv shortcut when dimensions differ.
    - Configurable dropout for regularization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        dilation: int = 1,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)

        # First causal conv block
        self.conv1 = CausalConv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second causal conv block
        self.conv2 = CausalConv1d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Residual shortcut: 1x1 conv if channel dimensions change, else identity
        if self.in_channels != self.out_channels:
            self.downsample: Optional[nn.Module] = nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
            )
        else:
            self.downsample = None

        self.relu_out = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, in_channels, T]
        Returns:
            Output tensor of shape [B, out_channels, T]
        """
        # Residual branch
        residual = x if self.downsample is None else self.downsample(x)

        # Convolutional branch
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        # Residual sum and activation
        return self.relu_out(out + residual)


class TCNForecaster(nn.Module):
    """
    Temporal Convolutional Network for Multi-Step Workload Forecasting.

    Inputs:
        X: Tensor of shape [B, in_channels, input_length] (e.g. [B, F, 288])
    Outputs:
        y_hat: Tensor of shape [B, forecast_horizon] (e.g. [B, 12])

    Architectural Configuration:
        - in_channels: 1 (M1), 6 (M2), or 10 (M3)
        - input_length: 288 steps (24 hours at 5-minute sampling)
        - forecast_horizon: 12 steps (1 hour at 5-minute sampling)
        - hidden_channels: 32 (lightweight baseline default)
        - kernel_size: 3
        - dilations: [1, 2, 4, 8, 16, 32, 64, 128] (receptive field = 511 steps)
        - dropout: 0.1
    """

    def __init__(
        self,
        in_channels: int,
        input_length: int = DEFAULT_INPUT_LENGTH,
        forecast_horizon: int = DEFAULT_FORECAST_HORIZON,
        hidden_channels: int = DEFAULT_HIDDEN_CHANNELS,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        dilations: Optional[List[int]] = None,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.input_length = int(input_length)
        self.forecast_horizon = int(forecast_horizon)
        self.hidden_channels = int(hidden_channels)
        self.kernel_size = int(kernel_size)
        self.dilations = list(dilations) if dilations is not None else list(DEFAULT_DILATIONS)
        self.dropout = float(dropout)

        # Receptive field verification
        self.receptive_field = calculate_receptive_field(self.kernel_size, self.dilations)
        assert self.receptive_field >= self.input_length, (
            f"Receptive field ({self.receptive_field}) must cover input length ({self.input_length})"
        )

        # Build dilated residual backbone
        layers = []
        num_blocks = len(self.dilations)
        for i, d in enumerate(self.dilations):
            in_ch = self.in_channels if i == 0 else self.hidden_channels
            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=self.hidden_channels,
                    kernel_size=self.kernel_size,
                    dilation=d,
                    dropout=self.dropout,
                )
            )
        self.network = nn.Sequential(*layers)

        # Multi-step linear projection head
        self.head = nn.Linear(
            in_features=self.hidden_channels,
            out_features=self.forecast_horizon,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of TCN forecaster.

        Args:
            x: Input tensor of shape [B, in_channels, input_length]
        Returns:
            y_hat: Forecast vector of shape [B, forecast_horizon]
        """
        # Shape assertions
        assert x.dim() == 3, f"Expected 3D input [B, F, L], got shape {list(x.shape)}"
        assert x.size(1) == self.in_channels, (
            f"Expected in_channels={self.in_channels}, got {x.size(1)}"
        )

        # Pass through causal dilated backbone: output shape [B, hidden_channels, T]
        feat = self.network(x)

        # Extract temporal feature representation at cutoff step t (final position index -1)
        h_t = feat[:, :, -1]  # Shape: [B, hidden_channels]

        # Multi-step projection: shape [B, forecast_horizon]
        y_hat = self.head(h_t)
        return y_hat


def count_parameters(model: nn.Module) -> int:
    """Returns total trainable parameter count of the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# VERIFICATION SUITE & SMOKE TEST RUNNERS
# =============================================================================

def verify_causality(
    model: TCNForecaster,
    in_channels: int = 10,
    input_length: int = DEFAULT_INPUT_LENGTH,
    cutoff_step: int = 150,
) -> Dict[str, Any]:
    """
    Rigorously verifies temporal causality of the TCN architecture:
    1. Passes an initial input sequence X1 through the network.
    2. Modifies strictly future timesteps t > cutoff_step in X2.
    3. Verifies that all feature representations at past and present timesteps
       t <= cutoff_step remain 100% IDENTICAL (max absolute difference == 0.0).
    4. Demonstrates that future observations cannot leak backwards into past states.
    """
    model.eval()
    with torch.no_grad():
        torch.manual_seed(42)
        X1 = torch.randn(2, in_channels, input_length)

        # Forward pass on original sequence
        feat1 = model.network(X1)
        rep1_past = feat1[:, :, : cutoff_step + 1]

        # Create perturbed sequence with extreme future changes strictly for t > cutoff_step
        X2 = X1.clone()
        X2[:, :, cutoff_step + 1 :] += torch.randn_like(X2[:, :, cutoff_step + 1 :]) * 1000.0

        # Forward pass on future-perturbed sequence
        feat2 = model.network(X2)
        rep2_past = feat2[:, :, : cutoff_step + 1]

        # Verify difference is strictly zero
        diff = (rep1_past - rep2_past).abs().max().item()

    assert diff == 0.0, f"Causality leak detected! Future changes altered past representations: diff={diff}"
    return {
        "cutoff_step": cutoff_step,
        "max_past_difference": diff,
        "causality_status": "PASSED (Zero Future Leakage)",
    }


def run_gradient_smoke_test(
    model: TCNForecaster,
    in_channels: int = 10,
    batch_size: int = 4,
) -> Dict[str, Any]:
    """
    Executes a dummy forward and backward pass on synthetic random tensors
    to verify architectural gradient flow.

    Guarantees:
    - This is NOT model training.
    - Zero Bitbrains data is used.
    - Zero optimizer steps or weight updates are performed.
    - Verifies that all parameter gradients are present and finite.
    """
    model.train()
    torch.manual_seed(42)
    x_dummy = torch.randn(batch_size, in_channels, model.input_length)
    y_dummy = torch.randn(batch_size, model.forecast_horizon)

    # Zero existing gradients
    model.zero_grad()

    # Forward pass
    y_hat = model(x_dummy)
    loss = F.mse_loss(y_hat, y_dummy)

    # Backward pass
    loss.backward()

    # Gradient check across all parameters
    grad_checks = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Missing gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"
            grad_checks.append((name, tuple(param.grad.shape)))

    return {
        "loss_value": float(loss.item()),
        "gradients_checked": len(grad_checks),
        "gradient_status": "PASSED (All Gradients Present & Finite)",
    }


def run_model_smoke_test():
    """
    Main smoke test executing all Stage 5.2 verification checks:
    1. Receptive field derivation and mathematical proof.
    2. Forward pass across M1 (F=1), M2 (F=6), M3 (F=10).
    3. Output shapes [B, 12] and numerical validity (zero NaNs, zero Infs).
    4. Trainable parameter accounting.
    5. Rigorous empirical causality verification.
    6. Architecture backward gradient smoke test on synthetic dummy data.
    """
    print("=" * 80)
    print("STAGE 5.2: TCN MODEL ARCHITECTURE & CAUSALITY SMOKE TEST")
    print("=" * 80)

    # 1. Receptive field verification
    rf = calculate_receptive_field(DEFAULT_KERNEL_SIZE, DEFAULT_DILATIONS)
    print("\n[1/5] Receptive Field Calculation & Theoretical Coverage:")
    print(f"  - Kernel Size (k):           {DEFAULT_KERNEL_SIZE}")
    print(f"  - Dilation Schedule:         {DEFAULT_DILATIONS}")
    print(f"  - Sum of Dilations:          {sum(DEFAULT_DILATIONS)}")
    print(f"  - Formula:                   RF = 1 + (k - 1) * sum(dilations)")
    print(f"  - Computed Receptive Field:  RF = {rf} steps")
    print(f"  - Target History Length (L): L  = {DEFAULT_INPUT_LENGTH} steps (24.0 hours)")
    print(f"  - Receptive Coverage Ratio:  {rf / DEFAULT_INPUT_LENGTH:.2f}x")
    assert rf == 511, f"Expected RF=511, got {rf}"
    assert rf >= DEFAULT_INPUT_LENGTH, f"Receptive field {rf} < {DEFAULT_INPUT_LENGTH}"
    print("  [PASS] Receptive field RF=511 fully covers 288-step (24-hour) history.")

    # 2. Multi-mode forward pass and parameter counts
    print("\n[2/5] Multi-Mode Forward Verification (M1, M2, M3):")
    modes = [
        ("M1_UNIVARIATE", 1),
        ("M2_RESOURCE", 6),
        ("M3_FULL", 10),
    ]

    models: Dict[str, TCNForecaster] = {}
    for mode_name, f_channels in modes:
        model = TCNForecaster(
            in_channels=f_channels,
            input_length=DEFAULT_INPUT_LENGTH,
            forecast_horizon=DEFAULT_FORECAST_HORIZON,
            hidden_channels=DEFAULT_HIDDEN_CHANNELS,
            kernel_size=DEFAULT_KERNEL_SIZE,
            dilations=DEFAULT_DILATIONS,
            dropout=DEFAULT_DROPOUT,
        )
        models[mode_name] = model

        n_params = count_parameters(model)
        x_sample = torch.randn(4, f_channels, DEFAULT_INPUT_LENGTH)
        y_hat = model(x_sample)

        # Output shape verification
        assert y_hat.shape == (4, DEFAULT_FORECAST_HORIZON), (
            f"Shape mismatch in {mode_name}: expected (4, {DEFAULT_FORECAST_HORIZON}), got {y_hat.shape}"
        )
        assert not torch.isnan(y_hat).any(), f"NaN in output of {mode_name}"
        assert not torch.isinf(y_hat).any(), f"Inf in output of {mode_name}"

        print(f"  [PASS] {mode_name:<14} | F={f_channels:2d} | Parameters: {n_params:,} | "
              f"Input: {list(x_sample.shape)} -> Output: {list(y_hat.shape)} | NaNs/Infs: NONE")

    # 3. Residual block and temporal dimension invariance check
    print("\n[3/5] Temporal Invariance & Residual Dimension Check:")
    m3_model = models["M3_FULL"]
    test_x = torch.randn(2, 10, DEFAULT_INPUT_LENGTH)
    with torch.no_grad():
        backbone_out = m3_model.network(test_x)
    assert backbone_out.shape == (2, DEFAULT_HIDDEN_CHANNELS, DEFAULT_INPUT_LENGTH), (
        f"Backbone shape mismatch: {backbone_out.shape}"
    )
    print(f"  [PASS] Backbone preserved exact sequence length: [2, 10, 288] -> {list(backbone_out.shape)}")

    # 4. Rigorous Causality Verification
    print("\n[4/5] Rigorous Empirical Causality Verification:")
    causality_res = verify_causality(m3_model, in_channels=10, input_length=288, cutoff_step=150)
    print(f"  - Cutoff Timestep Evaluated:  t = {causality_res['cutoff_step']}")
    print(f"  - Future Perturbation Range:  t in [{causality_res['cutoff_step'] + 1} ... 287]")
    print(f"  - Max Past State Difference:  {causality_res['max_past_difference']:.8f}")
    print(f"  [PASS] Causality Status:      {causality_res['causality_status']}")

    # 5. Backward Pass Gradient Flow Check (Synthetic Data Only)
    print("\n[5/5] Backward Pass Architecture Gradient Smoke Test (Zero Training Data):")
    grad_res = run_gradient_smoke_test(m3_model, in_channels=10, batch_size=4)
    print(f"  - Parameter Tensors Checked:  {grad_res['gradients_checked']}")
    print(f"  - Dummy Synthetic Loss (MSE): {grad_res['loss_value']:.4f}")
    print(f"  [PASS] Gradient Flow Status:  {grad_res['gradient_status']}")

    print("\n" + "=" * 80)
    print("STAGE 5.2 TCN MODEL ARCHITECTURE SMOKE TEST COMPLETED SUCCESSFULLY")
    print("ALL ARCHITECTURAL & CAUSALITY INVARIANTS RIGOROUSLY VERIFIED")
    print("NO MODEL TRAINING PERFORMED | RAW DATASETS UNTOUCHED")
    print("=" * 80)


if __name__ == "__main__":
    run_model_smoke_test()

