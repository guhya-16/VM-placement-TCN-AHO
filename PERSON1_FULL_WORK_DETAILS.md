# Person 1 Full Technical Work Details & Methodology Record
**Project Title**: *Predictive Energy-Efficient VM Placement in Cloud Using ML and Adaptive Hippopotamus Optimization*  
**Role**: Person 1 — Telemetry Preprocessing, Deep Learning Workload Forecasting & Workload Intelligence  
**Downstream Consumers**: Person 3 (Optimization / AHO) & Person 2 (Simulation / CloudSim Plus)  
**Status**: **COMPLETE / FROZEN / HANDED OFF** (Branch: `person1-ml-risk-handoff`)

---

## 1. Project Overview

### 1.1 Overarching Research Context
In modern virtualized cloud data centers, virtual machine (VM) consolidation onto physical machines (PMs) is a primary mechanism for reducing idle server energy waste. However, aggressive server consolidation risks service-level agreement (SLA) violations if colocated VMs simultaneously experience sudden resource demand spikes. Static or reactive heuristic placement schemes fail to anticipate such bursts, leading to reactive VM live migrations, host thrashing, and SLA degradation.

This research project resolves this challenge by introducing a multi-step predictive consolidation architecture that combines deep sequence modeling with nature-inspired metaheuristic optimization:

```text
====================================================================================================
                                      END-TO-END PROJECT PIPELINE
====================================================================================================
Bitbrains Telemetry (GWA-T-12 fastStorage)
    │
    ▼
Stage 1-3: Forensic Ingestion, Trace Profiling & Canonical 5-Minute Alignment (Policy C)
    │
    ▼
Stage 4: Chronological Splitting (70/15/15) & Leakage-Safe Sliding Windows (L=288, H=12)
    │
    ▼
Stage 5: Dilated Causal Temporal Convolutional Network (TCNForecaster, F=10, RF=511, Params=48,300)
    │
    ▼
Phase 9: Unbiased Test Evaluation & 82,980 Multi-Step Forecasts (predictions.csv)
    │
    ▼
Phase 10-11: Causal Volatility (K=24) + Trajectory Spread -> Frozen Risk Proxy (risk_state.csv)
    │
    ▼  [OFFICIAL HANDOFF BOUNDARY: person1-ml-risk-handoff branch]
Person 3: Adaptive Hippopotamus Optimization (AHO) -> VM-to-PM Placement Engine (placement.csv)
    │
    ▼
Person 2: CloudSim Plus Simulation -> Energy (kWh), SLA Violations, PM Utilization & Migrations
====================================================================================================
```

### 1.2 Team Responsibility Division
The 3-person research project enforces strict methodological separation of concerns:
* **Person 1 (PREDICT — Our Role)**: Owns the raw dataset ingestion, exploratory forensic data analysis, Policy C canonical grid alignment, split-safe sliding window construction, TCN architecture design, 12-configuration ablation, final test evaluation, causal workload volatility modeling, validation-frozen prediction-risk scoring, and the master placement-facing snapshot table (`risk_state.csv`).
* **Person 3 (OPTIMIZE)**: Owns Standard Hippopotamus Optimization (HO), Adaptive Hippopotamus Optimization (AHO), chromosome/hippo candidate encoding, fitness function formulation, host capacity constraint enforcement, dynamic parameter adaptation driven by `risk_state.csv`, and generation of `placement.csv`.
* **Person 2 (SIMULATE)**: Owns the CloudSim Plus simulation environment, datacenter infrastructure modeling, physical host power curves, live migration execution, energy consumption ($kWh$) accounting, and SLA violation metrics.

> **Crucial Boundary Invariant**: Person 1 does **NOT** implement Adaptive HO, metaheuristic exploration/exploitation equations, host placement logic, or CloudSim Plus simulations. Person 1's work terminates at the data/workload intelligence handoff interface.

---

## 2. Person 1 Responsibility Boundary

| Phase / Component | Inside Person 1 Responsibility? | Outside Person 1 Responsibility? |
| :--- | :---: | :---: |
| Bitbrains GWA-T-12 Ingestion & Inspection | **YES** | No |
| Anomaly & Interleaved Telemetry Forensics | **YES** | No |
| Policy C 5-Minute Canonical Alignment | **YES** | No |
| Chronological 70/15/15 Splitting & Sliding Windows | **YES** | No |
| Feature Scaling (`StandardScalerWrapper` Train-Only) | **YES** | No |
| Causal TCN Architecture (`TCNForecaster`, 48,300 params) | **YES** | No |
| Controlled 12-Run GPU Ablation Experimentation | **YES** | No |
| Final Test Benchmark Evaluation (`predictions.csv`) | **YES** | No |
| Causal Volatility ($K=24$, $ddof=1$) & Spread Modeling | **YES** | No |
| Frozen Validation Risk Calibration (`risk_calibration.json`) | **YES** | No |
| Placement Input Table (`risk_state.csv`, 6,915 rows) | **YES** | No |
| Git Branch Handoff (`person1-ml-risk-handoff`) | **YES** | No |
| Standard / Adaptive Hippopotamus Optimization (HO/AHO) | No | **YES (Person 3)** |
| Host Power Curve & Energy Fitness Formulation | No | **YES (Person 3)** |
| PM Capacity Margins & Colocation Policy | No | **YES (Person 3)** |
| Output `placement.csv` Allocation Generation | No | **YES (Person 3)** |
| CloudSim Plus Discrete Event Simulation | No | **YES (Person 2)** |
| SLA Penalty & Host Power Consumption Simulation | No | **YES (Person 2)** |

---

## 3. Dataset Invariants & Forensic Characteristics

### 3.1 Bitbrains GWA-T-12 Benchmark Reference
* **Dataset Identifier**: Bitbrains Grid Workloads Archive trace GWA-T-12 (Shen et al., 2015).
* **Target Environment**: `fastStorage` cohort (enterprise distributed storage hosting hosting performance-critical compute and disk workloads).
* **Location in Workspace**: `dataset/fastStorage/2013-8/` (1,250 CSV files, one file per VM, labeled `1.csv` to `1250.csv`).
* **Git Status**: Kept strictly **read-only**; never modified, overwritten, or committed to Git (13+ GB).

### 3.2 Dataset File Structure & Telemetry Schema
Each VM trace file contains comma/semicolon-delimited records formatted as follows:
* **Raw Header**: `"Timestamp [ms];\tCPU cores;\tCPU capacity provisioned [MHZ];\tCPU usage [MHZ];\tCPU usage [%];\tMemory capacity provisioned [KB];\tMemory usage [KB];\tDisk read throughput [KB/s];\tDisk write throughput [KB/s];\tNetwork received throughput [KB/s];\tNetwork transmitted throughput [KB/s]"`
* **Observed Delimiter**: Semicolons followed by tabs (`;\t`).
* **Observed Timestamp Unit**: **Unix Epoch Seconds** (integer, e.g., `1376314846`). While the header indicates `Timestamp [ms]`, empirical verification confirmed the values are in seconds. (Interpreting them as milliseconds yields timestamps in January 1970).
* **Sampling Rate**: 300 seconds nominal ($5$ minutes).

### 3.3 Official Documentation vs. Forensic Empirical Findings

| Characteristic | Official Documentation Claim | Forensic Empirical Reality (Stages 1–2) |
| :--- | :--- | :--- |
| **Sampling Interval** | Uniform, regular 5-minute sampling | Highly irregular in subset: jitter ($\pm 10$s), isolated missing steps, and duplicate timestamps |
| **VM Lifetime** | 30 full continuous days (August 2013) | Subset of VMs de-provisioned mid-month; multiple VMs active $< 2$ days |
| **Telemetry State** | Continuous active VM execution | Post-decommission telemetry trails: CPU capacity dropped to 0, residual idle daemon logs |
| **Record Uniqueness** | Exactly one row per sampling step | Several VM files contain identical duplicate timestamps and interleaved sequence blocks |

---

## 4. Stage 1: Dataset Ingestion & Systematic Inspection

Stage 1 established automated data loading and integrity auditing tools ([`ml/data_loader.py`](file:///d:/SEM%207/VMplacement/ml/data_loader.py), [`ml/inspect_dataset.py`](file:///d:/SEM%207/VMplacement/ml/inspect_dataset.py)).

### 4.1 Systematic Inspection Protocol
1. **Schema Consistency**: Verified all 1,250 CSV files share identical 11-column definitions.
2. **Missing Values**: Raw CSV files have zero `NaN`, `null`, or unparseable float tokens.
3. **Monotonicity & Continuity**: Audited timestamp deltas $\Delta t = t_{i} - t_{i-1}$:
   - Identified two distinct behavioral cohorts: **Type A** (strictly monotonic, clean) and **Type B** (non-monotonic, containing duplicate timestamps and multi-day sequence anomalies).
4. **Representative Cohort Selection**: Selected **14 representative VMs** encompassing all operational profiles:
   - **Type A Clean Long-Running (7 VMs)**: `VM_001` (bursty), `VM_003` (steady), `VM_011` (cyclic spikes), `VM_012` (low steady), `VM_013` (idle provisioned), `VM_023` (sustained moderate), `VM_025` (sustained heavy).
   - **Type B De-provisioned (5 VMs)**: `VM_002`, `VM_004`, `VM_006`, `VM_014`, `VM_018` (de-provisioned on August 29, 2013; contain duplicate timestamps and zero-capacity tails).
   - **Transient Ephemeral (2 VMs)**: `VM_161` (active 2.2 days), `VM_1209` (sub-day ephemeral, active 9.7 hours).

---

## 5. Stage 2: Forensic Dataset Analysis

Stage 2 executed a deep investigation into the root cause of Type B anomalies ([`ml/forensic_analysis.py`](file:///d:/SEM%207/VMplacement/ml/forensic_analysis.py)).

### 5.1 Analysis of Duplicate & Interleaved Timestamps
* In files such as `2.csv` and `4.csv`, thousands of records shared identical timestamps.
* Forensic analysis showed that rows sharing identical timestamps had identical hardware capacity columns but diverging or zeroed telemetry.
* Crucially, the duplicates were concentrated after a clear lifecycle transition event on **August 29, 2013**:
  - `CPU capacity provisioned [MHZ]` collapsed to `0.0`.
  - `Memory capacity provisioned [KB]` collapsed to `0.0` or dropped to minimal static levels.
  - Actual CPU usage fell to `0.0%`.

### 5.2 Scientific Interpretation & Hypothesis Rejection
* **Rejected Speculative Claims**: We explicitly rejected unverified narratives such as "ghost logging daemons" or "corrupted hypervisors."
* **Scientifically Sound Reality**: In enterprise virtualization platforms (e.g., VMware vCenter), when a VM is decommissioned or unallocated, the hypervisor's monitoring agent can continue polling and logging records for an unprovisioned object until the monitoring task is deregistered.
* **Methodological Conclusion**: Telemetry collected when a VM has zero allocated capacity (`capacity == 0`) is **inactive de-provisioned data**, not genuine workload demand. Including it would distort workload forecasting and train models to predict decommission artifacts.

---

## 6. Stage 3: Final Preprocessing Policy (Policy C)

Stage 3 evaluated three competing preprocessing policies (Policy A: naive deduplication; Policy B: capacity filtering without alignment; Policy C: canonical timeline construction with constrained interpolation).

### 6.1 Policy C Specification (9 Invariants)
1. **Read-Only Ingestion**: Raw CSV traces are loaded without in-place modification.
2. **Active State Definition**: A VM observation is valid if and only if:
   $$\text{CPU capacity provisioned [MHZ]} > 0 \quad \text{AND} \quad \text{Memory capacity provisioned [KB]} > 0$$
3. **Prefix/Suffix Truncation**: Unallocated leading records and post-decommission trailing records are stripped.
4. **Canonical 5-Minute Grid**: For each VM, construct a rigid mathematical reference grid:
   $$t_k = t_0 + k \times 300 \quad (k = 0, 1, \dots, N-1)$$
5. **Exact Temporal Binning**: Raw observations are mapped to the nearest 300-second grid point (tolerance $\pm 150$s).
6. **Constrained Gap Interpolation**: Linear interpolation is permitted **ONLY for isolated micro-gaps of $\le 2$ consecutive missing steps** ($\le 10$ minutes).
7. **Sequence Break Preservation**: Any gap $> 2$ consecutive missing steps ($> 10$ minutes) is treated as an unbridgeable sequence break. No synthetic data is fabricated across major outages.
8. **Deduplication Safeguard**: Where multiple valid active observations map to the same bin, the primary record is selected deterministically without distortion.
9. **Raw Immutability**: All transformed outputs are serialized to interim Parquet tables (`data/interim/stage3_canonical_5min_clean.parquet`); raw Bitbrains files remain byte-intact.

### 6.2 Empirical Stage 3 Verification Statistics
* **14 Representative VMs Aligned**: Yielded exactly **`63,353`** canonical 5-minute steps.
* **Synthetic Interpolation Count**: Only **122** missing steps were interpolated across all 63,353 steps (**0.19%** interpolation rate), demonstrating that $>99.8\%$ of aligned data consists of genuine empirical measurements.
* **Reference Windows Formed**: **`59,617`** potential sequence windows of length $W=300$.
* **Short-Lived VM Finding**: 7 VMs (`VM_002`, `VM_004`, `VM_006`, `VM_014`, `VM_018`, `VM_161`, `VM_1209`) have total active lifespans shorter than the required 25-hour window span ($W = L + H = 300$ steps) or contain $< 300$ steps in their test splits, yielding zero valid test sliding windows without artificial data fabrication.

---

## 7. Stage 4: Temporal Windows & Leakage-Safe Splitting

Stage 4 established the dataset partitioning and virtual window construction engine ([`ml/temporal_windows.py`](file:///d:/SEM%207/VMplacement/ml/temporal_windows.py)).

### 7.1 Window Dimensions & Temporal Parameters
* **Sampling Step**: $\Delta t = 300\text{s}$ (5 minutes).
* **Lookback History ($L$)**: $288$ steps ($24.0$ hours) — captures full diurnal cycle.
* **Forecast Horizon ($H$)**: $12$ steps ($1.0$ hour) — supports medium-term proactive placement.
* **Total Window Length ($W$)**: $W = L + H = 300$ steps ($25.0$ hours).

### 7.2 Chronological 70 / 15 / 15 Partitioning & Leakage Barrier
To prevent data leakage, each VM trace is partitioned **chronologically**:
$$\text{Train} = \text{First } 70\% \quad | \quad \text{Validation} = \text{Next } 15\% \quad | \quad \text{Test} = \text{Final } 15\%$$

* **Split Containment Rule**: Both history $X$ ($t-287 \dots t$) and target horizon $y$ ($t+1 \dots t+12$) must reside **entirely** within the same split partition. Any window that crosses a train/val or val/test boundary is omitted.
* **No Cross-VM Straddling**: Windows are strictly confined to a single VM identifier.
* **Step Accounting**:
  - Total Aligned Steps: **`63,353`**
  - Train Steps: **`44,343`** ($70.00\%$)
  - Validation Steps: **`9,498`** ($14.99\%$)
  - Test Steps: **`9,512`** ($15.01\%$)
* **Sliding Window Accounting**:
  - Reference Candidate Windows: `59,617`
  - Boundary Omissions: `5,006` (windows straddling train/val/test splits)
  - Gap Omissions: `1,087` (windows straddling sequence breaks $>10$m)
  - **Final Valid Sliding Windows**: **`53,524`**
    * **Train Windows**: **`40,052`**
    * **Validation Windows**: **`6,557`**
    * **Test Windows**: **`6,915`**

### 7.3 Train-Only Scaler Fitting
Feature scalers are fitted **exclusively on the 44,343 unique train steps**. Zero validation or test observations are ever seen during `scaler.fit()`.

---

## 8. Stage 5: TCN Architecture Design

Stage 5 developed the deep sequence forecasting model ([`ml/tcn_model.py`](file:///d:/SEM%207/VMplacement/ml/tcn_model.py)).

### 8.1 Input Feature Configurations
We formulated three nested feature sets to assess the value of cross-resource and temporal context:
* **`M1_UNIVARIATE` ($F=1$)**: `cpu_usage_percent` only.
* **`M2_RESOURCE` ($F=6$)**: `cpu_usage_percent`, `memory_usage_kb`, `disk_read_kbps`, `disk_write_kbps`, `network_received_kbps`, `network_transmitted_kbps`.
* **`M3_FULL` ($F=10$)**: M2 features $+$ 4 cyclical calendar embeddings (`hour_sin`, `hour_cos`, `day_sin`, `day_cos`).

### 8.2 Architectural Specifications
* **Network Topology**: Dilated 1D Causal Convolutional Network (`TCNForecaster`).
* **Kernel Size ($k$)**: $3$
* **Hidden Channels**: $32$ per layer.
* **Dilation Schedule**: Exponential base-2: $d \in [1, 2, 4, 8, 16, 32, 64, 128]$ across 8 residual blocks.
* **Receptive Field ($RF$)**:
  $$RF = 1 + 2 \times (k - 1) \times \sum_{i=0}^{7} d_i = 1 + 2 \times 2 \times 255 = 1 + 1020 = \mathbf{511 \text{ steps}}$$
  Because $RF = 511 \ge L = 288$ steps ($24$ hours), the network can model long-range temporal dependencies across the full lookback horizon.
* **Regularization**: Spatial dropout ($p=0.10$).
* **Linear Forecasting Head**: `nn.Linear(32, 12)` projecting the final temporal representation $h_{L-1}$ to the 12 forecast steps.
* **Tensor Shapes**: Input $[B, F, 288] \to$ Output $[B, 12]$.

### 8.3 Exact Parameter Count Resolution (48,300 vs. 48,876)
Introspection confirms the model has exactly **`48,300`** trainable parameters:
* **Block 0** ($F=10 \to 32$): $960 + 32 + 3072 + 32 + 320 + 32 = 4,448$ params.
* **Blocks 1–7** (7 blocks $\times 6,208$): $43,456$ params.
* **Linear Head** ($32 \to 12$): $384 + 12 = 396$ params.
* **Total**: $4,448 + 43,456 + 396 = \mathbf{48,300}$.
* *Origin of 48,876*: In literature architectures utilizing `torch.nn.utils.weight_norm` across all 18 layers (17 convs + 1 linear head), an additional $18 \times 32 = 576$ scalar magnitude parameters exist ($48,300 + 576 = 48,876$). Our implementation uses modern standard PyTorch `nn.Conv1d` without deprecated `weight_norm`, giving exactly **48,300**.

---

## 9. Stage 5.1: Dataset Implementation (`ml/tcn_dataset.py`)

* **Virtual Windowing**: Rather than materializing $53,524 \times 300 \times 10$ floating point numbers in RAM (~640 MB redundant tensors), the dataset stores pre-extracted 2D telemetry matrices per VM and dynamically slices $[F, 288]$ and $[12]$ tensors using a prevalidated index table.
* **Decoupled Scalers**:
  - `StandardScalerWrapper`: Zero-mean, unit-variance standardization.
  - `RobustScalerWrapper`: Median-centered, IQR-scaled transformation.
  - `Log1pStandardScalerWrapper`: Natural log $\ln(1+x)$ applied strictly to non-negative resource telemetry ($[0, \infty)$), treating cyclical calendar features ($[-1, 1]$) as unscaled pass-through invariants.
  - `NativeScaler`: Identity pass-through.
* **Target Scaler Independence**: The model predicts native CPU% directly (`target_strategy="native"`), eliminating inverse-transform error compounding.
* **Integrity Guard**: Built-in assertions verify zero `NaN` or `Inf` values in every batch.

---

## 10. Stage 5.2: TCN Model Implementation (`ml/tcn_model.py`)

* **Causal Padding**: `Chomp1d(padding)` strips right-side padding after standard convolution, ensuring calculation at time $t$ uses only inputs $\le t$.
* **TemporalBlock Structure**: Each residual block contains:
  $$\text{Conv1D}_1 \to \text{Chomp1d} \to \text{ReLU} \to \text{Dropout} \to \text{Conv1D}_2 \to \text{Chomp1d} \to \text{ReLU} \to \text{Dropout} + \text{Residual}$$
* **Channel Projection**: When input and output channels mismatch (Block 0: $F \to 32$), a $1 \times 1$ convolution (`downsample`) projects the skip connection.
* **Empirical Causality Check**: Automated test verifies that perturbing input timestep $t_k$ produces zero gradient or activation changes at any output $\le t_{k-1}$.

---

## 11. Stage 5.3: Training Protocol (`ml/tcn_train.py`)

* **Objective Function**: Mean Squared Error (MSE) on native CPU% scale.
* **Optimizer**: AdamW ($\text{learning rate} = 10^{-3}$, $\text{weight decay} = 10^{-4}$).
* **Batch Size**: 64 (shuffle enabled on training set only).
* **Gradient Clipping**: Maximum norm capped at $1.0$ (`clip_grad_norm_`).
* **Learning Rate Scheduler**: `ReduceLROnPlateau` (metric: validation loss, factor $=0.5$, patience $=2$ epochs, minimum LR $=10^{-5}$).
* **Early Stopping**: Patience of 5 epochs monitoring validation loss.
* **Maximum Epochs**: 50.
* **Strict Model Selection Barrier**: All model selection decisions were governed strictly by validation loss. The test split remained completely untouched.

---

## 12. Stage 5.3: 12-Configuration Controlled GPU Ablation Study

We completed the full $3 \times 4$ factorial ablation study on Google Colab GPU (NVIDIA Tesla T4).

### Complete Verified Ablation Ranking Table
*(Ranked by Best Validation Loss, native MSE scale)*

| Rank | Experiment ID | Features ($F$) | Scaling Strategy | Best Epoch | Total Epochs | Best Val Loss (MSE) | Val MAE (% CPU) | Val RMSE (% CPU) | Val $R^2$ | Val sMAPE (%) | Val MAPE ($\ge 1\%$) | Train Time |
| :---: | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **1** | **`M3_FULL_standard`** | **10** | **Standard** | **9** | **14** | **`69.368351`** | **1.5304** | **8.3435** | **0.2589** | **36.31%** | **14.85%** | **173.2s** |
| 🥈 2 | `M2_RESOURCE_standard` | 6 | Standard | 7 | 12 | 70.100062 | 1.5217 | 8.3936 | 0.2499 | 42.18% | 13.94% | 149.6s |
| 🥉 3 | `M1_UNIVARIATE_robust` | 1 | Robust | 12 | 14 | 72.133335 | 1.4833 | 8.5105 | 0.2289 | 36.64% | 15.67% | 164.1s |
| 4 | `M1_UNIVARIATE_log1p_standard` | 1 | Log1p+Standard | 19 | 24 | 72.600252 | 1.4415 | 8.5412 | 0.2233 | 34.69% | 12.75% | 288.8s |
| 5 | `M3_FULL_log1p_standard` | 10 | Log1p+Standard | 12 | 17 | 72.651733 | 1.4989 | 8.5422 | 0.2231 | 34.32% | 14.28% | 209.8s |
| 6 | `M1_UNIVARIATE_standard` | 1 | Standard | 11 | 16 | 72.816329 | 1.4612 | 8.5517 | 0.2214 | 29.31% | 13.71% | 190.2s |
| 7 | `M2_RESOURCE_log1p_standard` | 6 | Log1p+Standard | 5 | 10 | 73.819563 | 1.6329 | 8.6126 | 0.2103 | 40.34% | 16.01% | 125.9s |
| 8 | `M1_UNIVARIATE_native` | 1 | Native | 6 | 6 | 75.504162 | 1.5312 | 8.7004 | 0.1941 | 28.58% | 14.47% | 8000.6s |
| 9 | `M2_RESOURCE_robust` | 6 | Robust | 1 | 6 | 81.264189 | 1.8372 | 8.9318 | 0.1507 | 50.33% | 22.51% | 76.7s |
| 10 | `M3_FULL_robust` | 10 | Robust | 1 | 6 | 94.730013 | 1.8316 | 9.3509 | 0.0691 | 48.25% | 20.15% | 73.8s |
| 11 | `M2_RESOURCE_native` | 6 | Native | 50 | 50 | 109.265534 | 2.4720 | 9.8132 | -0.0252 | 65.80% | 44.60% | 606.0s |
| 12 | `M3_FULL_native` | 10 | Native | 22 | 27 | 735.187960 | 4.4173 | 11.5606 | -0.4229 | 152.41% | 124.62% | 333.4s |

### Champion Model Selection Rationale
* **`M3_FULL_standard`** achieved the lowest validation loss (**69.3684**), lowest validation RMSE (**8.3435% CPU**), and highest validation $R^2$ (**0.2589**).
* Multi-modal features ($F=10$) provided temporal context that outperformed univariate models.
* Standard Gaussian scaling stabilized multi-scale telemetry (KB vs. throughput vs. percentages).
* The winning checkpoint was locked as:
  [`results/stage5/checkpoints/M3_FULL_standard_best.pt`](file:///d:/SEM%207/VMplacement/results/stage5/checkpoints/M3_FULL_standard_best.pt)

---

## 13. Phase 9: Final Test Benchmark Evaluation

The locked champion model was evaluated **strictly once** on the untouched test partition ([`ml/tcn_evaluate.py`](file:///d:/SEM%207/VMplacement/ml/tcn_evaluate.py)).

### 13.1 Overall Benchmark Performance Metrics
* **Total Test Sliding Windows**: **`6,915`**
* **Total Forecast Points Evaluated**: **`82,980`** ($6,915 \times 12$ steps)
* **Mean Absolute Error (MAE)**: **`1.7179% CPU`**
* **Root Mean Squared Error (RMSE)**: **`9.5368% CPU`**
* **Coefficient of Determination ($R^2$)**: **`0.2213`**
* **Symmetric MAPE (sMAPE)**: **`44.7842%`**
* **Thresholded MAPE ($\ge 1.0\%$ CPU)**: **`15.4665%`** (evaluated on 60,659 eligible observations, 73.10% of test points)
* **Missing / NaN / Infinite Values**: **0** across all predictions and ground truth.

### 13.2 Accuracy Degradation Across Forecast Horizon

| Step | Horizon Offset | Lead Time | MAE (% CPU) | RMSE (% CPU) | $R^2$ | sMAPE (%) | Thresholded MAPE ($\ge 1\%$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | $+1$ step | $+5$ min | **1.3667** | **7.4828** | **0.5206** | 42.78% | 14.25% |
| 2 | $+2$ steps | $+10$ min | 1.4732 | 8.2230 | 0.4211 | 43.17% | 14.46% |
| 3 | $+3$ steps | $+15$ min | 1.5817 | 8.8431 | 0.3305 | 43.63% | 15.33% |
| 4 | $+4$ steps | $+20$ min | 1.6000 | 8.9566 | 0.3132 | 43.13% | 14.88% |
| 5 | $+5$ steps | $+25$ min | 1.6730 | 9.1401 | 0.2847 | 45.08% | 15.27% |
| 6 | $+6$ steps | $+30$ min | 1.7051 | 9.3878 | 0.2454 | 44.93% | 15.33% |
| 7 | $+7$ steps | $+35$ min | 1.7589 | 9.6844 | 0.1970 | 45.41% | 15.71% |
| 8 | $+8$ steps | $+40$ min | 1.7915 | 9.8787 | 0.1646 | 45.24% | 15.54% |
| 9 | $+9$ steps | $+45$ min | 1.8392 | 10.1587 | 0.1166 | 45.89% | 15.93% |
| 10 | $+10$ steps | $+50$ min | 1.8841 | 10.4283 | 0.0691 | 45.71% | 15.99% |
| 11 | $+11$ steps | $+55$ min | 1.9126 | 10.5973 | 0.0387 | 46.25% | 16.15% |
| 12 | $+12$ steps | $+60$ min | 1.9287 | 10.7570 | 0.0096 | 45.79% | 16.08% |

Accuracy degrades gracefully and monotonically as lead time increases. Near-term accuracy at $+5$ minutes achieves $R^2 = 0.5206$ and $\text{MAE} = 1.3667\%$.

### 13.3 Diagnostic Visualizations
* **Trajectory Tracking**: [`stage5_actual_vs_predicted_representative_vms.png`](file:///d:/SEM%207/VMplacement/results/stage5/figures/stage5_actual_vs_predicted_representative_vms.png) demonstrates high tracking fidelity across bursty (`VM_001`), steady (`VM_012`), and sustained cyclic (`VM_025`) workloads.
* **Error Distribution**: [`stage5_test_prediction_error_distribution.png`](file:///d:/SEM%207/VMplacement/results/stage5/figures/stage5_test_prediction_error_distribution.png) shows errors tightly centered around zero (median $-0.0025\%$ CPU, IQR $[-0.16\%, +0.14\%]$).
* **Degradation Curve**: [`stage5_test_accuracy_degradation_vs_horizon.png`](file:///d:/SEM%207/VMplacement/results/stage5/figures/stage5_test_accuracy_degradation_vs_horizon.png) displays error scaling across the 12 lead-time steps.

---

## 14. Phase 10: ML $\to$ Adaptive HO Interface Design

Phase 10 transitioned Person 1's role from point forecasting to **workload intelligence** ([`docs/phase10_ml_to_aho_interface_design.md`](file:///d:/SEM%207/VMplacement/docs/phase10_ml_to_aho_interface_design.md)).

### 14.1 Motivation: Predictive Consolidation vs. Static Heuristics
Providing an optimizer with a single point forecast is insufficient. A high average workload with low volatility warrants aggressive PM colocation; an identical average workload characterized by extreme turbulence requires safety headroom to avoid SLA violations.

### 14.2 Mathematical Components of Workload Intelligence
1. **Decision Cutoff ($t$)**: Telemetry is available strictly for $x \le t$.
2. **Causal Workload Volatility ($V(t)$)**: Sample standard deviation ($ddof=1$) of the preceding $K=24$ observations ($2.0$ hours) up to time $t$:
   $$V(t) = \sqrt{\frac{1}{K-1} \sum_{i=0}^{K-1} (x_{t-i} - \bar{x})^2} \quad (K=24)$$
3. **Forecast Trajectory Spread ($S(t)$)**: Range of predicted demand across the upcoming 1-hour horizon:
   $$S(t) = \max_{h \in [1..12]} \hat{y}_{t+h} - \min_{h \in [1..12]} \hat{y}_{t+h}$$
4. **Validation-Frozen Normalization**: Normalized against 95th percentiles calibrated exclusively on validation sliding windows:
   $$v_{\text{norm}}(t) = \min\left(1.0, \frac{V(t)}{V_{\max}}\right), \quad s_{\text{norm}}(t) = \min\left(1.0, \frac{S(t)}{S_{\max}}\right)$$
5. **Composite Workload-Risk Score**:
   $$\text{RiskScore}(t) = 0.60 \cdot v_{\text{norm}}(t) + 0.40 \cdot s_{\text{norm}}(t) \in [0.0, 1.0]$$
6. **Categorical Risk States**:
   $$\text{risk\_state}(t) = \begin{cases} \text{LOW} & \text{if } \text{RiskScore}(t) < \tau_{\text{low}} \\ \text{MEDIUM} & \text{if } \tau_{\text{low}} \le \text{RiskScore}(t) < \tau_{\text{high}} \\ \text{HIGH} & \text{if } \text{RiskScore}(t) \ge \tau_{\text{high}} \end{cases}$$

> **Important Conceptual Invariant**: $\text{RiskScore}$ is a **heuristic workload-risk proxy** reflecting physical turbulence and projected trajectory spread. It is **NOT** a Bayesian posterior variance or probabilistic uncertainty interval.

### 14.3 Empirical Validation Audit
Audit on the 6,557 validation sliding windows proved strong correlation with forecast difficulty:
* Causal Volatility ($K=24$) vs. Validation MAE: $r = \mathbf{+0.5716}$
* Forecast Trajectory Spread vs. Validation MAE: $r = \mathbf{+0.4882}$
* Volatility vs. Forecast Spread: $r = \mathbf{+0.5389}$

---

## 15. Phase 11: Workload Intelligence Implementation & Pre-Git Audit

Stage 5.5 / Phase 11 implemented the full pipeline in [`ml/tcn_risk.py`](file:///d:/SEM%207/VMplacement/ml/tcn_risk.py).

### 15.1 Frozen Validation Calibration Constants
Calibrated exclusively on 6,557 validation windows and exported to [`results/stage5/risk/risk_calibration.json`](file:///d:/SEM%207/VMplacement/results/stage5/risk/risk_calibration.json):
* $V_{\max}$ (95th percentile validation volatility): **`4.675352% CPU`**
* $S_{\max}$ (95th percentile validation forecast spread): **`3.382352% CPU`**
* $\tau_{\text{low}}$ (33.3rd percentile validation RiskScore): **`0.029545`**
* $\tau_{\text{high}}$ (66.7th percentile validation RiskScore): **`0.174569`**

### 15.2 Placement Input Artifact (`risk_state.csv`)
* **Total Rows**: Exactly **`6,915`**
* **Columns**: Exactly **15 approved columns in order**:
  `vm_id`, `decision_timestamp`, `current_cpu`, `predicted_cpu_t5m`, `predicted_cpu_t10m`, `predicted_cpu_t15m`, `predicted_cpu_t30m`, `predicted_cpu_t45m`, `predicted_cpu_t60m`, `predicted_mean_cpu`, `predicted_peak_cpu`, `predicted_std_cpu`, `volatility_score`, `risk_score`, `risk_state`
* **Zero-Leakage Compliance**: Zero future actual CPU values, residuals, absolute errors, or squared errors exist in the table.
* **Option B Physical Clipping**: All placement-facing predicted CPU values in `risk_state.csv` are clipped to $[0.0, 100.0]\%$ (`predicted_mean_cpu min = +0.0060%`, 0 negative values). Meanwhile, scientific evaluation table `predictions.csv` preserves the raw unclipped benchmark.

### 15.3 Empirical Test Risk Distribution
Applying the frozen validation thresholds to the 6,915 test instances yields:
* **`LOW`**: **`2,060`** instances ($29.79\%$)
* **`MEDIUM`**: **`2,464`** instances ($35.63\%$)
* **`HIGH`**: **`2,391`** instances ($34.58\%$)

*(Note: These represent the empirical test outcome; equal test tertiles are not guaranteed or claimed).*

### 15.4 Automated Acceptance Testing Suite (17 / 17 Passed)
All 17 regression checks in [`results/stage5/risk/risk_verification.json`](file:///d:/SEM%207/VMplacement/results/stage5/risk/risk_verification.json) passed:
1. File exists and non-empty `[PASS]`
2. Exactly 6,915 rows `[PASS]`
3. Exactly 15 approved columns `[PASS]`
4. Zero NaN values `[PASS]`
5. Zero Inf values `[PASS]`
6. Exactly 7 active VMs represented `[PASS]`
7. 7 short-lived VMs excluded without fabrication `[PASS]`
8. Exactly 997 decision timestamps `[PASS]`
9. Zero future actuals present `[PASS]`
10. All 6 horizon steps present and finite `[PASS]`
11. `predicted_mean_cpu` matches arithmetic mean `[PASS]`
12. `predicted_peak_cpu` matches maximum forecast `[PASS]`
13. `risk_score` bounded in $[0, 1]$ `[PASS]`
14. `risk_state` in `{'LOW', 'MEDIUM', 'HIGH'}` `[PASS]`
15. Calibration JSON exists and valid `[PASS]`
16. README guide exists with boundaries `[PASS]`
17. Overall verification status is `PASS` `[PASS]`

---

## 16. Final Person 1 Artifact Inventory

| Artifact | Repository Path | Size | Primary Purpose | Consumer |
| :--- | :--- | :---: | :--- | :--- |
| **Window Pipeline** | [`ml/temporal_windows.py`](file:///d:/SEM%207/VMplacement/ml/temporal_windows.py) | 22.8 KB | Stage 4 windowing & calendar features | Pipeline / Person 1 |
| **Model Topology** | [`ml/tcn_model.py`](file:///d:/SEM%207/VMplacement/ml/tcn_model.py) | 16.9 KB | Causal dilated 1D TCN definition | PyTorch / Person 1 |
| **Dataset Engine** | [`ml/tcn_dataset.py`](file:///d:/SEM%207/VMplacement/ml/tcn_dataset.py) | 40.4 KB | Virtual windowing & scaler classes | PyTorch / Person 1 |
| **Training Engine** | [`ml/tcn_train.py`](file:///d:/SEM%207/VMplacement/ml/tcn_train.py) | 40.9 KB | 12-run ablation & checkpointing | PyTorch / Person 1 |
| **Evaluation Engine**| [`ml/tcn_evaluate.py`](file:///d:/SEM%207/VMplacement/ml/tcn_evaluate.py) | 28.5 KB | Phase 9 test evaluation engine | Person 1 / Paper |
| **Risk Pipeline** | [`ml/tcn_risk.py`](file:///d:/SEM%207/VMplacement/ml/tcn_risk.py) | 30.4 KB | Workload intelligence & verification | Person 1 / Audit |
| **Champion Model** | [`results/stage5/checkpoints/M3_FULL_standard_best.pt`](file:///d:/SEM%207/VMplacement/results/stage5/checkpoints/M3_FULL_standard_best.pt) | 628.6 KB | Locked TCN weights (Epoch 9) | Reproducibility |
| **Ablation Summary** | [`results/stage5/ablation/ablation_summary.csv`](file:///d:/SEM%207/VMplacement/results/stage5/ablation/ablation_summary.csv) | 2.6 KB | Full 12-run GPU ablation ranking | Report / Paper |
| **Ablation JSON** | [`results/stage5/ablation/ablation_summary.json`](file:///d:/SEM%207/VMplacement/results/stage5/ablation/ablation_summary.json) | 6.7 KB | Machine-readable ablation metadata | Audit |
| **Scientific Preds** | [`results/stage5/predictions/predictions.csv`](file:///d:/SEM%207/VMplacement/results/stage5/predictions/predictions.csv) | 6.0 MB | 82,980 test forecasts with ground truth | Person 2 (CloudSim) |
| **Test Summary** | [`results/stage5/predictions/evaluation_summary.json`](file:///d:/SEM%207/VMplacement/results/stage5/predictions/evaluation_summary.json) | 1.9 KB | Final test metrics (MAE 1.7179, RMSE 9.5368) | Report / Viva |
| **Horizon Metrics** | [`results/stage5/predictions/horizon_metrics.csv`](file:///d:/SEM%207/VMplacement/results/stage5/predictions/horizon_metrics.csv) | 755 B | Lead-time accuracy degradation table | Report / Paper |
| **Placement Input** | [`results/stage5/risk/risk_state.csv`](file:///d:/SEM%207/VMplacement/results/stage5/risk/risk_state.csv) | 720.8 KB | **Master online placement input table** | **Person 3 (AHO)** |
| **Frozen Calib** | [`results/stage5/risk/risk_calibration.json`](file:///d:/SEM%207/VMplacement/results/stage5/risk/risk_calibration.json) | 1.15 KB | Validation parameters ($V_{\max}, S_{\max}, \tau$) | Reproducibility |
| **Verification** | [`results/stage5/risk/risk_verification.json`](file:///d:/SEM%207/VMplacement/results/stage5/risk/risk_verification.json) | 1.60 KB | 17/17 acceptance test audit report | Audit / Faculty |
| **Person 3 Guide** | [`results/stage5/risk/README.md`](file:///d:/SEM%207/VMplacement/results/stage5/risk/README.md) | 5.94 KB | Person 3 integration instructions | Person 3 |
| **Design Document** | [`docs/phase10_ml_to_aho_interface_design.md`](file:///d:/SEM%207/VMplacement/docs/phase10_ml_to_aho_interface_design.md) | 15.7 KB | Formal mathematical interface design | Faculty / Thesis |
| **Schema Document** | [`docs/risk_state_schema.md`](file:///d:/SEM%207/VMplacement/docs/risk_state_schema.md) | 4.5 KB | Detailed data dictionary & typing | Person 3 / Person 2 |

---

## 17. Git Handoff Audit

* **Handoff Branch**: [`person1-ml-risk-handoff`](https://github.com/guhya-16/VM-placement-TCN-AHO/tree/person1-ml-risk-handoff)
* **Commit Hash**: `3aff9606f3d04252d72b8d0e9796274271f5888c`
* **Remote Repository**: `https://github.com/guhya-16/VM-placement-TCN-AHO.git` (`origin`)
* **Push Result**: **SUCCESS** (Remote tracking established)
* **Committed Files**: Exactly **29 approved files** (no raw data, no `__pycache__`, no temporary zip archives).
* **Working Tree**: Clean.

---

## 18. Final Person 1 Status

```text
================================================================================
PERSON 1 WORKLOAD FORECASTING & INTELLIGENCE PIPELINE:
STATUS: COMPLETE / FROZEN / HANDED OFF
================================================================================
```
Person 1's methodology, TCN model, risk proxy calibration, and interface schemas are frozen. Person 1 will not alter any artifacts unless the research team formally approves a joint architectural revision.

