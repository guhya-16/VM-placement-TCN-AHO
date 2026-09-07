# Walkthrough — Stage 4: EDA + Temporal Window Generation

## Objective & Experimental Context

**Stage 4** establishes the formal methodological bridge between validated dataset preprocessing (Stage 3 Policy C: operational active-lifetime extraction, canonical 5-minute grid alignment, and limited-gap interpolation) and downstream Temporal Convolutional Network (TCN) workload prediction.

The primary responsibilities of Stage 4 are:
1. **Exploratory Data Analysis (EDA)**: Conduct comprehensive statistical profiling, resource cross-correlation (both pooled cohort-level and within-VM), rolling volatility analysis, and objective autocorrelation evaluation up to lag 288 (24 hours) across diverse cloud operational regimes.
2. **Temporal Window Generation & Validation**: Transform aligned active time-series telemetry into ML-ready sliding input/target sequence pairs ($X \in \mathbb{R}^{N \times 288 \times F}$, $y \in \mathbb{R}^{N \times 12}$) adhering strictly to temporal causality, split-safe chronological partitioning, and exact 300-second contiguity validation.

> 📌 **Methodological Disclaimer & Dataset Safety**
>
> All evaluations are conducted on the controlled 14-VM representative cohort established in Stage 3, spanning 6 distinct operational regimes (continuous bursty, steady moderate, periodic spikes, idle, decommissioned Type B, and transient ephemeral). Raw Bitbrains CSV files in `dataset/fastStorage/2013-8/` remain strictly read-only and unmodified. Model training is deferred to Stage 5.

---

## Visual Summary: Workload Profiles, Temporal Dynamics & Splitting

### 1. Workload Distributions & Operating Regimes
![CPU Usage Distribution](results/stage4/figures/cpu_distribution.png)
*Figure 1: Workload distribution profiling across representative regimes. (1A) Comparative boxplots showing extreme skewness in bursty VMs (`VM_001`) versus tightly bounded steady workloads (`VM_003`, `VM_012`). (1B) Gaussian Kernel Density Estimates (KDE) demonstrating distinct utilization density regimes.*

![Workload Timelines](results/stage4/figures/cpu_workload_examples.png)
*Figure 2: Empirical CPU utilization trajectories illustrating 5 operational profiles: bursty peak load (`VM_001`), steady moderate load (`VM_003`), idle provisioned infrastructure (`VM_013`), Type B active lifecycle (`VM_002`, 1.89 days), and transient ephemeral execution (`VM_1209`, 9.75 hours).*

---

### 2. Rolling Volatility & Cross-Metric Correlations
![Rolling Volatility](results/stage4/figures/rolling_volatility.png)
*Figure 3: Workload volatility dynamics for `VM_001`. (3A) Workload trajectory overlaid with 1-hour rolling mean and $\pm 1\sigma$ rolling volatility envelope. (3B) Direct comparison of 1-hour short-term local volatility ($\sigma_{12}$) versus 6-hour intermediate macro volatility ($\sigma_{72}$).*

![Correlation Matrix](results/stage4/figures/correlation_matrix.png)
*Figure 4: Empirical Pearson cross-correlation heatmap across cohort telemetry. Demonstrates strong collinearity between CPU % and CPU MHz ($r=0.98$), high correlation with disk write throughput ($r=0.77$) and memory consumption ($r=0.69$), and low linear correlation with network traffic.*

---

### 3. Autocorrelation Analysis & Split-Safe Partitioning
![Autocorrelation Function](results/stage4/figures/autocorrelation.png)
*Figure 5: Empirical Autocorrelation Function (ACF) curves evaluated objectively up to lag 288 (24 hours). Demonstrates strong short-term temporal persistence (lag 1..12) across all VMs, with `VM_011` exhibiting distinct periodic diurnal cyclic peaks around lag 288 (24h).*

![Temporal Split Diagram](results/stage4/figures/temporal_split.png)
*Figure 6: Split-safe chronological partitioning (70% Train / 15% Validation / 15% Test) for continuous 30-day `VM_001` and decommissioned 1.89-day `VM_002`. Both historical input $X$ and forecast target $y$ are constrained entirely within the same partition, preventing boundary leakage.*

---

## 1. Comprehensive Statistical Profiling (14-VM Cohort)

Descriptive statistics, operational utilization fractions, and distribution metrics computed across the preprocessed 5-minute aligned telemetry:

| VM ID | Source File | Operational Category | Aligned Steps | Active Days | Mean CPU [%] | Std Dev [%] | Median [%] | Min [%] | Max [%] | $p_{90}$ [%] | $p_{99}$ [%] | CV ($\sigma/\mu$) | Skewness | Kurtosis | Zero Fraction | High Util ($>80\%$) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`VM_001`** | `1.csv` | Type A (Bursty Peak) | 8,640 | 30.00 | 4.028 | 16.913 | 0.617 | 0.500 | 97.867 | 0.717 | 93.577 | 4.199 | 4.888 | 22.147 | 0.0000 | 0.0338 |
| **`VM_003`** | `3.csv` | Type A (Steady Moderate) | 8,639 | 30.00 | 3.255 | 0.510 | 3.133 | 2.867 | 9.467 | 3.500 | 6.000 | 0.157 | 6.771 | 57.729 | 0.0000 | 0.0000 |
| **`VM_011`** | `11.csv` | Type A (Periodic Spikes) | 8,640 | 30.00 | 2.526 | 4.447 | 2.000 | 1.000 | 54.333 | 2.000 | 20.470 | 1.760 | 9.657 | 97.729 | 0.0000 | 0.0000 |
| **`VM_012`** | `12.csv` | Type A (Low Steady) | 8,640 | 30.00 | 1.008 | 0.198 | 1.000 | 1.000 | 10.733 | 1.000 | 1.133 | 0.197 | 44.605 | 2043.513 | 0.0000 | 0.0000 |
| **`VM_013`** | `13.csv` | Type A (Idle Provisioned) | 8,640 | 30.00 | 0.009 | 0.107 | 0.000 | 0.000 | 2.000 | 0.000 | 0.133 | 11.569 | 14.713 | 225.475 | 0.9796 | 0.0000 |
| **`VM_023`** | `23.csv` | Type A (High Sustained) | 8,639 | 30.00 | 5.538 | 1.867 | 5.000 | 1.900 | 55.733 | 7.867 | 11.221 | 0.337 | 5.788 | 122.278 | 0.0000 | 0.0000 |
| **`VM_025`** | `25.csv` | Type A (Heavy Sustained) | 8,635 | 30.00 | 5.170 | 1.884 | 4.467 | 0.500 | 62.567 | 7.767 | 11.067 | 0.365 | 4.520 | 101.428 | 0.0000 | 0.0000 |
| **`VM_002`** | `2.csv` | Type B (Decommissioned) | 545 | 1.89 | 7.131 | 23.213 | 0.250 | 0.250 | 92.267 | 1.330 | 91.954 | 3.255 | 3.215 | 8.473 | 0.0000 | 0.0679 |
| **`VM_004`** | `4.csv` | Type B (Decommissioned) | 545 | 1.89 | 7.100 | 23.111 | 0.250 | 0.250 | 92.200 | 1.440 | 91.487 | 3.255 | 3.217 | 8.490 | 0.0000 | 0.0679 |
| **`VM_006`** | `6.csv` | Type B (Decommissioned) | 545 | 1.89 | 7.181 | 23.277 | 0.250 | 0.000 | 92.321 | 1.340 | 91.993 | 3.241 | 3.205 | 8.407 | 0.0018 | 0.0697 |
| **`VM_014`** | `14.csv` | Type B (Decommissioned) | 545 | 1.89 | 7.137 | 23.177 | 0.250 | 0.250 | 92.300 | 1.243 | 91.854 | 3.248 | 3.213 | 8.467 | 0.0000 | 0.0679 |
| **`VM_018`** | `18.csv` | Type B (Decommissioned) | 545 | 1.89 | 7.141 | 23.232 | 0.250 | 0.250 | 92.283 | 1.493 | 91.899 | 3.253 | 3.217 | 8.491 | 0.0000 | 0.0679 |
| **`VM_161`** | `161.csv` | Transient Lifecycle | 31 | 0.11 | 5.289 | 12.294 | 0.000 | 0.000 | 61.133 | 13.333 | 52.033 | 2.325 | 3.467 | 12.354 | 0.5484 | 0.0000 |
| **`VM_1209`**| `1209.csv`| Transient Ephemeral | 117 | 0.41 | 0.970 | 0.653 | 0.933 | 0.800 | 7.867 | 1.067 | 1.403 | 0.673 | 10.173 | 104.866 | 0.0000 | 0.0000 |

---

## 2. Feature Candidate Analysis & Dual Correlation Evaluation

To avoid confounding between-VM differences with individual workload dynamics, Stage 4 evaluates both **cohort-pooled correlations** and **within-VM correlations across individual VM time series**.

### 1. Within-VM Correlation Summary (Dynamic Features)

Correlations evaluated within each individual VM's time-series, reported as the mean, median, min, and max across the 14 VMs:

| Dynamic Feature | Valid VMs | Mean Pearson $r$ | Median Pearson $r$ | Min Pearson $r$ | Max Pearson $r$ | Mean Spearman $\rho$ | Median Spearman $\rho$ | Workload Relationship & Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`cpu_usage_mhz`** | 14 | **1.0000** | **1.0000** | 1.0000 | 1.0000 | 0.9221 | 1.0000 | Within each VM, CPU usage MHz is deterministically proportional to CPU usage %, because CPU capacity is constant during the active lifetime. Excluded as redundant ($r \equiv 1.0000$). |
| **`memory_usage_kb`** | 14 | **0.5462** | **0.6384** | 0.0336 | 0.9178 | 0.3944 | 0.3875 | Positive relationship; dynamic RAM footprint tracks active computation. |
| **`disk_write_kbps`** | 14 | **0.5993** | **0.7590** | 0.1307 | 0.9815 | 0.4421 | 0.4870 | High correlation with compute bursts due to synchronous write flushes. |
| **`disk_read_kbps`** | 13 | **0.3640** | **0.2322** | 0.0855 | 0.8673 | 0.3763 | 0.4029 | Moderate correlation; captures batch reads (`VM_012` has 0 reads). |
| **`network_received_kbps`**| 12 | **0.1903** | **0.1437** | 0.0186 | 0.4254 | **0.5314** | **0.5844** | High monotonic rank correlation ($\rho=0.58$) indicates bursty request arrival. |
| **`network_transmitted_kbps`**| 12 | **0.3647** | **0.3993** | 0.0453 | 0.7775 | **0.5306** | **0.5333** | Strong rank correlation reflecting egress response bursts. |

### 2. Descriptive Cohort-Level Feature Analysis Table

| Candidate Feature | Unit | Role | Status | Pooled Pearson $r$ | Pooled Spearman $r$ | Skewness | Zero Fraction | Temporal Leakage Audit | Methodological Justification |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **`cpu_usage_percent`** | % | Primary Input / Target | **INCLUDED** | 1.0000 | 1.0000 | 9.27 | 13.39% | **Zero Leakage**: Input strictly indexed $t-287 \dots t$. | Target variable ($t+1 \dots t+12$) and core autoregressive input. |
| **`cpu_usage_mhz`** | MHz | Telemetry | **EXCLUDED** | 0.9765 | 0.9102 | 10.53 | 13.39% | N/A (Excluded) | Within each VM, CPU usage MHz is deterministically proportional to CPU usage %, because CPU capacity is constant during the active lifetime. |
| **`cpu_capacity_mhz`** | MHz | Provisioning | **EXCLUDED (Metadata)** | 0.1426 | 0.1759 | 1.17 | 0.00% | Invariant | Static throughout active life ($\sigma=0$ within VM); causes division-by-zero in standard scalers. |
| **`cpu_cores`** | count | Provisioning | **EXCLUDED (Metadata)** | 0.1572 | 0.2262 | 1.01 | 0.00% | Invariant | Static infrastructure configuration; zero temporal variance. |
| **`memory_usage_kb`** | KB | Telemetry | **INCLUDED** | 0.6949 | 0.8431 | 13.09 | 22.76% | **Zero Leakage**: Known at or before $t$. | Dynamic memory consumption; strong rank correlation with computational demand. |
| **`memory_capacity_kb`**| KB | Provisioning | **EXCLUDED (Metadata)** | 0.0682 | 0.2636 | 2.03 | 0.00% | Invariant | Static allocated RAM capacity; zero variance within VM. Preserved as metadata. |
| **`disk_read_kbps`** | KB/s | I/O Throughput | **INCLUDED** | 0.2404 | 0.1856 | 44.17 | 90.38% | **Zero Leakage**: Known at or before $t$. | Captures I/O-intensive read operations; highly zero-inflated. |
| **`disk_write_kbps`** | KB/s | I/O Throughput | **INCLUDED** | 0.7681 | 0.6672 | 16.66 | 23.26% | **Zero Leakage**: Known at or before $t$. | Strong correlation ($r=0.77$) with CPU spikes due to active computation flushes and logging. |
| **`network_received_kbps`**| KB/s| Network Traffic | **INCLUDED** | 0.1974 | 0.7003 | 37.29 | 58.96% | **Zero Leakage**: Known at or before $t$. | Ingress traffic; leading indicator of incoming client requests. |
| **`network_transmitted_kbps`**| KB/s| Network Traffic| **INCLUDED** | 0.2173 | 0.3810 | 55.16 | 40.53% | **Zero Leakage**: Known at or before $t$. | Egress traffic; reflects server response volume. |
| **`hour_sin`** | $[-1, 1]$ | Calendar Temporal | **INCLUDED** | -0.0597 | -0.0223 | -0.01 | 0.35% | **Zero Leakage**: Deterministic function of UTC $t$. | Continuous smooth encoding of daily diurnal cycles ($\sin(2\pi \cdot \text{hour}_{\text{UTC}} / 24)$). |
| **`hour_cos`** | $[-1, 1]$ | Calendar Temporal | **INCLUDED** | -0.0451 | -0.0158 | -0.01 | 0.00% | **Zero Leakage**: Deterministic function of UTC $t$. | Cosine counterpart resolving circular boundary ($23:55 \rightarrow 00:00$). |
| **`day_sin`** | $[-1, 1]$ | Calendar Temporal | **INCLUDED** | 0.0582 | -0.0234 | -0.17 | 14.19% | **Zero Leakage**: Deterministic function of UTC $t$. | Continuous encoding of weekly business vs weekend workload cycles ($\sin(2\pi \cdot \text{day}_{\text{UTC}} / 7)$). |
| **`day_cos`** | $[-1, 1]$ | Calendar Temporal | **INCLUDED** | 0.0392 | 0.0212 | -0.04 | 0.00% | **Zero Leakage**: Deterministic function of UTC $t$. | Cosine counterpart resolving weekly circular boundary. |

### 3. Configurable Feature Modes for Stage 5 Ablation Studies
- **Mode 1 (Univariate Baseline)**: $F = 1$ channel (`cpu_usage_percent`)
- **Mode 2 (Multivariate Resource)**: $F = 6$ channels (`cpu_usage_percent`, `memory_usage_kb`, `disk_read_kbps`, `disk_write_kbps`, `network_received_kbps`, `network_transmitted_kbps`)
- **Mode 3 (Full Multivariate + Temporal)**: $F = 10$ channels (Mode 2 + `hour_sin`, `hour_cos`, `day_sin`, `day_cos`)

---

## 3. Split-Safe Chronological Partitioning & Empirical Window Accounting

### Partitioning Rules
Each VM's aligned trace is partitioned chronologically into:
- **Train Split (70%)**: First 70% of chronological steps.
- **Validation Split (15%)**: Intermediate 15% of chronological steps.
- **Test Split (15%)**: Final 15% of chronological steps.

### Strict Split Containment Rule
A sliding window $[t_i \dots t_{i+299}]$ ($L=288$ history, $H=12$ forecast) is valid if and only if **both $X$ and $y$ reside entirely within the same partition**:
- **Train Window**: $0 \le i \le N_{\text{train}} - 300$
- **Validation Window**: $N_{\text{train}} \le i \le (N_{\text{train}} + N_{\text{val}}) - 300$
- **Test Window**: $(N_{\text{train}} + N_{\text{val}}) \le i \le N_{\text{total}} - 300$

No arbitrary sliding buffers are added. Windows straddling partition boundaries are naturally excluded.

### Empirical Window Counts Across Cohort

| VM ID | Source File | Category | Aligned Steps | Train Steps | Val Steps | Test Steps | Valid Train Windows | Valid Val Windows | Valid Test Windows | Total Valid Windows | Reference Whole Trace | Boundary Exclusions | Unhandled Gap Exclusions |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`VM_001`** | `1.csv` | Type A | 8,640 | 6,048 | 1,296 | 1,296 | 5,749 | 997 | 997 | **7,743** | 8,341 | 598 | 0 |
| **`VM_003`** | `3.csv` | Type A | 8,640 | 6,048 | 1,296 | 1,296 | 5,449 | 997 | 997 | **7,443** | 8,341 | 598 | 300 (Train) |
| **`VM_011`** | `11.csv` | Type A | 8,640 | 6,048 | 1,296 | 1,296 | 5,749 | 997 | 997 | **7,743** | 8,341 | 598 | 0 |
| **`VM_012`** | `12.csv` | Type A | 8,640 | 6,048 | 1,296 | 1,296 | 5,749 | 997 | 997 | **7,743** | 8,341 | 598 | 0 |
| **`VM_013`** | `13.csv` | Type A | 8,640 | 6,048 | 1,296 | 1,296 | 5,749 | 997 | 997 | **7,743** | 8,341 | 598 | 0 |
| **`VM_023`** | `23.csv` | Type A | 8,640 | 6,048 | 1,296 | 1,296 | 5,749 | 997 | 965 | **7,711** | 8,341 | 598 | 32 (Test) |
| **`VM_025`** | `25.csv` | Type A | 8,640 | 6,048 | 1,296 | 1,296 | 5,448 | 575 | 965 | **6,988** | 8,341 | 598 | 755 (Tr/Val/Te) |
| **`VM_002`** | `2.csv` | Type B | 545 | 381 | 81 | 83 | 82 | 0 | 0 | **82** | 246 | 164 | 0 |
| **`VM_004`** | `4.csv` | Type B | 545 | 381 | 81 | 83 | 82 | 0 | 0 | **82** | 246 | 164 | 0 |
| **`VM_006`** | `6.csv` | Type B | 545 | 381 | 81 | 83 | 82 | 0 | 0 | **82** | 246 | 164 | 0 |
| **`VM_014`** | `14.csv` | Type B | 545 | 381 | 81 | 83 | 82 | 0 | 0 | **82** | 246 | 164 | 0 |
| **`VM_018`** | `18.csv` | Type B | 545 | 381 | 81 | 83 | 82 | 0 | 0 | **82** | 246 | 164 | 0 |
| **`VM_161`** | `161.csv`| Transient | 31 | 21 | 4 | 6 | 0 | 0 | 0 | **0** | 0 | 0 | 0 |
| **`VM_1209`**| `1209.csv`| Transient | 117 | 81 | 17 | 19 | 0 | 0 | 0 | **0** | 0 | 0 | 0 |
| **Total** | — | — | **63,353** | **44,343** | **9,498** | **9,512** | **40,052** | **6,557** | **6,915** | **53,524** | **59,617** | **5,006** | **1,087** |

> 📌 **Exact Step & Window Accounting Breakdown**
>
> - **Total Aligned Steps (63,353)**: Sum of Train steps (44,343 [69.99%]), Validation steps (9,498 [15.00%]), and Test steps (9,512 [15.01%]) across all 14 VMs matches 63,353 exactly.
>   *(Note: The 40-step delta between individual Train/Test floors arises because integer truncation $\lfloor 0.70 \times N \rfloor$ on sub-day transient traces like VM_161 [21/4/6] and VM_1209 [81/17/19] allocates remainder steps to the test partition, ensuring zero observation loss).*
> - **Stage 3 Reference Count**: **59,617** whole-trace contiguous windows.
> - **Split Boundary Omissions**: Exactly **5,006** windows omitted (4,186 across the 7 continuous Type A VMs [$7 \times 598$], plus 820 across the 5 Type B VMs [$5 \times 164$]).
> - **Unhandled Gap Omissions**: Exactly **1,087** candidate windows contained un-interpolated missing steps ($>2$ steps) and were rejected by the completeness check.
> - **Final Split-Safe Windows**: **53,524 valid windows** ($N_{\text{train}} = 40,052$ [74.8%], $N_{\text{val}} = 6,557$ [12.3%], $N_{\text{test}} = 6,915$ [12.9%]).

---

## 4. Continuity Validation & Interpolation Audit

### 1. Step Contiguity & Causal Verification
- **Exact 300-Second Contiguity**: **100.0%** of the 53,524 validated windows satisfy $\Delta t = t_{j+1} - t_j \equiv 300\text{ seconds}$ across all 299 adjacent step pairs. No tolerance window was permitted on aligned data.
- **Single-VM Constraint**: 100.0% of windows originate from exactly one VM trace.
- **Strict Causality**: For every window, $\max(\text{timestamp}(X)) < \min(\text{timestamp}(y))$ was asserted and verified.
- **Completeness**: Exactly 0 `NaN` or infinite values across all feature and target matrices.

### 2. Quantitative Interpolation Audit (`is_interpolated`)
- Total feature steps across all valid training windows: $53,524 \times 288 = 15,414,912$ step observations.
- Total interpolated synthetic steps in $X$: **14,640 steps** (**0.0950%** of all historical feature observations).
- Total interpolated synthetic steps in $y$: **621 steps** (**0.0967%** of all forecast target observations).
- Windows containing $\ge 1$ synthetic step: 9,193 windows (17.18%).

---

## 5. Sequence-Length Sensitivity & Short-Lived VM Handling

We explicitly tested window yield across 3 sequence scales to verify that short-lived VMs produce 0 windows under $W=300$ strictly due to sequence-length physics, not arbitrary deletion:

| VM ID | Source File | Category | Active Steps | Active Lifespan | $W=300$ Windows ($25\text{h}$) | $W=72$ Windows ($6\text{h}$) | $W=36$ Windows ($3\text{h}$) | Operational Lifecycle Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`VM_1209`** | `1209.csv` | Transient | 117 | 9.75 hours | **0** | **46** | **82** | Sub-day execution; yields 46 windows for 6h models. |
| **`VM_161`** | `161.csv` | Transient | 31 | 2.58 hours | **0** | **0** | **0** | 2.5h task; yields 8 windows for 2h models ($W=24$). |
| **`VM_002`** | `2.csv` | Type B | 545 | 1.89 days | **82** | **332** | **440** | Decommissioned node; produces windows across all scales. |
| **`VM_001`** | `1.csv` | Type A | 8,640 | 30.00 days | **7,743** | **8,427** | **8,535** | Continuous 30-day node. |

Short-lived VMs are retained in raw storage and documented in the pipeline.

---

## 6. Complete Train-Only Feature Scaling Investigation

Feature distributions evaluated strictly on training partitions ($t \in \text{Train}$) comparing Raw, `StandardScaler`, `RobustScaler`, and $\log_{1p} + \text{StandardScaler}$:

| Feature | Raw Skewness | StandardScaler Skewness | StandardScaler Max ($z_{\max}$) | RobustScaler Skewness | RobustScaler Max | $\log_{1p}$ + StandardScaler Skewness | $\log_{1p}$ + StandardScaler Max | Distributional Candidate Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`cpu_usage_percent`** | 9.58 | 9.58 | 11.44 | 9.58 | 29.88 | **0.79** | **4.73** | Distributionally preferred candidate for skew compression; final choice validated on model loss in Stage 5. |
| **`memory_usage_kb`** | 13.76 | 13.76 | 32.64 | 13.76 | 43.51 | **-1.09** | **1.42** | Distributionally preferred candidate for outlier compression. |
| **`disk_read_kbps`** | 35.09 | 35.09 | 54.34 | 35.09 | 3,624.67 | **4.84** | **10.53** | Distributionally preferred candidate for zero-inflated throughput. |
| **`disk_write_kbps`** | 15.08 | 15.08 | 26.99 | 15.08 | 4,486.82 | **2.70** | **7.48** | Distributionally preferred candidate for skew compression. |
| **`network_received_kbps`**| 37.90 | 37.90 | 63.10 | 37.90 | 6,377.78 | **2.24** | **8.70** | Distributionally preferred candidate for burst compression. |
| **`network_transmitted_kbps`**| 46.65 | 46.65 | 87.34 | 46.65 | 1,082.33 | **2.10** | **10.78** | Distributionally preferred candidate for burst compression. |

> 📌 **Scaling Strategy Justification & Cautious Scope**
>
> 1. **Linear Standardization Invariance**: Linear scaling ($z = (x - \mu)/\sigma$) preserves skewness identically ($\text{Skew}(z) = \text{Skew}(x)$), leaving normalized maxima up to $87.3\sigma$.
> 2. **RobustScaler Breakdown on Sparse Throughput**: For zero-inflated throughput features (`disk_read_kbps`, `network_received_kbps`), the interquartile range (IQR) is small, causing divisions that inflate scaled maxima to $> 3,000 \dots 6,000$.
> 3. **Nonlinear Compression**: $\log_{1p}(x) = \ln(1 + x)$ followed by standardization compresses skewness down to $0.79 \dots 4.84$ and bounds maxima to $\le 10.8\sigma$.
> 4. **No Premature Optimization Claim**: $\log_{1p} + \text{StandardScaler}$ is reported as the **distributionally preferred candidate** based on statistical properties. Final model-performance selection (comparing native/MinMax vs log-standardized) will be conducted empirically in Stage 5 based on validation loss.

---

## 7. Audit & Verification of `ml/preprocessing_evaluation.py`

To guarantee that the approved Stage 3 Policy C pipeline remains perfectly intact and non-destructive, `ml/preprocessing_evaluation.py` was audited against its Stage 3 baseline.

### Exact Changes Made
A single non-interfering modification was made inside `evaluate_policy_c_active_aligned()`:
```diff
--- a/ml/preprocessing_evaluation.py (Stage 3 Baseline)
+++ b/ml/preprocessing_evaluation.py (Stage 4 Active)
@@ -288,6 +288,7 @@
         resampled["cpu_usage_percent"].interpolate(method="linear", limit=2).notna()
     )
     interpolated_steps = int(interpolated_mask.sum())
+    resampled["is_interpolated"] = interpolated_mask.astype(int)
 
     for col in numeric_cols:
         resampled[col] = resampled[col].interpolate(method="linear", limit=2)
```

### Invariant Core Policy Confirmation
- **Candidate Active-Lifetime Rule**: Unchanged (`active = (memory_capacity_kb > 0) & (cpu_capacity_mhz > 0)`).
- **Canonical 5-Minute Grid**: Unchanged (`.resample("5min").mean()`).
- **Limited-Gap Interpolation**: Unchanged (linear interpolation strictly restricted to isolated gaps $\le 2$ steps via `limit=2`).
- **Unhandled Gaps**: Unchanged (gaps $> 2$ steps remain NaN, and all overlapping windows are rejected).
- **Raw File Protection**: Unchanged (all files in `dataset/fastStorage/2013-8/` remain strictly read-only).

---

## 8. Artifacts Created & Updated

| Artifact Path | Type | Description |
| :--- | :--- | :--- |
| **`ml/stage4_eda.py`** | Python Module | Ingests cohort, computes descriptive/distributional statistics, pooled and within-VM correlations, rolling volatility, empirical ACF up to lag 288, and exports figures. |
| **`ml/temporal_windows.py`** | Python Module | Implements split-safe 70/15/15 chronological partition, exact 300s window generation ($L=288, H=12, W=300$), interpolation auditing, and complete scaling evaluation. |
| **`results/stage4/eda_summary.csv`** | CSV Dataset | Full statistical profile across the 14 VMs (mean, std, percentiles, CV, skewness, zero fraction). |
| **`results/stage4/feature_analysis.csv`** | CSV Dataset | Pooled cohort-level empirical correlations, skewness, and feature candidate inclusion status. |
| **`results/stage4/within_vm_correlations.csv`** | CSV Dataset | Dedicated within-VM correlation summary (mean, median, min, max Pearson/Spearman across 14 VMs). |
| **`results/stage4/scaling_comparison.csv`** | CSV Dataset | Complete scaling comparison table (Raw, StandardScaler, RobustScaler, log1p + StandardScaler). |
| **`results/stage4/split_summary.csv`** | CSV Dataset | Chronological partition step counts, valid window counts, and interpolation statistics per split. |
| **`results/stage4/vm_window_counts.csv`** | CSV Dataset | Multi-scale window counts ($W=300, W=72, W=36$) and boundary-omission accounting per VM. |
| **`results/stage4/window_validation_report.txt`** | Text Report | Comprehensive audit of window contiguity, zero boundary crossings, and $<0.10\%$ synthetic data exposure. |
| **`results/stage4/stage4_summary.txt`** | Text Report | Executive summary of Stage 4 findings and dataset accounting. |
| **`results/stage4/figures/` (6 PNGs)** | Visual Figures | Boxplots/KDE, timelines, rolling volatility, correlation matrix, ACF curve, and chronological split diagram. |
| **`walkthrough_stage4.md`** | Root Documentation | Master walkthrough for Stage 4 saved directly in the project workspace root. |
