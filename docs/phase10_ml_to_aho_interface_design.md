# Phase 10: ML → Adaptive Hippopotamus Optimization (AHO) Interface Design & Audit (Revised)
**Role**: Person 1 — Data Preprocessing, ML Workload Forecasting & Workload Intelligence  
**Downstream Consumer**: Person 3 — Adaptive Hippopotamus Optimization (AHO) & VM-to-PM Placement  
**Simulation Consumer**: Person 2 — CloudSim Plus Simulation Engine  
**Status**: REVISED DESIGN & AUDIT SPECIFICATION (Pre-Implementation)

---

## 1. Executive Summary & Architectural Scope

This document formally specifies the **workload intelligence interface** through which Person 1's Temporal Convolutional Network (TCN) forecasting outputs are converted into risk-aware placement signals consumable by Person 3's Adaptive Hippopotamus Optimization (AHO) algorithm.

```
+---------------------------------------------------------------------------------------------------+
| PERSON 1: DATA + ML ENGINE                                                                        |
| Bitbrains 2013-8 Telemetry                                                                        |
|   --> Stage 3 Policy C Preprocessing (5-minute canonical grid, 14 VMs)                            |
|   --> Stage 4 Split-Safe Sliding Window Extraction (L=288, H=12, W=300)                           |
|   --> Stage 5.3 Champion TCN Forecaster (M3_FULL_standard, F=10, RF=511)                          |
|   --> 12-Step Forecast Vectors [y_hat_{t+1}, ..., y_hat_{t+12}]                                   |
|   --> Workload Volatility Metric V(t) (strictly causal lookback K=24, past 2 hours)               |
|   --> Prediction-Risk Proxy R(t) (calibrated strictly on Validation sliding windows)              |
|   --> Categorical Risk State (LOW / MEDIUM / HIGH via Validation tertiles)                        |
|   --> Output Artifact: results/stage5/risk/risk_state.csv (6,915 decision rows)                   |
+---------------------------------------------------------------------------------------------------+
                                              |
                                              | Git Branch Handoff (Phase 11)
                                              v
+---------------------------------------------------------------------------------------------------+
| PERSON 3: OPTIMIZATION & PLACEMENT ENGINE                                                         |
| Adaptive Hippopotamus Optimization (AHO)                                                          |
|   --> Ingests risk_state.csv (TEST-TIME PLACEMENT INPUT / ONLINE-SIMULATION ARTIFACT)             |
|   --> Uses predicted_peak_cpu for capacity feasibility & SLA headroom constraints                 |
|   --> Uses predicted_mean_cpu for energy cost estimation in fitness function                      |
|   --> Uses risk_score & risk_state to adapt exploration vs. exploitation balance                  |
|   --> Person 3 OWNS all AHO parameters, capacity buffers, and VM-to-PM placement logic            |
|   --> Generates: placement.csv (VM-to-PM assignment matrix)                                       |
+---------------------------------------------------------------------------------------------------+
                                              |
                                              v
+---------------------------------------------------------------------------------------------------+
| PERSON 2: SIMULATION & VALIDATION ENGINE                                                          |
| CloudSim Plus Simulation                                                                          |
|   --> Ingests placement.csv + real physical host specs                                            |
|   --> Receives actual test-set workload ground truth ONLY at simulation runtime                   |
|   --> Simulates physical execution, dynamic migrations, host power states                         |
|   --> Evaluates SLA violations, power consumption (kWh), host shutdowns                           |
+---------------------------------------------------------------------------------------------------+
```

### Strict Ownership Boundaries
1. **Person 1 does NOT implement AHO**: Person 1 communicates semantic workload intelligence. Person 3 owns the mathematical mapping from risk/volatility to AHO algorithmic equations (e.g. inertia weights, predator-evasion probabilities, step sizes, and capacity buffers).
2. **Person 1 does NOT implement CloudSim**: Person 2 owns the simulation infrastructure, physical host modeling, and migration costs.
3. **Deterministic TCN Semantics**: The TCN is a deterministic neural network. Its uncertainty signal is termed a **"Prediction-Risk Proxy"** and never claimed to be Bayesian uncertainty or a probabilistic confidence interval.
4. **Strict Temporal Causality & Zero Leakage**: Offline test residuals are used solely for reporting model accuracy. No future actuals or test residuals may be accessed when computing online placement risk.

---

## 2. Issue 1 Resolution: Artifact Nature — "TEST-TIME PLACEMENT INPUT / ONLINE-SIMULATION ARTIFACT"

We explicitly clarify the methodological nature of `results/stage5/risk/risk_state.csv`:

### Official Artifact Classification:
> **`risk_state.csv` is a TEST-TIME PLACEMENT INPUT / ONLINE-SIMULATION ARTIFACT.**

### Operational Mechanics at Decision Time $t$:
1. **Input Information Available to AHO**:
   - Historical telemetry strictly up to cutoff $t$: $x_{\tau \le t}$ (`current_cpu`).
   - TCN predictions for $t+5\text{m} \dots t+60\text{m}$ (`predicted_cpu_t5m` through `predicted_cpu_t60m`).
   - Summary forecasts: `predicted_mean_cpu`, `predicted_peak_cpu`, `predicted_std_cpu`.
   - Historical workload volatility: `volatility_score` (computed from past 2 hours up to $t$).
   - Prediction-risk proxy: `risk_score` (computed using calibration parameters frozen from the Validation split).
   - Discrete risk tier: `risk_state` (`LOW`, `MEDIUM`, `HIGH`).
2. **Information Strictly Excluded from AHO**:
   - Future actual CPU values ($y_{\text{actual}}$ at $t+5\text{m} \dots t+60\text{m}$) are **NOT present**.
   - Test prediction errors, residuals, test MAE, and test RMSE are **NOT present**.
3. **Role of Future Ground Truth**:
   - The actual future test-set observations are revealed **exclusively to CloudSim Plus (Person 2)** at simulation runtime to benchmark whether the placement chosen by AHO resulted in SLA violations, high energy draw, or excessive migrations.

---

## 3. Issue 2 Resolution: Complete Exclusion of Future Actuals from `risk_state.csv`

A strict column audit confirms the interface separation:

- **INCLUDED in `risk_state.csv`**:
  - `vm_id`, `decision_timestamp` ($t$)
  - `current_cpu` ($x_t$)
  - `predicted_cpu_t5m` through `predicted_cpu_t60m`
  - `predicted_mean_cpu`, `predicted_peak_cpu`, `predicted_std_cpu`
  - `volatility_score` ($s_{K=24}(t)$)
  - `risk_score` ($R(t)$)
  - `risk_state` (`LOW`, `MEDIUM`, `HIGH`)
- **PROHIBITED from `risk_state.csv`**:
  - `actual_cpu`, `actual_cpu_t5m` ... `actual_cpu_t60m`
  - `error`, `absolute_error`, `squared_error`, `residual`
  - `test_mae`, `test_rmse`, `test_smape`
  - Any ground truth test-period telemetry

All ground-truth actuals, point-level errors, and squared errors remain restricted to [`results/stage5/predictions/predictions.csv`](file:///d:/SEM%207/VMplacement/results/stage5/predictions/predictions.csv) for retrospective model evaluation only.

---

## 4. Issue 3 Resolution: Removal of Prescribed AHO Buffers & Placement Policies

In accordance with Person 1's project boundary:
- All previously suggested buffer values (e.g. $5\%, 15\%, 25\%$) and placement rules (e.g. "isolate HIGH-risk VMs") are **formally removed** from Person 1's specifications.
- Person 1 provides **semantic workload intelligence only**:
  - **`LOW`**: Workload volatility and prediction unreliability risk are relatively low. The VM exhibits steady or quiescent behavior.
  - **`MEDIUM`**: Moderate workload fluctuations and prediction risk. The VM exhibits standard operational variance.
  - **`HIGH`**: Higher workload volatility, sharp dynamic transitions, or wider predictive unreliability.

> [!IMPORTANT]
> **Explicit Ownership Contract**:  
> **Person 3 is solely responsible for determining how `risk_state` and `risk_score` influence Adaptive HO exploration/exploitation parameters, PM capacity margins, VM grouping, and placement constraints.**

---

## 5. Issue 4 Resolution: Workload Volatility vs. Prediction Risk (Empirical Validation)

We explicitly separate physical workload dynamics from model forecast unreliability:

### 5.1 Workload Volatility (Physical Property)
- **Concept**: The physical turbulence, dispersion, and churn of the VM's observed telemetry over the immediate past.
- **Formula**:
  $$\text{Volatility}(t) = \sqrt{\frac{1}{K-1} \sum_{i=0}^{K-1} \left(x_{t-i} - \bar{x}_{t, K}\right)^2}, \quad K = 24 \text{ steps (2.0 hours)}$$
- **Properties**: Non-negative, zero-safe ($0.0\%$ CPU yields $0.0\%$ volatility), bounded in $[0, 50]\%$ CPU for typical workloads.

### 5.2 Prediction Risk (Model Unreliability Proxy)
- **Concept**: An empirical proxy representing the anticipated unreliability or error envelope of the deterministic TCN forecast.
- **Empirical Validation on the Validation Split (6,557 validation sliding windows)**:
  An empirical audit of the locked `M3_FULL_standard` model on the 6,557 validation sliding windows revealed:
  - Correlation between Input Volatility ($K=24$) and Actual Prediction Error ($\text{MAE}_{\text{val}}$): **`r = +0.5716`** (strong positive correlation).
  - Correlation between Forecast Trajectory Spread ($\max \hat{y} - \min \hat{y}$) and Actual Prediction Error ($\text{MAE}_{\text{val}}$): **`r = +0.4882`** (moderate-to-strong positive correlation).
  - Correlation between Input Volatility and Forecast Trajectory Spread: **`r = +0.5389`**.
- **Formula**:
  $$\text{RiskScore}(t) = w_v \cdot \min\left(1.0, \, \frac{\text{Volatility}(t)}{V_{\text{max}}}\right) + w_s \cdot \min\left(1.0, \, \frac{\text{Spread}(\hat{y})}{S_{\text{max}}}\right)$$
  where $w_v = 0.60, w_s = 0.40$, and $V_{\text{max}}, S_{\text{max}}$ are 95th percentiles of the validation sliding windows.
- **Honest Methodological Limitations**:
  - This is a **heuristic prediction-risk proxy**, not a Bayesian posterior or calibrated probabilistic uncertainty interval.
  - If a VM experiences an unprecedented spike from an idle baseline, the proxy will evaluate historical volatility as low and may underestimate risk.
  - Conversely, if a volatile VM is predicted with unusually high accuracy, the proxy will still classify it as high-risk due to its input turbulence.

---

## 6. Issue 5 Resolution: Leakage-Safe Calibration on Validation Sliding Windows

To ensure complete statistical transparency:
- **Sample Population**: Calibration parameters are computed across the **6,557 validation sliding windows**.
- **Window Dependency Note**: We explicitly document that these 6,557 validation sliding windows are **overlapping sliding windows** (derived from 9,498 contiguous validation observations across the 14 VMs) and are not independent identically distributed (i.i.d.) observations.
- **Frozen Calibration Parameters**:
  - $V_{\text{max}}$: 95th percentile of volatility across validation sliding windows.
  - $S_{\text{max}}$: 95th percentile of forecast spread across validation sliding windows.
  - $\tau_{\text{low}}$: 33.3rd percentile of validation risk scores.
  - $\tau_{\text{high}}$: 66.7th percentile of validation risk scores.
- These four scalar constants are exported to `results/stage5/risk/risk_calibration.json` and frozen before generating `risk_state.csv`. Zero test data is used.

---

## 7. Issue 6 Resolution: Decision Snapshot Format Interface

The interface uses the **Decision Snapshot Format**:
- Exactly **one row per VM per decision timestamp $t$**.
- `decision_timestamp` = cutoff timestamp $t$.
- At decision timestamp $t$, Person 3 filters:
  ```python
  snapshot_t = df_risk[df_risk["decision_timestamp"] == t]
  ```
  This returns the simultaneous workload intelligence vector for all active candidate VMs at that exact decision cycle.

---

## 8. Issue 7 Resolution: Handling of Short-Lived VMs

In our 14-VM cohort, 7 VMs (`VM_002`, `VM_004`, `VM_006`, `VM_014`, `VM_018`, `VM_161`, `VM_1209`) are short-lived:
- Their active lifespans range from 0.11 days to 1.89 days.
- In the 15% test partition, their telemetry contains only 6 to 83 observations (30 minutes to ~7 hours).
- Because the TCN architecture strictly requires lookback $L=288$ steps (24.0 hours) and horizon $H=12$ steps ($W=300$ steps = 25.0 hours), **no contiguous sliding window of length 300 can be formed within their test partitions without cross-partition leakage or synthetic zero-padding**.
- **Handoff Policy for Short-Lived VMs**:
  1. These 7 short-lived VMs produce **0 test sliding windows** and are therefore **excluded from `risk_state.csv`** (which contains the 6,915 valid test windows from the 7 active VMs).
  2. Person 1 does **NOT** invent an unverified fallback heuristic model (e.g. simple moving average or static mean) for short-lived VMs, as our scope is strictly the validated TCN forecaster.
  3. If Person 2/3 requires placing short-lived VMs in simulation during their brief lifespans, Person 3/Person 2 must apply a documented non-TCN fallback placement rule (e.g. reactive placement based on instantaneous request capacity).

---

## 9. Issue 8 Resolution: Final Person 1 → Person 3 Contract

### PERSON 1 GUARANTEES:
1. **Valid VM ID**: Canonical VM identifiers (`VM_001` .. `VM_025`).
2. **Decision Timestamp**: Synchronized Unix epoch timestamp $t$ (5-minute canonical grid).
3. **Current CPU**: Ground-truth observed utilization at cutoff step $t$: $x_t \in [0.0, 100.0]\%$.
4. **12-Step TCN Forecast**: Non-negative multi-step predictions $\hat{y}_{t+1} \dots \hat{y}_{t+12}$ in native CPU%.
5. **Predicted Mean**: Exact arithmetic mean of the 12 forecast steps: $\frac{1}{12}\sum_{h=1}^{12} \hat{y}_{t+h}$.
6. **Predicted Peak**: Exact maximum of the 12 forecast steps: $\max_{h \in \{1..12\}} \hat{y}_{t+h}$.
7. **Predicted Forecast Variability**: Standard deviation across the 12 forecast steps.
8. **Historical Workload Volatility**: Sample standard deviation over past 24 steps ($K=24$, 2 hours) up to $t$.
9. **Prediction-Risk Proxy**: Standardized risk score in $[0.0, 1.0]$ combining volatility and forecast spread.
10. **Categorical Risk State**: Discrete state $\in \{\text{"LOW"}, \text{"MEDIUM"}, \text{"HIGH"}\}$ derived via Validation tertiles.
11. **Calibration Metadata**: Fully documented scalar constants frozen from Validation sliding windows.
12. **Zero Future Actual Leakage**: No future test actuals, test residuals, or test error metrics are present.

### PERSON 3 OWNS:
1. **VM $\to$ PM Encoding**: Vector or matrix representations of VM-to-host allocations.
2. **PM Capacity Constraints**: Defining physical server CPU, RAM, and bandwidth feasibility boundaries.
3. **Fitness Function Formulation**: Mathematical weighting of power consumption, migration costs, and SLA risks.
4. **Standard Hippopotamus Optimization (HO)**: Baseline metaheuristic implementation.
5. **Adaptive Hippopotamus Optimization (AHO)**: Incorporating dynamic parameter adaptations.
6. **Mapping Risk $\to$ Optimization Behavior**: Deciding how `risk_state` and `risk_score` adjust exploration/exploitation.
7. **Capacity Headroom & Buffers**: Defining safety margins (e.g. host headroom allowances).
8. **Final `placement.csv`**: Generating the final VM-to-PM mapping artifact for CloudSim Plus.
