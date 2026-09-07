# Walkthrough — Stage 5.5 / Phase 11: Workload Intelligence, Prediction-Risk Pipeline & Final Pre-Git Audit
**Role**: Person 1 (Data Preprocessing, ML Forecaster & Workload Intelligence)  
**Consumer**: Person 3 (Adaptive Hippopotamus Optimization — AHO)  
**Downstream Simulation**: Person 2 (CloudSim Plus Simulation Engine)  
**Status**: `READY FOR GIT HANDOFF` (Awaiting user confirmation of `[APPROVAL REQUIRED]` Option B)

---

## 1. Executive Summary & Deliverables

Phase 11 implements Person 1's final workload intelligence and prediction-risk proxy layer ([`ml/tcn_risk.py`](file:///d:/SEM%207/VMplacement/ml/tcn_risk.py)), bridging the champion Temporal Convolutional Network ([`TCNForecaster`](file:///d:/SEM%207/VMplacement/ml/tcn_model.py)) with Person 3's Adaptive Hippopotamus Optimization (AHO).

All artifacts are generated, validated, and frozen in [`results/stage5/risk/`](file:///d:/SEM%207/VMplacement/results/stage5/risk/):

| Artifact | Path | Size | Status | Purpose |
| :--- | :--- | :---: | :---: | :--- |
| **Pipeline Script** | [`ml/tcn_risk.py`](file:///d:/SEM%207/VMplacement/ml/tcn_risk.py) | 30.4 KB | **VERIFIED** | End-to-end calibration, test inference, export & 17 verification tests |
| **Frozen Calibration JSON** | [`results/stage5/risk/risk_calibration.json`](file:///d:/SEM%207/VMplacement/results/stage5/risk/risk_calibration.json) | 1.15 KB | **FROZEN** | $V_{\max}=4.6754\%$, $S_{\max}=3.3824\%$, $\tau_{\text{low}}=0.0295$, $\tau_{\text{high}}=0.1746$ |
| **Placement Input Table** | [`results/stage5/risk/risk_state.csv`](file:///d:/SEM%207/VMplacement/results/stage5/risk/risk_state.csv) | 720.8 KB | **GENERATED** | Master placement input table (6,915 rows, 15 columns, zero future actuals) |
| **Person 3 README Guide** | [`results/stage5/risk/README.md`](file:///d:/SEM%207/VMplacement/results/stage5/risk/README.md) | 5.94 KB | **WRITTEN** | Integration guide, schema dictionary, Python/Java snippets & contract |
| **Verification Suite Report** | [`results/stage5/risk/risk_verification.json`](file:///d:/SEM%207/VMplacement/results/stage5/risk/risk_verification.json) | 1.60 KB | **PASSED** | 17/17 automated acceptance tests passed (`PASS`) |

---

## 2. Frozen Validation Calibration Results

In strict adherence to zero-leakage protocols, all normalization parameters and categorical risk boundaries were calibrated **strictly on the 6,557 validation sliding windows** using the locked champion model ([`M3_FULL_standard_best.pt`](file:///d:/SEM%207/VMplacement/results/stage5/checkpoints/M3_FULL_standard_best.pt)):

```text
================================================================================
FROZEN VALIDATION RISK CALIBRATION CONSTANTS (results/stage5/risk/risk_calibration.json)
================================================================================
Calibration Split:                Validation only (6,557 sliding windows)
Lookback (L) / Horizon (H):       288 steps (24.0 h) / 12 steps (1.0 h)
Historical Volatility Window (K): 24 steps (2.0 h), sample standard deviation (ddof=1)
Risk Weights:                     0.60 * Volatility_norm + 0.40 * Spread_norm

- Volatility 95th Percentile (V_max):      4.675352% CPU
- Forecast Spread 95th Percentile (S_max):   3.382352% CPU
- Validation RiskScore Tertile 1 (tau_low, p33.3):   0.029545
- Validation RiskScore Tertile 2 (tau_high, p66.7):  0.174569
================================================================================
```

### Mathematical Risk Formulation
$$\text{RiskScore} = 0.60 \cdot \min\left(1.0, \frac{\text{volatility\_score}}{4.675352}\right) + 0.40 \cdot \min\left(1.0, \frac{\text{forecast\_spread}}{3.382352}\right)$$

$$\text{risk\_state} = \begin{cases} \text{LOW} & \text{if } \text{RiskScore} < 0.029545 \\ \text{MEDIUM} & \text{if } 0.029545 \le \text{RiskScore} < 0.174569 \\ \text{HIGH} & \text{if } \text{RiskScore} \ge 0.174569 \end{cases}$$

Zero test-partition data was accessed or utilized during calibration.

---

## 3. Placement Input Artifact (`risk_state.csv`) Structure

`risk_state.csv` acts as the **test-time placement input / online-simulation artifact** for Person 3's Adaptive HO algorithm.

### Key Dimensions & Quality Checks
- **Total Rows**: Exactly **6,915** (matching 1:1 the 6,915 test sliding windows from Phase 9).
- **Decision Timestamps**: **997** unique 5-minute decision epochs.
- **Active VMs**: Exactly **7** long-running VMs (`VM_001`, `VM_003`, `VM_011`, `VM_012`, `VM_013`, `VM_023`, `VM_025`).
- **Columns**: Exactly **15 approved columns in order**:
  1. `vm_id`
  2. `decision_timestamp`
  3. `current_cpu`
  4. `predicted_cpu_t5m`
  5. `predicted_cpu_t10m`
  6. `predicted_cpu_t15m`
  7. `predicted_cpu_t30m`
  8. `predicted_cpu_t45m`
  9. `predicted_cpu_t60m`
  10. `predicted_mean_cpu`
  11. `predicted_peak_cpu`
  12. `predicted_std_cpu`
  13. `volatility_score`
  14. `risk_score`
  15. `risk_state`
- **Data Integrity**: **0** NaN values, **0** Inf values.
- **Physical Value Ranges (Verified via Audit)**:
  - `current_cpu`: $[0.00\%, 94.90\%]$
  - `predicted_cpu_t5m` to `t60m`: $[0.0000\%, 85.4331\%]$ (all $\ge 0.0\%$, 0 negative values)
  - `predicted_mean_cpu`: $[0.0060\%, 80.9573\%]$ (all $\ge 0.0\%$, 0 negative values)
  - `predicted_peak_cpu`: $[0.0346\%, 89.4167\%]$ (all $\ge 0.0\%$, 0 negative values)
  - `predicted_std_cpu`: $[0.0121, 30.7514]$
  - `volatility_score`: $[0.0000, 45.7498]$
  - `risk_score`: $[0.0109, 1.0000]$

### Empirical Risk State Distribution across Test Split

| Risk State | Count | Percentage | Operational Meaning for Person 3 AHO |
| :---: | :---: | :---: | :--- |
| **`LOW`** | 2,060 | 29.79% | Stable, predictable workload. AHO prioritizes **exploitation / consolidation** and standard capacity headroom. |
| **`MEDIUM`** | 2,464 | 35.63% | Moderate turbulence or trend. AHO maintains balanced search and standard safety margin. |
| **`HIGH`** | 2,391 | 34.58% | Elevated turbulence or projected spike. AHO prioritizes **exploration / diversification** and adds safety buffer. |
| **Total** | **6,915** | **100.00%** | |

The resulting test-set risk states are reasonably distributed across LOW, MEDIUM, and HIGH categories, providing Person 3's optimizer with a well-distributed signal for dynamic parameter adaptation. The thresholds were calibrated strictly on historical validation sliding windows; this test distribution represents the resulting empirical outcome.

---

## 4. Final Pre-Git Methodology & Artifact Audit

Prior to Git staging and branch handoff, an exhaustive read-only audit was conducted across 8 core dimensions:

### Audit 1: Negative Prediction Root-Cause & Option B Clipping Audit
- **Root Cause**: The locked TCN model uses an unconstrained linear output head (`nn.Linear(32, 12)`). On the raw ML evaluation split ([`predictions.csv`](file:///d:/SEM%207/VMplacement/results/stage5/predictions/predictions.csv)), 5.99% of point predictions are slightly negative (minimum $-21.78\%$) when predicting idling VMs near $0.0\%$ utilization.
- **Placement Input Protection**: In [`ml/tcn_risk.py`](file:///d:/SEM%207/VMplacement/ml/tcn_risk.py#L319), all placement-facing predictions are clipped via `np.clip(y_hat_test[i], 0.0, 100.0)`.
- **Finding**: In [`risk_state.csv`](file:///d:/SEM%207/VMplacement/results/stage5/risk/risk_state.csv), `predicted_mean_cpu` has an empirical minimum of **`+0.0060%`** (never negative). The $-0.07\%$ mentioned in discussions was a documentation typo from an unclipped draft calculation.
- **Recommendation**: **OPTION B** (Clip placement-facing columns to $[0, 100]\%$ for physical host simulation while leaving scientific benchmark [`predictions.csv`](file:///d:/SEM%207/VMplacement/results/stage5/predictions/predictions.csv) 100% raw and unclipped). `[APPROVAL REQUIRED]`.

### Audit 2: Risk-State Distribution Wording
- Scientifically clarified that the test split was not forced into equal tertiles. Tertiles were calibrated strictly on validation, resulting in an empirical test distribution of 29.8% LOW, 35.6% MEDIUM, and 34.6% HIGH.

### Audit 3: Calibration Isolation & Zero-Leakage Data Flow
- $V_{\max}$, $S_{\max}$, $\tau_{\text{low}}$, and $\tau_{\text{high}}$ were computed exclusively on the 6,557 validation sliding windows.
- Zero test data was involved in calibration, threshold selection, normalization, or formula fitting.

### Audit 4: Person 3 Interface & Boundary Contract
- Person 3 receives **strictly 15 causal columns**: telemetry at time $t$, 6-step horizon forecasts, summary forecasts, volatility, risk score, and risk state.
- Person 3 does **NOT** receive future actual CPU values, prediction errors, residuals, or ground truth.
- Person 3 owns: VM-to-PM mapping, HO/AHO algorithms, fitness function, parameter adaptation, host power models, and `placement.csv`.
- Person 1 does **NOT** implement placement policies, PM buffers, or AHO adaptation logic.

### Audit 5: Causality Sample Verification (Row 0)
- Cross-referenced Row 0 of [`risk_state.csv`](file:///d:/SEM%207/VMplacement/results/stage5/risk/risk_state.csv) with canonical telemetry for `VM_001` at timestamp `1378604100` (index 7631):
  - `current_cpu` = $0.7500\%$ (Matches canonical trace at $t$).
  - `volatility_score` = $0.0560\%$ (Matches sample standard deviation `ddof=1` over $[t-23, t]$).
  - TCN lookback slice: exactly 288 steps $[t-287, t]$. Max timestamp $= 1378604100 \le t$.
  - Spread $= 0.1587\%$, $v_{\text{norm}} = 0.0120$, $s_{\text{norm}} = 0.0469 \implies \text{RiskScore} = 0.0259 \implies \text{LOW}$.
  - Future actuals at $t+5\text{m}$ ($0.6333\%$) and $t+60\text{m}$ ($0.5500\%$) are strictly excluded from `risk_state.csv`.

### Audit 6: Reproducibility
- The entire pipeline is deterministically reproducible on any CPU/GPU environment by running:
  ```powershell
  python ml/tcn_risk.py
  ```
- Re-executes in under 90 seconds, re-verifies calibration, re-generates `risk_state.csv`, and validates all 17 acceptance checks.

### Audit 7: Short-Lived VMs Audit
- Seven short-lived VMs (`VM_002`, `VM_004`, `VM_006`, `VM_014`, `VM_018`, `VM_161`, `VM_1209`) have active test partition lengths $< 300$ steps and yield 0 test sliding windows.
- In accordance with Phase 10 design rules, they are excluded without synthetic padding or fabricated predictions.
- Exactly 7 active VMs appear in `risk_state.csv`.

### Audit 8: Acceptance Test Suite (17 / 17 Passed)
- All 17 automated acceptance tests pass:
  1. `risk_state.csv` exists and non-empty `[PASS]`
  2. Row count equals exactly 6,915 `[PASS]`
  3. Schema matches approved 15 columns in order `[PASS]`
  4. Zero NaN values across table `[PASS]`
  5. Zero Inf values across table `[PASS]`
  6. Exactly 7 active VMs represented `[PASS]`
  7. Exactly 7 short-lived VMs excluded without fabrication `[PASS]`
  8. Exactly 997 decision timestamps `[PASS]`
  9. Zero future actuals in placement table `[PASS]`
  10. All 6 horizon predictions present and finite `[PASS]`
  11. `predicted_mean_cpu` equals arithmetic mean `[PASS]`
  12. `predicted_peak_cpu` equals maximum forecast `[PASS]`
  13. `risk_score` bounded strictly in $[0.0, 1.0]$ `[PASS]`
  14. `risk_state` strictly in `{'LOW', 'MEDIUM', 'HIGH'}` `[PASS]`
  15. `risk_calibration.json` exists with frozen parameters `[PASS]`
  16. `README.md` exists with integration guidance `[PASS]`
  17. Overall verification status is `PASS` `[PASS]`

---

## 5. Git Handoff Inventory & Staging Instructions

### Exact Files TO Commit
```text
# Pipeline Code Modules
ml/temporal_windows.py
ml/tcn_model.py
ml/tcn_dataset.py
ml/tcn_train.py
ml/tcn_evaluate.py
ml/tcn_risk.py

# Architecture & Interface Specifications
docs/phase10_ml_to_aho_interface_design.md
docs/risk_state_schema.md

# Stage 5 Model Checkpoints & Scientific Benchmark
results/stage5/checkpoints/M3_FULL_standard_best.pt
results/stage5/ablation/ablation_summary.csv
results/stage5/ablation/ablation_summary.json
results/stage5/predictions/predictions.csv
results/stage5/predictions/per_vm_test_metrics.csv
results/stage5/predictions/horizon_metrics.csv
results/stage5/predictions/evaluation_summary.json
results/stage5/figures/stage5_actual_vs_predicted_representative_vms.png
results/stage5/figures/stage5_test_prediction_error_distribution.png
results/stage5/figures/stage5_test_accuracy_degradation_vs_horizon.png

# Workload Intelligence & Placement Handoff Artifacts
results/stage5/risk/risk_state.csv
results/stage5/risk/risk_calibration.json
results/stage5/risk/risk_verification.json
results/stage5/risk/README.md

# Project Configuration & Comprehensive Walkthroughs
requirements.txt
walkthrough_stage1.md
walkthrough_stage2.md
walkthrough_stage3.md
walkthrough_stage4.md
walkthrough_stage5.md
walkthrough_phase11.md
walkthrough_phase11_workload_intelligence.md
```

### Exact Files NOT to Commit
```text
dataset/                              # Raw Bitbrains traces (13 GB+)
stage5_ablation_results.zip           # Redundant temporary archive
colab_stage5_gpu/                     # Colab-specific packaging scripts
.vscode/                              # Local IDE settings
**/__pycache__/                       # Python bytecode
*.log                                 # System execution logs
.system_generated/                    # Tool scratch directories
scratch/                              # Temporary inspection scripts
```

### Recommended Commit Message
```text
feat(ml): complete Person 1 workload forecasting and risk-state handoff interface

- Finalize TCNForecaster (M3_FULL_standard) test evaluation (MAE 1.7179% CPU, RMSE 9.5368% CPU)
- Implement Stage 5.5 workload intelligence and prediction-risk pipeline (ml/tcn_risk.py)
- Calibrate risk proxy on 6,557 validation sliding windows (V_max=4.6754%, S_max=3.3824%)
- Generate zero-leakage test placement input table (results/stage5/risk/risk_state.csv, 6,915 rows)
- Clip placement-facing predicted CPU values to physical [0, 100]% range (Option B)
- Document Person 3 AHO integration contract, schema, and Python/Java examples (README.md)
- Verify 17/17 automated regression acceptance checks (results/stage5/risk/risk_verification.json)
```

---

## 6. Verdict

```text
================================================================================
PERSON 1 PHASE 11 STATUS: READY FOR GIT HANDOFF
================================================================================
```
