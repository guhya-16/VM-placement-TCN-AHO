# Risk State Artifact Schema & Data Dictionary (Revised)
**File Path**: `results/stage5/risk/risk_state.csv`  
**Classification**: **TEST-TIME PLACEMENT INPUT / ONLINE-SIMULATION ARTIFACT**  
**Producer**: Person 1 (ML Workload Forecasting Engine)  
**Consumer**: Person 3 (Adaptive Hippopotamus Optimization - AHO)  
**Granularity**: Exactly one row per active VM per decision timestamp $t$  
**Total Rows (Test Split)**: Exactly 6,915 rows (from the 7 active VMs possessing $W \ge 300$-step test windows)  

---

## 1. Column Specifications & Interface Contract

| Column Index | Column Name | Data Type | Physical Units | Valid Range | Nullable? | Description / Optimization Role |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| 1 | `vm_id` | `string` | — | `VM_001` .. `VM_025` | **NO** | Canonical VM identifier for placement optimization. |
| 2 | `decision_timestamp` | `int64` | Unix seconds | 5-min grid | **NO** | Cutoff time $t$ when the placement decision is executed. |
| 3 | `current_cpu` | `float32` | % CPU | $[0.0, 100.0]$ | **NO** | Observed CPU utilization at decision time $t$ ($x_t$). |
| 4 | `predicted_cpu_t5m` | `float32` | % CPU | $[0.0, 100.0]$ | **NO** | Predicted CPU utilization at $t + 5\text{ min}$ ($h=1$). |
| 5 | `predicted_cpu_t10m` | `float32` | % CPU | $[0.0, 100.0]$ | **NO** | Predicted CPU utilization at $t + 10\text{ min}$ ($h=2$). |
| 6 | `predicted_cpu_t15m` | `float32` | % CPU | $[0.0, 100.0]$ | **NO** | Predicted CPU utilization at $t + 15\text{ min}$ ($h=3$). |
| 7 | `predicted_cpu_t30m` | `float32` | % CPU | $[0.0, 100.0]$ | **NO** | Predicted CPU utilization at $t + 30\text{ min}$ ($h=6$). |
| 8 | `predicted_cpu_t45m` | `float32` | % CPU | $[0.0, 100.0]$ | **NO** | Predicted CPU utilization at $t + 45\text{ min}$ ($h=9$). |
| 9 | `predicted_cpu_t60m` | `float32` | % CPU | $[0.0, 100.0]$ | **NO** | Predicted CPU utilization at $t + 60\text{ min}$ ($h=12$). |
| 10 | `predicted_mean_cpu` | `float32` | % CPU | $[0.0, 100.0]$ | **NO** | Average predicted CPU over the 1-hour horizon ($\frac{1}{12}\sum \hat{y}$). For **energy consumption estimation**. |
| 11 | `predicted_peak_cpu` | `float32` | % CPU | $[0.0, 100.0]$ | **NO** | Maximum predicted CPU over the 1-hour horizon ($\max \hat{y}$). For **capacity feasibility & SLA constraints**. |
| 12 | `predicted_std_cpu` | `float32` | % CPU | $\ge 0.0$ | **NO** | Standard deviation of the predicted 12-step trajectory path. |
| 13 | `volatility_score` | `float32` | % CPU | $[0.0, 100.0]$ | **NO** | Sample standard deviation over past 24 steps (2.0 hours). Physical turbulence metric. |
| 14 | `risk_score` | `float32` | Index | $[0.0, 1.0]$ | **NO** | Continuous prediction-risk proxy combining volatility and forecast trajectory spread. |
| 15 | `risk_state` | `string` | Categorical | `LOW`, `MEDIUM`, `HIGH` | **NO** | Discrete policy state derived via Validation tertiles. |

> [!IMPORTANT]
> **Strict Leakage Prevention Guarantee**:
> `risk_state.csv` does **NOT** contain future actual CPU values, residuals, point errors, squared errors, or test error statistics.
> Ground truth test actuals are provided exclusively to CloudSim (Person 2) during simulation execution to evaluate placement quality.

---

## 2. Guidance for Person 3 (How to Ingest in AHO)

### 2.1 Filtering to Current Decision Cycle
At any CloudSim decision cycle $t$:
```python
import pandas as pd

# Load master placement interface table
df_risk = pd.read_csv("results/stage5/risk/risk_state.csv")

# Filter candidate VMs active at the current decision timestamp
vms_at_t = df_risk[df_risk["decision_timestamp"] == current_simulation_timestamp]
```

### 2.2 Semantic Workload Intelligence Mapping
- `predicted_mean_cpu`: Estimated average utilization over the next 1 hour, suitable for evaluating host power consumption in the fitness function.
- `predicted_peak_cpu`: Estimated maximum demand over the next 1 hour, suitable for checking host capacity feasibility and preventing oversubscription.
- `risk_score` & `risk_state`: Semantic indicators of predictive unreliability and workload turbulence:
  - **`LOW`**: Low turbulence, stable trajectory.
  - **`MEDIUM`**: Moderate turbulence and forecast spread.
  - **`HIGH`**: High turbulence, sharp shifts, or wider forecast spread.

> [!NOTE]
> **Ownership Boundary**:
> Person 3 is solely responsible for determining how `risk_state` and `risk_score` influence Adaptive HO exploration/exploitation parameters, PM capacity margins, VM grouping, and placement constraints.
