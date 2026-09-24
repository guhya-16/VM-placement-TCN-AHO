# Person 1 → Person 3 contract

`risk_state.csv` is the only workload/risk handoff required from Person 1.

Required header:

```csv
vm_id,decision_timestamp,current_cpu,predicted_cpu_t5m,predicted_cpu_t10m,predicted_cpu_t15m,predicted_cpu_t30m,predicted_cpu_t45m,predicted_cpu_t60m,predicted_mean_cpu,predicted_peak_cpu,predicted_std_cpu,volatility_score,risk_score,risk_state
```

Rules:
- `decision_timestamp` is preserved as Unix seconds (`long`).
- One row = one VM at one decision timestamp.
- All rows with the same timestamp form one `DecisionEpoch`.
- Active VM IDs are taken from the file; there is no fixed VM count.
- Actual future CPU/error columns are not required by Person 3 and are not used.
- `current_cpu` and forecast CPU values are interpreted with the configured per-value `cpu.scale=auto` rule: values in [0,1] are treated as fractions and larger values as percentages. This matches the existing Person 3 branch behavior used during development.
- `predicted_mean_cpu` is used as the representative `predicted_cpu_utilization` in the candidate handoff.
- `risk_score` and `risk_state` remain Person 1's supplied risk intelligence; Person 3 does not recompute them.
