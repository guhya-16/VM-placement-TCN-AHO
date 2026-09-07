# Walkthrough — Stage 3: Preprocessing Policy Evaluation

## Objective & Experimental Context

The objective of Stage 3 is to conduct a controlled, empirical evaluation of three candidate preprocessing policies on a representative cohort of 14 Bitbrains fastStorage VM traces before establishing a finalized preprocessing pipeline for downstream Temporal Convolutional Network (TCN) CPU workload prediction.

> 📌 **Methodological Disclaimer**
>
> The selected 14 VMs are representative cases covering distinct workload, lifecycle, and data-quality regimes across the cluster. They are not claimed to be statistically representative of all 1,250 VMs in the Bitbrains dataset. All evaluations are strictly non-destructive; raw CSV files remain read-only and unmodified.

---

## Visual Summary: Workload Regimes & Policy Comparisons

### 1. Representative Workload Regimes
![Type A Raw CPU Timeline](results/preprocessing_evaluation/figures/type_a_raw_cpu.png)
*Figure 1: Continuous 30-day operational profile (Type A — `1.csv`) exhibiting dynamic, bursty CPU load with 100% continuous provisioned capacity.*

![Type B Raw CPU Timeline](results/preprocessing_evaluation/figures/type_b_raw_cpu.png)
*Figure 2: Representative Type B trace (`2.csv`) exhibiting unallocated prefix rows, a 1.89-day active lifecycle, and 11,260 zero-capacity tail rows post-decommissioning.*

---

### 2. Transition Boundary & Sampling Distributions
![Active-Inactive Transition](results/preprocessing_evaluation/figures/active_inactive_transition.png)
*Figure 3: Zoomed transition boundary in `2.csv` showing the exact drop of CPU usage and provisioned memory capacity on August 29, 2013.*

![Timestamp Intervals Distribution](results/preprocessing_evaluation/figures/timestamp_intervals.png)
*Figure 4: Sampling interval distributions before preprocessing — contrasting Type A's tight 300s clustering against Type B's zero-delta duplicates and staggered polling.*

---

### 3. Policy Performance & Alignment Effects
![Policy Comparison](results/preprocessing_evaluation/figures/policy_comparison.png)
*Figure 5: Usable time-contiguous TCN windows ($W=300$ steps / 25h) across Policies A, B, and C.*

![Active vs Aligned Example](results/preprocessing_evaluation/figures/active_vs_aligned_example.png)
*Figure 6: Granular zoom comparing raw active step jitter against the canonical 5-minute aligned grid in `2.csv`.*

---

## 1. Selected Representative Cohort (14 VMs)

| File | Operational Category | Rationale & Workload Characteristics |
| :--- | :--- | :--- |
| **`1.csv`** | Type A (Continuous) | High-peak bursty utilization (Mean: 4.02%, Max: 97.87%, 0% zeros, 30 days) |
| **`3.csv`** | Type A (Continuous) | Moderate steady utilization (Mean: 3.26%, Max: 9.47%, 0% zeros, 30 days) |
| **`11.csv`** | Type A (Continuous) | Medium utilization with periodic spikes (Mean: 2.53%, Max: 54.33%, 30 days) |
| **`12.csv`** | Type A (Continuous) | Low steady utilization (Mean: 1.01%, Max: 10.73%, 30 days) |
| **`13.csv`** | Type A (Continuous) | Idle operational VM with 100% capacity provisioned (Mean: 0.01%, Max: 2.00%) |
| **`23.csv`** | Type A (Continuous) | Higher sustained workload (Mean: 5.54%, Max: 55.73%, 30 days) |
| **`25.csv`** | Type A (Continuous) | Heavy sustained workload (Mean: 5.17%, Max: 62.57%, 30 days) |
| **`2.csv`** | Type B (Decommissioned) | De-provisioned on Aug 29; 3,800 duplicate timestamps and tailing zero-capacity rows |
| **`4.csv`** | Type B (Decommissioned) | De-provisioned on Aug 29; residual memory telemetry (13,981 KB constant) |
| **`6.csv`** | Type B (Decommissioned) | Baseline Type B de-provisioned node; 3,800 duplicate timestamps |
| **`14.csv`** | Type B (Decommissioned) | Baseline Type B de-provisioned node; 3,800 duplicate timestamps |
| **`18.csv`** | Type B (Decommissioned) | Baseline Type B de-provisioned node; 3,800 duplicate timestamps |
| **`161.csv`** | Transient Lifecycle | Short active lifecycle (~2.2 days, 1,256 rows, 15 duplicate timestamps) |
| **`1209.csv`** | Ephemeral Workload | Sub-day short execution (~9.7 hours, 117 rows, 0 duplicates) |

---

## 2. Candidate Active-Lifetime Rule Evaluation

The candidate operational definition evaluated was:
$$\text{active} = (\text{memory\_capacity\_kb} > 0) \land (\text{cpu\_capacity\_mhz} > 0)$$

### Empirical Results Across the 14 VMs

| File | Category | Total Rows | Active Rows | Inactive Prefix | Inactive Suffix | Contiguous? | Active Days | Observed Infrastructure State |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `1.csv` | Type A | 8,634 | 8,634 | 0 | 0 | **Yes** | 30.00 | Fully provisioned throughout |
| `3.csv` | Type A | 8,634 | 8,634 | 0 | 0 | **Yes** | 30.00 | Fully provisioned throughout |
| `11.csv` | Type A | 8,635 | 8,635 | 0 | 0 | **Yes** | 30.00 | Fully provisioned throughout |
| `12.csv` | Type A | 8,635 | 8,635 | 0 | 0 | **Yes** | 30.00 | Fully provisioned throughout |
| `13.csv` | Type A | 8,635 | 8,635 | 0 | 0 | **Yes** | 30.00 | Fully provisioned throughout |
| `23.csv` | Type A | 8,614 | 8,614 | 0 | 0 | **Yes** | 29.93 | Fully provisioned throughout |
| `25.csv` | Type A | 8,614 | 8,614 | 0 | 0 | **Yes** | 29.93 | Fully provisioned throughout |
| `2.csv` | Type B | 16,142 | 544 | 4,338 | 11,260 | **Yes** | 1.89 | Unallocated prefix (4,338 rows); Post-decommissioning tail (11,260 rows) |
| `4.csv` | Type B | 16,139 | 543 | 4,336 | 11,260 | **Yes** | 1.89 | Unallocated prefix (4,336 rows); Post-decommissioning tail (11,260 rows) |
| `6.csv` | Type B | 16,141 | 543 | 4,338 | 11,260 | **Yes** | 1.89 | Unallocated prefix (4,338 rows); Post-decommissioning tail (11,260 rows) |
| `14.csv` | Type B | 16,143 | 544 | 4,339 | 11,260 | **Yes** | 1.89 | Unallocated prefix (4,339 rows); Post-decommissioning tail (11,260 rows) |
| `18.csv` | Type B | 16,143 | 544 | 4,339 | 11,260 | **Yes** | 1.89 | Unallocated prefix (4,339 rows); Post-decommissioning tail (11,260 rows) |
| `161.csv` | Transient | 1,256 | 31 | 1 | 1,224 | **Yes** | 0.10 | Unallocated prefix (1 row); Post-decommissioning tail (1,224 rows) |
| `1209.csv` | Transient | 117 | 117 | 0 | 0 | **Yes** | 0.40 | Clean contiguous ephemeral task |

> 📌 **Key Takeaway on Candidate Rule**
>
> The candidate provisioning rule produced **100% contiguous intervals** across all 14 files without a single internal zero-capacity flicker.
> 
> In Type B VMs, the rule revealed that the VMs were actively provisioned for **1.89 days** (544 observations from August 27 15:27 UTC to August 29 12:48 UTC). The preceding 4,338 rows and succeeding 11,260 rows represent unallocated infrastructure recorded by the cluster monitoring system.

---

## 3. Timestamp Jitter & Gap Analysis (Active Lifetime)

Within the identified active operational windows, empirical measurements of timestamp jitter and missing steps revealed:

| File | Active Rows | Active Dupes | $\le 1\text{ s}$ Jitter | $\le 5\text{ s}$ Jitter | $\le 30\text{ s}$ Jitter | Normal Steps (~300s) | 1-Step Gaps (~600s) | 2-Step Gaps (~900s) | Sub-300s Steps |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1.csv` | 8,634 | 0 | 99.92% | 99.93% | 99.94% | 8,628 | 4 | 1 | 0 |
| `3.csv` | 8,634 | 0 | 99.94% | 99.95% | 99.95% | 8,629 | 3 | 0 | 0 |
| `11.csv` | 8,635 | 0 | 99.93% | 99.94% | 99.95% | 8,630 | 3 | 1 | 0 |
| `12.csv` | 8,635 | 0 | 99.93% | 99.94% | 99.95% | 8,630 | 3 | 1 | 0 |
| `13.csv` | 8,635 | 0 | 99.93% | 99.94% | 99.95% | 8,630 | 3 | 1 | 0 |
| `23.csv` | 8,614 | 0 | 99.77% | 99.78% | 99.79% | 8,595 | 11 | 6 | 0 |
| `25.csv` | 8,614 | 0 | 99.79% | 99.80% | 99.81% | 8,597 | 11 | 1 | 0 |
| `2.csv` | 544 | 0 | 99.82% | 99.82% | 99.82% | 542 | 1 | 0 | 0 |
| `4.csv` | 543 | 0 | 99.63% | 99.63% | 99.63% | 540 | 2 | 0 | 0 |
| `6.csv` | 543 | 0 | 99.63% | 99.63% | 99.63% | 540 | 2 | 0 | 0 |
| `14.csv` | 544 | 0 | 99.82% | 99.82% | 99.82% | 542 | 1 | 0 | 0 |
| `18.csv` | 544 | 0 | 99.82% | 99.82% | 99.82% | 542 | 1 | 0 | 0 |
| `161.csv` | 31 | 0 | 100.0% | 100.0% | 100.0% | 30 | 0 | 0 | 0 |
| `1209.csv` | 117 | 0 | 100.0% | 100.0% | 100.0% | 116 | 0 | 0 | 0 |

### Forensic Discoveries on Jitter & Gaps
1. **Zero Active-Lifetime Duplicates**: During active execution, there are **0 duplicate timestamps** in all Type A and Type B traces. All 3,800 duplicates occurred strictly in the post-decommissioning tail.
2. **Extreme Jitter Tightness**: Across all active observations, **$>99.7\%$ of steps are within $\pm 1$ second of 300s** ($299\text{ s}$ to $301\text{ s}$).
3. **Missing Steps are Rare & Isolated**: Across an entire month, Type A traces contain only 3 to 11 single missing steps ($600\text{ s}$) and 0 to 6 double missing steps ($900\text{ s}$).
4. **Zero Sub-300s Intervals in Active Period**: Sub-300s intervals ($2\text{ s}$, $102\text{ s}$) never occur while VMs are provisioned.

---

## 4. Synthetic Interpolation & Volatility Impact Analysis

A central question in evaluating Policy C is whether inserting interpolated values materially distorts workload statistics or dampens peak volatility.

### Exact Quantification of Synthetic Points & Distribution Impact

| File | Policy B Obs | Policy C Obs | Point Increase | Interpolated Points | % Synthetic Points | Policy B Mean [%] | Policy C Mean [%] | $\Delta$ Mean | Policy B Std [%] | Policy C Std [%] | $\Delta$ Std Dev |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1.csv` | 8,634 | 8,640 | +6 | 6 | **0.069%** | 4.023 | 4.028 | +0.0057 | 16.904 | 16.913 | +0.0096 |
| `3.csv` | 8,634 | 8,640 | +6 | 5 | **0.058%** | 3.255 | 3.255 | -0.0000 | 0.511 | 0.510 | -0.0001 |
| `11.csv` | 8,635 | 8,640 | +5 | 5 | **0.058%** | 2.527 | 2.526 | -0.0004 | 4.448 | 4.447 | -0.0012 |
| `12.csv` | 8,635 | 8,640 | +5 | 5 | **0.058%** | 1.008 | 1.008 | -0.0000 | 0.199 | 0.198 | -0.0001 |
| `13.csv` | 8,635 | 8,640 | +5 | 5 | **0.058%** | 0.009 | 0.009 | -0.0000 | 0.107 | 0.107 | -0.0000 |
| `23.csv` | 8,614 | 8,640 | +26 | 25 | **0.289%** | 5.537 | 5.538 | +0.0005 | 1.868 | 1.867 | -0.0017 |
| `25.csv` | 8,614 | 8,640 | +26 | 21 | **0.243%** | 5.168 | 5.170 | +0.0012 | 1.885 | 1.884 | -0.0007 |
| `2.csv` | 544 | 545 | +1 | 1 | **0.183%** | 7.143 | 7.131 | -0.0126 | 23.232 | 23.213 | -0.0195 |
| `4.csv` | 543 | 545 | +2 | 2 | **0.367%** | 7.126 | 7.100 | -0.0252 | 23.150 | 23.111 | -0.0388 |
| `6.csv` | 543 | 545 | +2 | 2 | **0.367%** | 7.207 | 7.181 | -0.0255 | 23.316 | 23.277 | -0.0391 |
| `14.csv` | 544 | 545 | +1 | 1 | **0.183%** | 7.149 | 7.137 | -0.0127 | 23.197 | 23.177 | -0.0194 |
| `18.csv` | 544 | 545 | +1 | 1 | **0.183%** | 7.154 | 7.141 | -0.0127 | 23.252 | 23.232 | -0.0195 |
| `161.csv` | 31 | 31 | 0 | 0 | **0.000%** | 5.289 | 5.289 | 0.0000 | 12.294 | 12.294 | 0.0000 |
| `1209.csv` | 117 | 117 | 0 | 0 | **0.000%** | 0.970 | 0.970 | 0.0000 | 0.653 | 0.653 | 0.0000 |
| **Total** | **63,267** | **63,353** | **+86** | **84** | **0.133%** | — | — | — | — | — | — |

### Key Findings on Synthetic Impact
1. **Negligible Volume of Synthetic Data**: Across all 14 VMs (63,353 observations in Policy C), **only 84 points are synthetic/interpolated** ($<0.13\%$ overall). Over 99.86% of all points in Policy C are 100% empirical measurements.
2. **No Material Distortion of Volatility**: The change in CPU standard deviation ($\Delta \text{Std}$) is strictly $\le 0.039\%$, and the change in mean ($\Delta \text{Mean}$) is $\le 0.026\%$ across all files. Linear interpolation over isolated 1-step gaps does **not** dampen workload dynamics or artificially suppress volatility.

---

## 5. Explanation of the $4,519 \rightarrow 4,525$ Observation Increase

The mean observation count across the 14 VMs increases from **4,519.1 in Policy B** to **4,525.2 in Policy C** ($+6.1$ observations per VM on average).

* **Mathematical & Empirical Explanation**:
  * In Policy B, the observation count is strictly the number of raw lines logged in the active window. In an ideal 30-day trace at 5-minute intervals, there are:
    $$\frac{2,591,952\text{ seconds}}{300\text{ seconds/interval}} = 8,639.84 \approx 8,640\text{ canonical intervals}$$
  * In `1.csv`, only 8,634 raw rows exist because 6 discrete intervals were never recorded (4 single 600s gaps and 1 two-step 900s gap).
  * In Policy C, when resampling onto a canonical 5-minute calendar grid from start to end, the grid establishes a cell for every 5-minute interval ($8,640$ total cells). The 6 unrecorded intervals are created as grid cells and filled.
  * Across all 14 VMs, this creates an average increase of $+6.1$ grid cells, precisely accounting for the difference.

---

## 6. Actual Usable Windows Gained by Interpolation

Sliding window continuity requires that every consecutive observation within a window $[i, i+W-1]$ be contiguous ($\Delta t \in [270, 330]\text{ s}$). When a single 600s gap occurs in an unaligned trace (Policy B), it invalidates $W-1 = 299$ candidate overlapping windows.

### Window Availability Comparison ($W=300$ steps / 25 Hours)

| File | Category | Policy B Time-Contiguous Windows | Policy C Time-Contiguous Windows | Windows Gained by Interpolation | Relative Gain |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `1.csv` | Type A | 7,048 | 8,341 | **+1,293** | +18.3% |
| `3.csv` | Type A | 7,139 | 8,341 | **+1,202** | +16.8% |
| `11.csv` | Type A | 7,140 | 8,341 | **+1,201** | +16.8% |
| `12.csv` | Type A | 7,140 | 8,341 | **+1,201** | +16.8% |
| `13.csv` | Type A | 7,140 | 8,341 | **+1,201** | +16.8% |
| `23.csv` | Type A | 6,094 | 8,341 | **+2,247** | +36.9% |
| `25.csv` | Type A | 6,218 | 8,341 | **+2,123** | +34.1% |
| `2.csv` | Type B | 24 | 246 | **+222** | +925.0% |
| `4.csv` | Type B | 0 | 246 | **+246** | *From 0 to viable* |
| `6.csv` | Type B | 0 | 246 | **+246** | *From 0 to viable* |
| `14.csv` | Type B | 24 | 246 | **+222** | +925.0% |
| `18.csv` | Type B | 24 | 246 | **+222** | +925.0% |
| `161.csv` | Transient | 0 | 0 | 0 | Trace < 25h |
| `1209.csv` | Transient | 0 | 0 | 0 | Trace < 25h |
| **Total** | — | **47,991** | **59,617** | **+11,626** | **+24.2%** |

> 📌 **Key Takeaway on Window Gains**
>
> By interpolating just **84 isolated points** ($<0.13\%$), Policy C recovers **+11,626 usable training windows** ($+24.2\%$). In traces like `4.csv` and `6.csv`, two isolated missing steps in Policy B split the 543-step active duration into sub-segments shorter than 300 steps, rendering the VM completely unusable ($0$ windows). Policy C restores both traces to 246 valid windows.

---

## 7. Short-Lived VMs Multi-Scale Window Analysis

Rather than prematurely imposing an arbitrary 48-hour filtering threshold, we evaluated window production across multiple candidate model scales:
* **$W = 300$ steps** (25 hours: 24h input + 1h horizon)
* **$W = 72$ steps** (6 hours: 5h input + 1h horizon)
* **$W = 36$ steps** (3 hours: 2.5h input + 30m horizon)

| File | Active Observations | Active Duration | Usable Windows ($W=300$) | Usable Windows ($W=72$) | Usable Windows ($W=36$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`161.csv`** | 31 | 2.58 hours | 0 | 0 | 0 |
| **`1209.csv`** | 117 | 9.75 hours | 0 | **46** | **82** |
| **Type B (`2.csv`)** | 544 | 45.3 hours | **246** | **474** | **510** |

* **Empirical Observation**: `1209.csv` is fully viable if an agile, sub-daily TCN model ($W \le 72$ steps) is trained, producing 46 to 82 continuous windows. `161.csv` (31 observations) is too brief for models requiring $W \ge 36$ steps, but yields 8 windows for ultra-short $W=24$ steps (2 hours).

---

## 8. Summary Comparison of the Three Policies

| Characteristic | Policy A (RAW) | Policy B (ACTIVE ONLY) | Policy C (ACTIVE + ALIGNED) |
| :--- | :---: | :---: | :---: |
| **Data Retention** | 100.0% | 58.5% | 58.6% |
| **Synthetic Observations** | **0** | **0** | **84** ($0.13\%$) |
| **Duplicate Timestamps** | 19,015 | **0** | **0** |
| **Mean CPU Zero Fraction** | 48.57% (severely distorted) | 10.93% (realistic) | 10.93% (realistic) |
| **Mean CPU Volatility ($\sigma$)** | Artificially damped | 7.64% (preserved) | 7.64% (preserved) |
| **Total Usable Windows ($W=300$)** | 66,488 (corrupted in Type B) | 47,991 | **59,617** |
| **Viability for TCN Sequences** | **Infeasible** | **Marginal** (loses windows at gaps) | **Optimal** (continuous grid) |

---

## 9. Final Preprocessing Decision Framework

With all 7 empirical validation tasks completed, the evidence clearly establishes:
1. **Active-Lifetime Filtering is Mandatory**: Retaining unallocated prefix/suffix rows injects 96.6% flat zero telemetry into Type B traces and produces zero-delta duplicates. Truncating to active infrastructure provisioning (`memory_capacity_kb > 0`) is essential.
2. **5-Minute Alignment & Limited Interpolation is Defensible**:
   - Creates only 84 synthetic points across 63,353 observations ($<0.13\%$).
   - Does not alter CPU volatility ($\Delta \sigma \le 0.039\%$).
   - Recovers $+11,626$ usable continuous training windows ($+24.2\%$).
   - Eliminates temporal jitter and aligns observations with canonical cloud placement simulation intervals.

---

## Deliverables & Generated Artifacts

| Category | File Path | Description |
| :--- | :--- | :--- |
| **Evaluation Script** | `ml/preprocessing_evaluation.py` | Standalone script evaluating the candidate rule, jitter, gaps, and 3 policies |
| **Macro Summary** | `results/preprocessing_evaluation/summary.csv` | Macro-level metric comparison averaged across the 14 VMs |
| **Per-VM Breakdown** | `results/preprocessing_evaluation/per_vm_results.csv` | Detailed metric table for each VM under each of the 3 policies |
| **Candidate Rule Data** | `results/preprocessing_evaluation/candidate_rule_evaluation.csv` | Contiguity, prefix/suffix rows, and lifecycle evaluation |
| **Jitter & Gap Data** | `results/preprocessing_evaluation/jitter_and_gap_analysis.csv` | Step-to-step jitter and gap distribution analysis |
| **Interpolation & Volatility** | `results/preprocessing_evaluation/interpolation_and_volatility_impact.csv` | Point-by-point synthetic point counts, $\Delta \sigma$, and window gains |
| **Comprehensive Report** | `results/preprocessing_evaluation/report.txt` | Complete 400-line scientific forensic evaluation report |
| **Figure 1 (Type A)** | `results/preprocessing_evaluation/figures/type_a_raw_cpu.png` | Raw CPU timeline for Type A continuous VM (`1.csv`) |
| **Figure 2 (Type B)** | `results/preprocessing_evaluation/figures/type_b_raw_cpu.png` | Raw CPU timeline for Type B showing unallocated tail (`2.csv`) |
| **Figure 3 (Transition)** | `results/preprocessing_evaluation/figures/active_inactive_transition.png` | Zoomed active $\rightarrow$ inactive transition boundary |
| **Figure 4 (Intervals)** | `results/preprocessing_evaluation/figures/timestamp_intervals.png` | Pre-processing interval distributions for Type A vs Type B |
| **Figure 5 (Comparison)** | `results/preprocessing_evaluation/figures/policy_comparison.png` | Bar comparison of usable time-contiguous TCN windows |
| **Figure 6 (Alignment)** | `results/preprocessing_evaluation/figures/active_vs_aligned_example.png` | Granular zoom of raw active jitter vs 5-minute aligned grid |
