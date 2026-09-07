# Walkthrough — Stage 2: Forensic Analysis of Timestamp & Duplicate Behavior

## Objective & Scope
Following the completion of Stage 1, an empirical discrepancy was detected between VM trace files:
- **Type A** files (such as `1.csv` and `3.csv`) had 8,634 rows with zero duplicate timestamps and strict 5-minute sampling.
- **Type B** files (such as `2.csv`, `4.csv` through `10.csv`) contained $\approx 16,140$ rows with $\approx 3,800$ duplicate timestamps and a lower median interval ($\approx 209\text{ s}$).

Stage 2 was executed as a forensic data-quality investigation on raw traces (`1.csv`, `2.csv`, `3.csv`, `4.csv`, and cross-verified on `5.csv`–`10.csv`) to determine the root cause, sequence patterns, and resource identities of duplicate observations **without modifying, deduplicating, resampling, or aggregating raw data**.

---

## Visual Summary: Timeline & Duplicate Distribution

![Stage 2 Forensic Timeline Comparison](results/inspection/figures/duplicate_timeline_comparison.png)

---

## Detailed Forensic Analyses

### 1. Duplicate Timestamps Breakdown (Analysis 1)

For each target file, the total row count, unique timestamps, duplicate groups, and exact duplicate rows were computed:

| Metric | `1.csv` (Type A) | `2.csv` (Type B) | `3.csv` (Type A) | `4.csv` (Type B) |
| :--- | :---: | :---: | :---: | :---: |
| **Total Rows** | 8,634 | 16,142 | 8,634 | 16,139 |
| **Unique Timestamps** | 8,634 | 12,342 | 8,634 | 12,339 |
| **Duplicated Timestamp Groups** | 0 | 3,754 | 0 | 3,754 |
| **Rows Belonging to Dupe Groups** | 0 | 7,554 | 0 | 7,554 |
| **Extra Duplicate Timestamps** | 0 | 3,800 | 0 | 3,800 |
| **Exact Duplicate Rows (All Cols)** | 0 | 3,798 | 0 | 3,798 |

In Type B files, the distribution of duplicate occurrences is exact:
- **3,708 timestamps** appear exactly **twice** ($3,708 \times 1 = 3,708$ extra occurrences).
- **46 timestamps** appear exactly **three times** ($46 \times 2 = 92$ extra occurrences).
- Total extra duplicate timestamps: $3,708 + 92 = \mathbf{3,800}$.
- Total rows involved in duplicate groups: $(3,708 \times 2) + (46 \times 3) = \mathbf{7,554}$.

---

### 2. Resource Identity in Duplicate Rows: Case A vs. Case B (Analysis 2)

All resource metrics (`cpu_cores`, `cpu_capacity_mhz`, `cpu_usage_mhz`, `cpu_usage_percent`, `memory_capacity_kb`, `memory_usage_kb`, `disk_read_kbps`, `disk_write_kbps`, `network_received_kbps`, `network_transmitted_kbps`) were compared across duplicate timestamps:

* **Case A (100% Identical Resource Metrics across all columns)**:
  * **3,797 occurrences (99.92%)** in both `2.csv` and `4.csv`.
* **Case B (Different Resource Metrics for the same timestamp)**:
  * **Only 3 occurrences (0.08%)** in both `2.csv` and `4.csv`.

#### Concrete Case B Examples (Transition Boundaries)
The only 3 Case B instances occur at the exact two transition steps when the VM is powered down and decommissioned:

```
Timestamp 1: 1377780792 (2013-08-29 12:53:12 UTC)
  Row 4882: CPU [%]=0.0, CPU [MHz]=0.0, Mem [KB]=44739.2, Mem Cap [KB]=16353280.0, Net Recv [KB/s]=0.0000
  Row 4883: CPU [%]=0.0, CPU [MHz]=0.0, Mem [KB]=27962.0, Mem Cap [KB]=16353280.0, Net Recv [KB/s]=0.0833

Timestamp 2: 1377781092 (2013-08-29 12:58:12 UTC)
  Row 4884: CPU [%]=0.0, CPU [MHz]=0.0, Mem [KB]=27962.0, Mem Cap [KB]=16353280.0, Net Recv [KB/s]=0.0833
  Row 4885: CPU [%]=0.0, CPU [MHz]=0.0, Mem [KB]=0.0,     Mem Cap [KB]=0.0,          Net Recv [KB/s]=0.0000
  Row 4886: CPU [%]=0.0, CPU [MHz]=0.0, Mem [KB]=0.0,     Mem Cap [KB]=0.0,          Net Recv [KB/s]=0.0000
```

> 📌 **Key Observation**
>
> Following row 4885, **provisioned memory capacity drops to 0.0 KB** and CPU usage remains flat at $0.0\%$. For the remaining 11,257 rows (August 29 to September 11), all duplicate rows are 100% identical exact duplicates representing an inactive / de-provisioned VM state.

---

### 3. Timestamp Sequence Pattern (Analysis 3)

Inspection of consecutive rows in `2.csv` around the onset of duplicates:

```text
row_idx | timestamp_raw | timestamp (UTC)          | delta_s | cpu [%] | mem [KB]
--------------------------------------------------------------------------------
   4879 |    1377779892 | 2013-08-29 12:38:12+00:00 |   300.0 |    0.25 |   33554.4
   4880 |    1377780192 | 2013-08-29 12:43:12+00:00 |   300.0 |    0.25 |       0.0
   4881 |    1377780492 | 2013-08-29 12:48:12+00:00 |   300.0 |    0.25 |   44739.2
   4882 |    1377780792 | 2013-08-29 12:53:12+00:00 |   300.0 |    0.00 |   44739.2  <-- Onset
   4883 |    1377780792 | 2013-08-29 12:53:12+00:00 |     0.0 |    0.00 |   27962.0  <-- Dupe (delta 0s)
   4884 |    1377781092 | 2013-08-29 12:58:12+00:00 |   300.0 |    0.00 |   27962.0
   4885 |    1377781092 | 2013-08-29 12:58:12+00:00 |     0.0 |    0.00 |       0.0  <-- Cap drops to 0
   4886 |    1377781092 | 2013-08-29 12:58:12+00:00 |     0.0 |    0.00 |       0.0  <-- Triplet (delta 0s)
   4887 |    1377781392 | 2013-08-29 13:03:12+00:00 |   300.0 |    0.00 |       0.0
   4888 |    1377781392 | 2013-08-29 13:03:12+00:00 |     0.0 |    0.00 |       0.0
```

* **Observed Sequence Types**:
  1. **Clean Period (Rows 0–4,881)**: Uniform sequence: $T, T+300, T+600\dots$ ($\Delta t = 300\text{ s}$).
  2. **Zero-Delta Pairs/Triplets**: Immediately adjacent rows with identical timestamps ($T, T$, $\Delta t = 0\text{ s}$).
  3. **Interleaved Staggered Streams**: In the post-shutdown period, observations alternate between two sampling offsets:
     $$T, T, T+2\text{ s}, T+300\text{ s}, T+300\text{ s}, T+302\text{ s}\dots$$

---

### 4. Duplicates Over Time & Localization (Analysis 4)

* **Temporal Boundary**:
  * **First duplicate timestamp**: `1377780792` (`2013-08-29 12:53:12 UTC`).
  * **Last duplicate timestamp**: `1378906692` (`2013-09-11 13:38:12 UTC`).
* **Localization Pattern**:
  * Duplicates do **not** appear randomly throughout the month.
  * In rows 0 to 4,881 (first 17 days), there is **zero duplicate activity**.
  * The duplicates are strictly confined to the tailing 13 days of the trace file, aligning with the VM de-provisioning event.

---

### 5. Unique-Timestamp Sampling Interval Analysis (Analysis 5)

Interval statistics were evaluated after isolating unique timestamps (in-memory analysis only):

* **Type A Files (`1.csv`, `3.csv`)**:
  * Min: $299.0\text{ s}$, Median: **$300.0\text{ s}$**, Mean: $300.24\text{ s}$, Max: $900.0\text{ s}$
  * Distribution: $300\text{ s}$ ($96.12\%$), $301\text{ s}$ ($3.29\%$), $299\text{ s}$ ($0.51\%$).
* **Type B Files (`2.csv`, `4.csv`) — Split Operational Analysis**:
  * **Active Lifecycle (Rows 0 to 4,881, August 12 – August 29)**:
    * Min: $299.0\text{ s}$, Median: **$300.0\text{ s}$**, Mean: $300.28\text{ s}$, Max: $900.0\text{ s}$
    * Distribution: $300\text{ s}$ ($96.09\%$), $301\text{ s}$ ($3.26\%$), $299\text{ s}$ ($0.53\%$).
    * **Identical sampling properties to Type A.**
  * **Inactive Post-Shutdown (Rows 4,882 to 16,141, August 29 – September 11)**:
    * Median: $195.0\text{ s}$, Mean: $150.96\text{ s}$.
    * Interval pairs sum to $300\text{ s}$:
      * $198\text{ s} + 102\text{ s} = 300\text{ s}$
      * $298\text{ s} + 2\text{ s} = 300\text{ s}$
      * $274\text{ s} + 26\text{ s} = 300\text{ s}$
      * $284\text{ s} + 16\text{ s} = 300\text{ s}$

---

### 6. Cross-File Comparative Synthesis (Analysis 6)

| Trace | Total Rows | Unique Timestamps | Dupe Timestamps | Exact Dupe Rows | Raw Interval Median | Unique TS Interval Median | Active Window |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`1.csv`** | 8,634 | 8,634 | 0 | 0 | 300.0 s | 300.0 s | Entire 30 days |
| **`2.csv`** | 16,142 | 12,342 | 3,800 | 3,798 | 209.0 s | 266.0 s | Days 1–17 (then 0s) |
| **`3.csv`** | 8,634 | 8,634 | 0 | 0 | 300.0 s | 300.0 s | Entire 30 days |
| **`4.csv`** | 16,139 | 12,339 | 3,800 | 3,798 | 209.0 s | 266.0 s | Days 1–17 (then 0s) |
| **`5.csv`–`10.csv`** | $\approx 16,140$ | $\approx 12,340$ | 3,800 | 3,798 | 209.0 s | 266.0 s | Days 1–17 (then 0s) |

---

## Answers to Core Forensic Questions

1. **What was discovered?**  
   The cluster contains two operational VM behaviors: continuous 30-day VMs (Type A) and decommissioned VMs (Type B) that were shut down on August 29 at 12:53 UTC. All duplicate timestamps in the dataset originate from post-shutdown ghost telemetry.
2. **Are duplicate timestamp rows exact duplicates?**  
   **Yes, 99.92% are exact duplicates** across all 10 resource metrics (Case A). Only 3 instances (0.08%) differ across columns (Case B), corresponding to the 2-step capacity ramp-down to zero.
3. **What is the actual timestamp pattern?**  
   Strict uniform 300-second sequence while active. Once shut down, rows exhibit adjacent identical repetitions ($T, T$, $\Delta t = 0\text{ s}$) and interleaved dual-stream polling ($T, T+2\text{ s}, T+300\text{ s}\dots$) with complementary interval pairs summing to 300s.
4. **Does the unique-timestamp timeline follow approximately 5-minute sampling?**  
   **Yes, perfectly during active execution** (median 300.0s, mean 300.28s). Post-shutdown, the timeline is distorted by asynchronous secondary polling on zero-utilization records.
5. **Are duplicates systematic?**  
   **Yes, 100% systematic.** They initiate at the exact same second (`1377780792`) across all Type B files and share identical duplicate counts ($\approx 3,800$).
6. **What explanations are supported by the data?**  
   A datacenter lifecycle event (VM shutdown/decommissioning) occurred on August 29. The telemetry collection daemon continued recording ghost zero-utilization entries from multiple polling threads rather than closing trace files.
7. **What preprocessing decisions need to be made?**  
   - Deduplication: `drop_duplicates(subset=['timestamp_raw'], keep='first')` safely resolves 99.92% of identical rows.
   - Tail handling: Decide whether tailing zero-utilization periods post-shutdown should be truncated to the VM's active lifetime or retained with zero-padding for temporal windows.
   - Grid alignment: Resample unique observations to a strict, uniform 300-second timeline.

---

## Deliverables & Generated Artifacts

| Category | File Path | Description |
| :--- | :--- | :--- |
| **Analysis Script** | `ml/forensic_analysis.py` | Dedicated script executing the 6 forensic analysis modules |
| **Inspection Report** | `results/inspection/duplicate_analysis_report.txt` | Complete forensic data-quality text report |
| **Structured Output** | `results/inspection/duplicate_analysis.json` | Comprehensive machine-readable JSON metrics |
| **Forensic Visualization** | `results/inspection/figures/duplicate_timeline_comparison.png` | 4-panel timeline visualization for Stage 2 |
