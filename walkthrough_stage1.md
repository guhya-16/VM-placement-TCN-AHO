# Walkthrough — Stage 1: Bitbrains Dataset Loader & Inspection

## Summary of Completed Work
Stage 1 has been implemented and tested strictly according to the agreed design rules, keeping raw dataset files read-only and establishing the foundational dataset loader and validation system for Person 1.

### 1. Minimal Environment & Requirements
- Created [`requirements.txt`](file:///d:/SEM%207/VMplacement/requirements.txt) specifying only `pandas`, `numpy`, and `matplotlib`.
- Verified that local CPU-only Python 3.11 environment correctly runs all data parsing and validation routines without any CUDA, Docker, or unnecessary dependencies.

### 2. Bitbrains Dataset Loader (`ml/data_loader.py`)
- **Quoted row stripping**: Cleans row-enclosing quotes without touching raw files.
- **Delimiter parsing**: Correctly handles the `;	` delimiter.
- **Timestamp conversion**:
  - Validated that timestamps are 10-digit Unix epoch seconds (August–September 2013).
  - Converted to UTC `datetime64[ns]` in column `timestamp`.
  - Preserved original integer in `timestamp_raw` for traceability and future simulation integration.
  - Documented the misleading raw header `Timestamp [ms]` in code comments.
- **Column standardization**: Created bidirectional mapping between original headers and snake_case names while preserving all 11 original resource metrics.
- **Strict non-modification**: Raw CPU percentages remain in $[0, 100]\%$ range; duplicate rows/timestamps are preserved without deduplication or resampling.

### 3. Inspection & Validation Tool (`ml/inspect_dataset.py`)
- Single-file detailed report generator.
- Multi-file cross-file schema consistency validator.
- Exports structured summaries to [`results/inspection/`](file:///d:/SEM%207/VMplacement/results/inspection/).

---

## Validation & Test Results

### 1. Single-File Inspection (`1.csv`)
- **Observations**: 8,634 rows, 12 columns (11 original + `timestamp`).
- **Data Integrity**: 0 missing values, 0 duplicate rows, 0 duplicate timestamps, 0 negative values.
- **Time Range**: `2013-08-12 13:40:46 UTC` to `2013-09-11 13:39:58 UTC` (30.0 days).
- **Sampling Interval**:
  - Median: $300.0\text{ s}$ ($5.0\text{ min}$)
  - Mean: $300.24\text{ s}$
  - Min / Max: $299.0\text{ s}$ / $900.0\text{ s}$
  - Distribution: $300\text{ s}$ (96.12%), $301\text{ s}$ (3.29%), $299\text{ s}$ (0.51%), $600\text{ s}$ (0.03%), $303\text{ s}$ (0.01%).
- **CPU Usage [%]**: Min $0.5\%$, Max $97.87\%$, Mean $4.02\%$, Median $0.62\%$, Std Dev $16.90\%$.

### 2. Multi-File Inspection (5 Files & 10 Files)

| File | Rows | Missing | Duplicate Rows | Duplicate Timestamps | Date Range | Median Interval | CPU Mean [%] | CPU Max [%] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1.csv` | 8,634 | 0 | 0 | 0 | 2013-08-12 to 2013-09-11 | 300.0 s | 4.02% | 97.87% |
| `2.csv` | 16,142 | 0 | 3,798 | 3,800 | 2013-08-12 to 2013-09-11 | 209.0 s | 0.24% | 92.27% |
| `3.csv` | 8,634 | 0 | 0 | 0 | 2013-08-12 to 2013-09-11 | 300.0 s | 3.26% | 9.47% |
| `4.csv` | 16,139 | 0 | 3,798 | 3,800 | 2013-08-12 to 2013-09-11 | 209.0 s | 0.24% | 92.20% |
| `5.csv` | 16,140 | 0 | 3,799 | 3,800 | 2013-08-12 to 2013-09-11 | 209.0 s | 0.24% | 92.25% |
| `6.csv` | 16,141 | 0 | 3,798 | 3,800 | 2013-08-12 to 2013-09-11 | 209.0 s | 0.24% | 92.32% |
| `7.csv` | 16,140 | 0 | 3,798 | 3,800 | 2013-08-12 to 2013-09-11 | 209.0 s | 0.24% | 92.17% |
| `8.csv` | 16,142 | 0 | 3,798 | 3,800 | 2013-08-12 to 2013-09-11 | 209.0 s | 0.24% | 92.37% |
| `9.csv` | 16,141 | 0 | 3,798 | 3,800 | 2013-08-12 to 2013-09-11 | 209.0 s | 0.24% | 92.23% |
| `10.csv` | 16,140 | 0 | 3,798 | 3,800 | 2013-08-12 to 2013-09-11 | 209.0 s | 0.24% | 92.30% |

### 3. Empirical Discoveries
1. **100% Schema Consistency**: All 10 files share identical column names, ordering, compatible types, and valid $[0, 100]\%$ CPU ranges.
2. **Zero Missing Values**: No nulls or missing entries detected across any of the 10 files.
3. **Two Distinct VM File Behaviors**:
   - **Type A** (`1.csv`, `3.csv`): 8,634 rows, perfectly clean 5-minute sampling intervals, 0 duplicate timestamps.
   - **Type B** (`2.csv`, `4.csv`-`10.csv`): ~16,140 rows, containing exactly 3,800 duplicate timestamps and ~3,798 duplicate rows.
4. **Untouched Raw Data**: Raw data files in `dataset/fastStorage/2013-8/` remain unmodified.
