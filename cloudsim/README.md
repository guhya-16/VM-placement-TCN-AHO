# VM Placement — CloudSim Plus Simulation Engine

Simulation engine evaluating predictive, energy-efficient VM placement decisions within a 20-PM heterogeneous datacenter (PM1/PM2/PM3) running real Bitbrains fastStorage multi-trace workloads.

---

## 1. Supported Execution Modes

The simulation engine provides two distinct, documented execution modes:

### MODE A: Synthetic 24-VM PABFD Baseline
- **Execution Mode**: `ExecutionMode.SYNTHETIC_PABFD_BASELINE`
- **Scale**: Fixed 24 heterogeneous synthetic VMs (Types 1–4: Small, Medium-Low, Medium-High, Large).
- **Allocation Policy**: Power-Aware Best Fit Decreasing (`VmAllocationPolicyPabfd`).
- **Purpose**: Initial CloudSim Plus framework validation, heuristic baseline, and datacenter stress-testing under higher aggregate utilization across 20 PMs.
- **Scientific Role**: Separate baseline/validation experiment. Not directly comparable to small-scale real-data runs.

### MODE B: Real-Data Dynamic HO External Placement (Default)
- **Execution Mode**: `ExecutionMode.REAL_DATA_EXTERNAL_PLACEMENT`
- **Scale**: Data-driven, variable $N$ active VMs determined by the decision epoch from `placement.csv` (e.g., $N=5$ for Epoch 1, $N=6$, $N=4$, up to 7 validated VMs).
- **Allocation Policy**: `VmAllocationPolicyFromFile` with strict integrity and datacenter constraint validation.
- **VM Hardware Specification**: Uses the medium specification (2 PEs, 1000 MIPS, 1024 MB RAM, 60 GB storage) as a documented modeling simplification due to the lack of per-trace hardware telemetry in the dataset.
- **Purpose**: Evaluates real optimization decisions produced by Person 3's Standard HO and Adaptive HO metaheuristics driven by Person 1's TCN workload predictions.

---

## 2. Dynamic VM Architecture & Workflow

```
             REAL WORKLOAD DATA (Bitbrains)
                          ↓
               Current Active VMs (Epoch t)
                          ↓
                        N VMs (e.g. N=5, 6, 4)
                          ↓
                VM Workload Manifest
                          ↓
               Placement Optimization
                    /          \
            Standard HO      Adaptive HO
                    \          /
                         ↓
                   placement.csv
                   (vm_id, host_id)
                         ↓
                  CloudSim Plus
            (VmAllocationPolicyFromFile)
                         ↓
         ┌───────────────┼───────────────┐
         ↓               ↓               ↓
       Energy           SLA         Utilization
         ↓               ↓               ↓
              simulation_results.csv
```

---

## 3. Host Indexing Convention

- **Person 3 Optimizer**: Uses **1-indexed** Physical Machine IDs (`PM_001` to `PM_020` $\rightarrow$ IDs 1 to 20).
- **CloudSim Plus Engine**: Uses **0-indexed** Host IDs (`Host 0` to `Host 19` $\rightarrow$ IDs 0 to 19).
- **Conversion Rule**: `host_id = pm_id - 1` (handled by `extract_placement.py`).
  - *Example*: PM 17 (PM_017, PM3 class) $\rightarrow$ CloudSim Host 16.

---

## 4. End-to-End Execution Guide

### Step 1: Run Person 3 Optimization Experiments
```powershell
cd person3-optimization-FINAL-v2/person3_optimization
java -cp target\classes com.vmplacement.optimization.Main
```
*Generates*: `experiment_results.csv` containing multi-epoch optimization decisions.

### Step 2: Extract Placement for a Decision Epoch
```powershell
# From person3-optimization-FINAL-v2/person3_optimization:
python extract_placement.py 1 standard ../../cloudsim/placement.csv

# Or for adaptive optimization:
python extract_placement.py 1 adaptive ../../cloudsim/placement.csv
```
*Generates*: `cloudsim/placement.csv` formatted as:
```csv
vm_id,host_id
0,16
1,16
2,16
3,16
4,16
```

### Step 3: Run CloudSim Plus Simulation

#### Running Real-Data Mode (Default):
```powershell
cd cloudsim
.\mvnw.cmd clean compile
.\mvnw.cmd exec:java
```

#### Running Synthetic 24-VM PABFD Baseline:
Pass the CLI argument `pabfd`:
```powershell
.\mvnw.cmd exec:java "-Dexec.args=pabfd"
```
Or set `CURRENT_MODE = ExecutionMode.SYNTHETIC_PABFD_BASELINE;` in `Simulation.java`.

---

## 5. Output Telemetry & Manifests

- **Startup Validation Banner**: Displays active mode, placement file path, active VM count $N$, host count, and explicit `VM -> Host` mappings.
- **CPU Utilization Stats**: Replayed from Bitbrains traces over a fixed 1-hour observation window ($3600\text{s}$).
- **SLA Violation Detection**: Compares delivered CPU against trace demand (5% tolerance).
- **Power & Energy Consumption**: Computed from PM1/PM2/PM3 linear power models ($P_{\text{static}}$ to $P_{\text{max}}$).
- **Output Files**:
  - `results/simulation_results.csv`: Cumulative record of all simulation runs.
  - `results/vm_manifest.csv`: Active VM specifications.
  - `results/host_manifest.csv`: PM infrastructure manifest.
