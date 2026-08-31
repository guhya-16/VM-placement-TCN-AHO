# Predictive Energy-Efficient VM Placement — TCN + WA-AHO

Predictive Energy-Efficient Virtual Machine Placement in Cloud Data Centers using Temporal Convolutional Networks (TCN) and Workload-Aware Adaptive Hippopotamus Optimization (WA-AHO). This project presents an end-to-end framework for predicting future virtual machine (VM) CPU and memory utilization, calculating risk and volatility metrics from prediction residual errors, and dynamically placing VMs across physical machines (PMs) to minimize energy consumption, SLA violations, active host count, and VM migrations.

---

## 1. Pipeline Stages

1. **Data Collection**: Load Bitbrains `fastStorage` trace workload datasets (CPU and memory usage time-series).
2. **Data Preprocessing**: Clean, normalize, aggregate, and structure historical time-series sequences into temporal windows.
3. **TCN Utilization Prediction**: Train a Temporal Convolutional Network (TCN) model to forecast future VM CPU and memory demand.
4. **Risk & Volatility Computation**: Calculate prediction residuals, uncertainty, and volatility scores to identify risk-heavy VMs.
5. **CloudSim Plus Scaffolding**: Set up CloudSim Plus simulation environment (DataCenter, Host PMs, VM models, and Cloudlet workloads).
6. **PABFD Baseline**: Implement Power-Aware Best Fit Decreasing (PABFD) heuristic for baseline placement and energy/SLA benchmark.
7. **WA-AHO Optimization Policy**: Execute Workload-Aware Adaptive Hippopotamus Optimization algorithm for optimal feasible VM placement.
8. **Metrics Evaluation**: Compute energy consumption (kWh), SLA violation rate (%), active host count, migration count, and resource imbalance.

---

## 2. Project Status

Targeting **Review-3 milestone in 17 days** (~50% end-to-end scope working).

| Stage | Status | Target Completion | Notes |
| :--- | :---: | :---: | :--- |
| **1. Data Collection** | `Planned` | Week 1 | Bitbrains `fastStorage` subset ingestion |
| **2. Data Preprocessing** | `Planned` | Week 1 | Sequence formatting & scaling |
| **3. TCN Prediction** | `Planned` | Week 1 | PyTorch TCN architecture & training loop |
| **4. Risk Score Calculation** | `Planned` | Week 2 | Residual volatility scoring |
| **5. CloudSim Scaffolding** | `Planned` | Week 2 | Host/VM datacenter setup in Java |
| **6. PABFD Baseline** | `Planned` | Week 2 | Power-Aware Best Fit Decreasing implementation |
| **7. WA-AHO Optimization** | `Planned` | Week 2-3 | Workload-Aware Adaptive HO algorithm |
| **8. Metrics & Evaluation** | `Planned` | Week 3 | Energy, SLA, PM count, migration reporting |

---

## 3. Folder Structure

```
vm-placement-tcn-aho/
├── data/           # Raw and processed Bitbrains fastStorage trace datasets
├── notebooks/      # Exploratory notebooks (EDA, feature analysis, visualization)
├── tcn/            # PyTorch TCN model architecture, training, validation, and inference scripts
├── risk/           # Residual calculation, volatility estimation, and risk scoring module
├── cloudsim/       # Java Maven project with CloudSim Plus 8.0.0 simulation engine & WA-AHO policy
├── baselines/      # Baseline placement algorithms (PABFD policy, random, static heuristics)
├── results/        # Experiment logs, metrics, exported CSV summaries, and plot artifacts
└── docs/           # Architecture diagrams, research paper notes, and presentation materials
```

---

## 4. Environment Setup

### Python Environment (ML & Data Pipeline)
- **Python Version**: Python 3.10+
- **Core Packages**:
  - `torch` (PyTorch 2.x)
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`

To install Python dependencies:
```bash
pip install -r requirements.txt
```

### Java Environment (Simulation & WA-AHO Engine)
- **JDK Version**: Java 17 (OpenJDK 17 or Oracle JDK 17)
- **IDE**: IntelliJ IDEA (Recommended) or Eclipse / VS Code
- **Build Tool**: Apache Maven 3.8+
- **Key Dependency**: CloudSim Plus `8.0.0` (`org.cloudsimplus:cloudsim-plus:8.0.0`)

To build the Java project:
```bash
cd cloudsim
mvn clean compile
```

---

## 5. Team & Module Ownership

| Team Member | Track / Module Ownership | Primary Deliverables |
| :--- | :--- | :--- |
| **[ Member 1 Name ]** | Track 1: Data & TCN Prediction | Bitbrains pipeline, `data/`, `notebooks/`, `tcn/` |
| **[ Member 2 Name ]** | Track 2: Risk Scoring & WA-AHO Algorithm | `risk/`, `cloudsim/.../optimizer/`, WA-AHO Java policy |
| **[ Member 3 Name ]** | Track 3: CloudSim Simulation & Evaluation | `cloudsim/.../simulation/`, `baselines/` (PABFD), `results/` |

---

## 6. How Python and Java Talk to Each Other

To maintain decoupling, fast iteration, and clean modular boundaries between the Machine Learning pipeline (Python) and the Cloud Simulation engine (Java), we adopt a **file-handoff architecture**:

1. **Python Output**: The Python pipeline processes Bitbrains workload traces, runs inference through the trained TCN model, computes future CPU/memory utilization predictions and residual risk/volatility scores, and writes output files into structured CSV/JSON format:
   - `results/predictions.csv` (contains `vm_id`, `timestamp`, `predicted_cpu`, `predicted_ram`, `risk_score`)
2. **Java CloudSim Plus Ingestion**: The Java CloudSim Plus simulation scaffolding reads the `predictions.csv` during workload initialization. The custom `WA-AHO` VmAllocationPolicy and `PABFD` baseline policy consume these predicted loads and risk scores to execute host selection, migration decisions, and power monitoring.
3. **Results Logging**: Java exports simulation logs and performance metrics back to `results/simulation_metrics.csv` for unified plotting and tabular reporting in Python.
