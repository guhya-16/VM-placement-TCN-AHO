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

Targeting **Review-3 milestone** (~50% end-to-end scope working).

| Stage | Status | Notes |
| :--- | :---: | :--- |
| **1. Data Collection** | `Done` | Bitbrains `fastStorage` subset ingestion |
| **2. Data Preprocessing** | `In Progress` | Sequence formatting & scaling |
| **3. TCN Prediction** | `In Progress` | PyTorch TCN architecture & training loop |
| **4. Risk Score Calculation** | `In Progress` | Residual volatility scoring |
| **5. CloudSim Scaffolding** | `Done` | Host/VM datacenter setup in Java |
| **6. PABFD Baseline** | `Done` | Power-Aware Best Fit Decreasing implementation |
| **7. WA-AHO Optimization** | `In Progress` | Workload-Aware Adaptive HO algorithm |
| **8. Metrics & Evaluation** | `Done` | Energy, SLA, PM count, migration reporting |

---

## 3. Folder Structure
vm-placement-tcn-aho/
├── data/ # Raw and processed Bitbrains fastStorage trace datasets
├── notebooks/ # Exploratory notebooks (EDA, feature analysis, visualization)
├── tcn/ # PyTorch TCN model architecture, training, validation, and inference scripts
├── risk/ # Residual calculation, volatility estimation, and risk scoring module
├── cloudsim/ # Java Maven project with CloudSim Plus simulation engine, PABFD, and WA-AHO policy support
├── baselines/ # Baseline placement algorithms (PABFD policy, random, static heuristics)
├── results/ # Experiment logs, metrics, exported CSV summaries, and plot artifacts
└── docs/ # Architecture diagrams, research paper notes, and presentation materials

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
- **JDK Version**: Java 17+ (confirmed working with JDK 24)
- **IDE**: VS Code / Antigravity (Java Extension Pack) or IntelliJ IDEA
- **Build Tool**: Apache Maven 3.8+
- **Key Dependency**: CloudSim Plus `8.5.7` (`org.cloudsimplus:cloudsimplus:8.5.7`)

To build the Java project:
```bash
cd cloudsim
mvn clean compile
mvn exec:java
```

---

## 5. Team & Module Ownership

| Team Member | Track / Module Ownership | Primary Deliverables |
| :--- | :--- | :--- |
| **[ Member 1 Name ]** | Track 1: Data & TCN Prediction | Bitbrains pipeline, `data/`, `notebooks/`, `tcn/` |
| **[ Member 2 Name ]** | Track 2: CloudSim Simulation & PABFD | `cloudsim/`, `results/` |
| **[ Member 3 Name ]** | Track 3: Optimization & WA-AHO | `optimizer/`, WA-AHO policy, `placement.csv` |

---

## 6. How Python and Java Talk to Each Other

To maintain decoupling, fast iteration, and clean modular boundaries between the Machine Learning pipeline (Python) and the Cloud Simulation engine (Java), we adopt a **file-handoff architecture**:

1. **Python Output**: The Python pipeline processes Bitbrains workload traces, runs inference through the trained TCN model, computes future CPU/memory utilization predictions and residual risk/volatility scores, and writes output files into structured CSV/JSON format:
   - `results/predictions.csv` (contains `vm_id`, `timestamp`, `predicted_cpu`, `predicted_ram`, `risk_score`)
2. **Java CloudSim Plus Ingestion**: The Java CloudSim Plus simulation reads `placement.csv` (VM→Host mapping produced by the optimizer) via `VmAllocationPolicyFromFile`, and runs the simulation using real Bitbrains-driven workload traces.
3. **Results Logging**: Java exports simulation logs and performance metrics to `results/simulation_results.csv` for unified comparison across PABFD, HO, and WA-AHO.