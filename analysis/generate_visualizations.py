import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['figure.dpi'] = 300

os.makedirs("results/plots", exist_ok=True)

print("Loading dataset and simulation files...")
std_df = pd.read_csv("results/standard_ho_results.csv")
ada_df = pd.read_csv("results/adaptive_ho_results.csv")
exp_df = pd.read_csv("results/experiment_results.csv")
manifest_df = pd.read_csv("results/vm_manifest.csv")

pabfd_path = "results/pabfd_real_results.csv"
pabfd_df = pd.read_csv(pabfd_path) if os.path.exists(pabfd_path) else None

print(f"Loaded Standard HO: {len(std_df)} rows")
print(f"Loaded Adaptive HO: {len(ada_df)} rows")
print(f"Loaded Experiment Results: {len(exp_df)} rows")
if pabfd_df is not None:
    print(f"Loaded PABFD Real: {len(pabfd_df)} rows")

# -------------------------------------------------------------
# 1. Verification & Integrity Checks
# -------------------------------------------------------------
print("\n--- Integrity Verification ---")
assert len(std_df) == 997, f"Standard HO has {len(std_df)} rows instead of 997"
assert len(ada_df) == 997, f"Adaptive HO has {len(ada_df)} rows instead of 997"
assert (std_df['epoch'] == ada_df['epoch']).all(), "Epoch numbers mismatch"
assert (std_df['timestamp'] == ada_df['timestamp']).all(), "Timestamps mismatch"
print("[OK] 997 Epochs confirmed present exactly once with 100% timestamp match.")

# -------------------------------------------------------------
# GRAPH 1: Energy vs Epoch
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(std_df['epoch'], std_df['energy_wh'], label='Standard HO', color='#2563EB', alpha=0.85, linewidth=1.2)
ax.plot(ada_df['epoch'], ada_df['energy_wh'], label='Adaptive HO', color='#DC2626', linestyle='--', alpha=0.85, linewidth=1.2)
if pabfd_df is not None:
    ax.plot(pabfd_df['epoch'], pabfd_df['energy_wh'], label='PABFD (Real Data)', color='#059669', linestyle=':', alpha=0.85, linewidth=1.2)

ax.set_title("Datacenter Energy Consumption per Decision Epoch (997 Epochs)")
ax.set_xlabel("Decision Epoch Index (1 to 997)")
ax.set_ylabel("Datacenter Total Energy (Wh / epoch)")
ax.legend(loc='upper right', frameon=True)
ax.set_ylim(bottom=1420, top=1432)
plt.tight_layout()
plt.savefig("results/plots/energy_vs_epoch.png", dpi=300)
plt.close()
print("[OK] Saved results/plots/energy_vs_epoch.png")

# -------------------------------------------------------------
# GRAPH 2: Active PMs vs Epoch
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 4))
ax.step(std_df['epoch'], std_df['active_pm'], label='Standard HO', color='#2563EB', linewidth=1.2, where='mid')
ax.step(ada_df['epoch'], ada_df['active_pm'], label='Adaptive HO', color='#DC2626', linestyle='--', linewidth=1.2, where='mid')
if pabfd_df is not None:
    ax.step(pabfd_df['epoch'], pabfd_df['active_pm'], label='PABFD (Real Data)', color='#059669', linestyle=':', linewidth=1.2, where='mid')

ax.set_title("Active Physical Machines (PMs) per Decision Epoch")
ax.set_xlabel("Decision Epoch Index")
ax.set_ylabel("Number of Active PMs")
ax.set_ylim(0, 5)
ax.set_yticks(range(0, 6))
ax.legend(loc='upper right', frameon=True)
plt.tight_layout()
plt.savefig("results/plots/active_pms_vs_epoch.png", dpi=300)
plt.close()
print("[OK] Saved results/plots/active_pms_vs_epoch.png")

# -------------------------------------------------------------
# GRAPH 3: SLA Violations vs Epoch (Cumulative & Per-Epoch)
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

# Cumulative SLA
ax1.plot(std_df['epoch'], np.cumsum(std_df['sla_violations']), label='Standard HO', color='#2563EB', linewidth=1.8)
ax1.plot(ada_df['epoch'], np.cumsum(ada_df['sla_violations']), label='Adaptive HO', color='#DC2626', linestyle='--', linewidth=1.8)
if pabfd_df is not None:
    ax1.plot(pabfd_df['epoch'], np.cumsum(pabfd_df['sla_violations']), label='PABFD (Real Data)', color='#059669', linestyle=':', linewidth=1.8)

ax1.set_title("Cumulative and Per-Epoch SLA Violations Across 997 Epochs")
ax1.set_ylabel("Cumulative SLA Violations")
ax1.legend(loc='upper left', frameon=True)

# Per-Epoch SLA
ax2.bar(std_df['epoch'], std_df['sla_violations'], width=1.0, color='#6B7280', alpha=0.6, label='Violations in Epoch')
ax2.set_xlabel("Decision Epoch Index")
ax2.set_ylabel("Violations / Epoch")
ax2.legend(loc='upper right', frameon=True)

plt.tight_layout()
plt.savefig("results/plots/sla_vs_epoch.png", dpi=300)
plt.close()
print("[OK] Saved results/plots/sla_vs_epoch.png")

# -------------------------------------------------------------
# GRAPH 4: Inter-Epoch Migrations vs Epoch
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

ax1.plot(std_df['epoch'], np.cumsum(std_df['migrations']), label=f"Standard HO (Total: {std_df['migrations'].sum()})", color='#2563EB', linewidth=1.8)
ax1.plot(ada_df['epoch'], np.cumsum(ada_df['migrations']), label=f"Adaptive HO (Total: {ada_df['migrations'].sum()})", color='#DC2626', linestyle='--', linewidth=1.8)
if pabfd_df is not None:
    ax1.plot(pabfd_df['epoch'], np.cumsum(pabfd_df['migrations']), label=f"PABFD Real Data (Total: {pabfd_df['migrations'].sum()})", color='#059669', linestyle=':', linewidth=1.8)

ax1.set_title("Cumulative and Per-Epoch Inter-Epoch VM Migrations")
ax1.set_ylabel("Cumulative Migrations")
ax1.legend(loc='upper left', frameon=True)

# Per-epoch migration counts rolling 20-epoch average
roll_std = std_df['migrations'].rolling(20, min_periods=1).mean()
roll_ada = ada_df['migrations'].rolling(20, min_periods=1).mean()
ax2.plot(std_df['epoch'], roll_std, label='Standard HO (20-Epoch Rolling Mean)', color='#2563EB', alpha=0.8)
ax2.plot(ada_df['epoch'], roll_ada, label='Adaptive HO (20-Epoch Rolling Mean)', color='#DC2626', linestyle='--', alpha=0.8)
ax2.set_xlabel("Decision Epoch Index")
ax2.set_ylabel("Migrations / Epoch (Rolling Avg)")
ax2.legend(loc='upper right', frameon=True)

plt.tight_layout()
plt.savefig("results/plots/migrations_vs_epoch.png", dpi=300)
plt.close()
print("[OK] Saved results/plots/migrations_vs_epoch.png")

# -------------------------------------------------------------
# GRAPH 5: TCN Risk vs Adaptive HO Exploration Probability
# -------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(12, 5))

color = '#8B5CF6'
ax1.set_xlabel("Decision Epoch Index")
ax1.set_ylabel("TCN Composite Risk / Volatility", color=color)
l1 = ax1.plot(exp_df['epoch'], exp_df['prediction_risk'], label='TCN Prediction Risk (R)', color='#EC4899', alpha=0.7, linewidth=1.2)
l2 = ax1.plot(exp_df['epoch'], exp_df['workload_variation'], label='Workload Volatility (V)', color='#8B5CF6', alpha=0.7, linewidth=1.2)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color2 = '#D97706'
ax2.set_ylabel("Adaptive HO Exploration Probability ($p_{explore}$)", color=color2)
l3 = ax2.plot(exp_df['epoch'], exp_df['exploration_probability'], label='Dynamic Exploration Prob ($p_{explore}$)', color=color2, linewidth=1.8)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(0.1, 0.9)

# Added unified legend
lines = l1 + l2 + l3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', frameon=True)
plt.title("Dynamic Coupling: TCN Workload Risk vs Adaptive HO Exploration Dynamics")
plt.tight_layout()
plt.savefig("results/plots/tcn_risk_vs_adaptation.png", dpi=300)
plt.close()
print("[OK] Saved results/plots/tcn_risk_vs_adaptation.png")

# -------------------------------------------------------------
# GRAPH 6: Placement Divergence Between Standard & Adaptive HO
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))

# Calculate divergence indicator
diff_indicator = [1 if exp_df.loc[i, 'standard_placement'] != exp_df.loc[i, 'adaptive_placement'] else 0 for i in range(len(exp_df))]
diff_pct = (sum(diff_indicator) / len(diff_indicator)) * 100

ax1.fill_between(exp_df['epoch'], 0, diff_indicator, step='mid', color='#6366F1', alpha=0.6, label=f'Placement Divergence ({diff_pct:.1f}% Differing)')
ax1.set_title("Algorithmic Decision Divergence: Standard HO vs Adaptive HO Across 997 Epochs")
ax1.set_ylabel("Divergence Indicator (1=Differ, 0=Same)")
ax1.set_yticks([0, 1])
ax1.set_yticklabels(['Identical', 'Divergent'])
ax1.legend(loc='upper right', frameon=True)

# Pie chart of divergence proportion
labels = [f'Differing Placements\n({sum(diff_indicator)} epochs, {diff_pct:.1f}%)',
          f'Identical Placements\n({len(diff_indicator) - sum(diff_indicator)} epochs, {100-diff_pct:.1f}%)']
colors = ['#6366F1', '#E5E7EB']
ax2.pie([sum(diff_indicator), len(diff_indicator) - sum(diff_indicator)], labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=140, textprops={'fontsize': 11, 'weight': 'bold'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2})
ax2.set_title("Overall Placement Divergence Distribution")

plt.tight_layout()
plt.savefig("results/plots/placement_divergence.png", dpi=300)
plt.close()
print("[OK] Saved results/plots/placement_divergence.png")

# -------------------------------------------------------------
# GRAPH 7: 3-Way Summary Benchmark Bar Chart
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

algs = ['Standard HO', 'Adaptive HO']
if pabfd_df is not None:
    algs = ['PABFD (Real)', 'Standard HO', 'Adaptive HO']

# Mean Energy
energies = [std_df['energy_wh'].mean(), ada_df['energy_wh'].mean()]
if pabfd_df is not None:
    energies = [pabfd_df['energy_wh'].mean(), std_df['energy_wh'].mean(), ada_df['energy_wh'].mean()]

colors = ['#059669', '#2563EB', '#DC2626'] if pabfd_df is not None else ['#2563EB', '#DC2626']
bars1 = axes[0].bar(algs, energies, color=colors, width=0.5, edgecolor='black', linewidth=0.8)
axes[0].set_title("Mean Energy (Wh / epoch)")
axes[0].set_ylabel("Wh")
axes[0].set_ylim(1420, 1430)
for bar in bars1:
    yval = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.2f}", ha='center', va='bottom', fontsize=9, weight='bold')

# Total SLA Violations
slas = [std_df['sla_violations'].sum(), ada_df['sla_violations'].sum()]
if pabfd_df is not None:
    slas = [pabfd_df['sla_violations'].sum(), std_df['sla_violations'].sum(), ada_df['sla_violations'].sum()]

bars2 = axes[1].bar(algs, slas, color=colors, width=0.5, edgecolor='black', linewidth=0.8)
axes[1].set_title("Total SLA Violations (997 Epochs)")
axes[1].set_ylabel("Count")
for bar in bars2:
    yval = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{int(yval)}", ha='center', va='bottom', fontsize=9, weight='bold')

# Total Migrations
migs = [std_df['migrations'].sum(), ada_df['migrations'].sum()]
if pabfd_df is not None:
    migs = [pabfd_df['migrations'].sum(), std_df['migrations'].sum(), ada_df['migrations'].sum()]

bars3 = axes[2].bar(algs, migs, color=colors, width=0.5, edgecolor='black', linewidth=0.8)
axes[2].set_title("Total Inter-Epoch Migrations")
axes[2].set_ylabel("Count")
for bar in bars3:
    yval = bar.get_height()
    axes[2].text(bar.get_x() + bar.get_width()/2.0, yval + 50, f"{int(yval)}", ha='center', va='bottom', fontsize=9, weight='bold')

plt.tight_layout()
plt.savefig("results/plots/aggregate_comparison_barchart.png", dpi=300)
plt.close()
print("[OK] Saved results/plots/aggregate_comparison_barchart.png")

print("\nAll publication-quality graphs successfully generated in results/plots/!")
