#!/usr/bin/env python3
"""
Live Results Dashboard & Watcher for Predictive VM Placement Optimization (WA-AHO)
==================================================================================

Monitors real-time progress of the running Java AHO optimization and CloudSim Plus simulation.
Refreshes every 15-30 seconds, plotting completed epoch metrics and candidate evaluation telemetry.

Usage:
    python tools/live_results_dashboard.py [OPTIONS]

Options:
    --run-id RUN_ID         Filter to specific run ID (default: auto-detect latest run in CSV)
    --interval SECONDS      Refresh interval in seconds (default: 20)
    --total-epochs N        Total expected epochs (default: 997)
    --headless              Run headless (save PNGs without opening GUI window)
    --plots-dir DIR         Directory to save live PNG figures (default: results/live_plots)
    --metrics-file PATH     Path to epoch_metrics.csv (default: results/epoch_metrics.csv)
    --candidates-file PATH  Path to candidate_evaluation_results.csv (default: results/candidate_evaluation_results.csv)
"""

import sys
import os
import time
import argparse
import datetime
from pathlib import Path

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Handle headless environments safely
def check_gui_available():
    try:
        backend = matplotlib.get_backend()
        if 'agg' in backend.lower():
            return False
        return True
    except Exception:
        return False

def parse_args():
    parser = argparse.ArgumentParser(description="Live Results Dashboard for WA-AHO VM Placement")
    parser.add_argument("--run-id", type=str, default=None, help="Specific run_id to filter (default: latest run in epoch_metrics.csv)")
    parser.add_argument("--interval", type=int, default=20, help="Refresh interval in seconds (default: 20)")
    parser.add_argument("--total-epochs", type=int, default=997, help="Total expected epochs (default: 997)")
    parser.add_argument("--headless", action="store_true", help="Force headless mode (save PNGs, no interactive window)")
    parser.add_argument("--once", action="store_true", help="Run once and exit instead of continuous loop")
    parser.add_argument("--plots-dir", type=str, default="results/live_plots", help="Directory to save updated plot images")
    parser.add_argument("--metrics-file", type=str, default="results/epoch_metrics.csv", help="Path to epoch_metrics.csv")
    parser.add_argument("--candidates-file", type=str, default="results/candidate_evaluation_results.csv", help="Path to candidate_evaluation_results.csv")
    return parser.parse_args()

def robust_read_csv(filepath, required_cols=None):
    """
    Safely reads CSV that is being actively written to by another process.
    Handles partial trailing lines, file locks, and encoding issues.
    """
    if not os.path.exists(filepath):
        return None
    try:
        # Check file size
        if os.path.getsize(filepath) < 10:
            return None
        
        # Read CSV with on_bad_lines='skip' to gracefully ignore partially flushed lines
        df = pd.read_csv(
            filepath,
            on_bad_lines='skip',
            encoding='utf-8',
            engine='python'
        )
        if df.empty:
            return None
        
        if required_cols:
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                return None
        return df
    except Exception as e:
        # File might be temporarily locked by writer
        return None

def filter_active_run(metrics_df, requested_run_id=None):
    """
    Selects rows for the active run_id. If none requested, picks the latest run_id.
    """
    if metrics_df is None or metrics_df.empty:
        return None, None
    
    if 'run_id' not in metrics_df.columns:
        return metrics_df, "default_run"
    
    run_ids = metrics_df['run_id'].dropna().unique().tolist()
    if not run_ids:
        return metrics_df, "default_run"
    
    selected_run = requested_run_id if requested_run_id else run_ids[-1]
    filtered = metrics_df[metrics_df['run_id'] == selected_run].copy()
    
    # Sort by epoch and deduplicate by keeping the latest record for each epoch
    if 'epoch' in filtered.columns:
        filtered = filtered.sort_values('epoch').drop_duplicates(subset=['epoch'], keep='last')
        # Only retain rows where completed is true
        if 'completed' in filtered.columns:
            filtered = filtered[filtered['completed'].astype(str).str.lower() == 'true']
    return filtered, selected_run

def plot_dashboard(epoch_df, cand_df, run_id, total_epochs, plots_dir, fig=None, axes=None):
    """
    Generates and updates the multi-panel live visualization grid.
    """
    os.makedirs(plots_dir, exist_ok=True)
    
    epochs = epoch_df['epoch'].values
    num_completed = len(epochs)
    latest_epoch = int(epochs[-1]) if num_completed > 0 else 0
    pct_complete = (num_completed / total_epochs) * 100.0 if total_epochs > 0 else 0.0
    
    # Calculate candidate statistics for this run
    cand_count = len(cand_df) if cand_df is not None else 0
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    if fig is None or axes is None:
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    else:
        for ax_row in axes:
            for ax in ax_row:
                ax.cla()
                ax.grid(True, linestyle='--', alpha=0.6)
    
    fig.suptitle(
        f"LIVE VM Placement Optimization — Run: {run_id} | Epoch: {latest_epoch}/{total_epochs} ({pct_complete:.1f}%) | Refreshed: {datetime.datetime.now().strftime('%H:%M:%S')}",
        fontsize=14, fontweight='bold', color='#1E293B'
    )
    
    # Panel 1: Energy Consumption vs Epoch
    ax1 = axes[0, 0]
    ax1.plot(epochs, epoch_df['energy_wh'], color='#2563EB', marker='o', markersize=4, linewidth=1.5, label='Datacenter Energy')
    ax1.set_title("1. Datacenter Energy Consumption (Wh / Epoch)", fontweight='bold')
    ax1.set_xlabel("Completed Decision Epoch")
    ax1.set_ylabel("Total Energy (Wh)")
    if num_completed > 0:
        latest_energy = epoch_df['energy_wh'].iloc[-1]
        ax1.axhline(latest_energy, color='#93C5FD', linestyle=':', label=f'Latest: {latest_energy:.2f} Wh')
    ax1.legend(loc='upper right', frameon=True)
    
    # Panel 2: SLA Violations vs Epoch
    ax2 = axes[0, 1]
    ax2.bar(epochs, epoch_df['sla_violations'], color='#EF4444', alpha=0.7, width=0.8, label='Per-Epoch Violations')
    cum_sla = epoch_df['sla_violations'].cumsum()
    ax2_twin = ax2.twinx()
    ax2_twin.plot(epochs, cum_sla, color='#7F1D1D', linestyle='--', linewidth=1.5, label='Cumulative SLA')
    ax2_twin.set_ylabel("Cumulative SLA Violations", color='#7F1D1D')
    ax2.set_title("2. SLA Violations (Per-Epoch & Cumulative)", fontweight='bold')
    ax2.set_xlabel("Completed Decision Epoch")
    ax2.set_ylabel("Violations / Epoch", color='#EF4444')
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True)
    
    # Panel 3: Active Physical Machines vs Epoch
    ax3 = axes[1, 0]
    ax3.step(epochs, epoch_df['active_pms'], color='#10B981', where='mid', linewidth=2.0, label='Active PMs')
    ax3.set_title("3. Server Consolidation: Active Physical Hosts (PMs)", fontweight='bold')
    ax3.set_xlabel("Completed Decision Epoch")
    ax3.set_ylabel("Active PMs")
    ax3.set_ylim(0, 20)
    ax3.axhline(20, color='#9CA3AF', linestyle='--', label='Datacenter Capacity (20 PMs)')
    ax3.legend(loc='upper right', frameon=True)
    
    # Panel 4: Mean CPU Utilization vs Epoch
    ax4 = axes[1, 1]
    mean_cpu = epoch_df['mean_cpu_utilization'] * 100.0 if epoch_df['mean_cpu_utilization'].max() <= 1.0 else epoch_df['mean_cpu_utilization']
    ax4.plot(epochs, mean_cpu, color='#8B5CF6', marker='s', markersize=3, linewidth=1.5, label='Mean CPU Util (%)')
    ax4.set_title("4. Active Workload Mean CPU Utilization (%)", fontweight='bold')
    ax4.set_xlabel("Completed Decision Epoch")
    ax4.set_ylabel("CPU Utilization (%)")
    ax4.legend(loc='upper right', frameon=True)
    
    # Panel 5: Inter-Epoch VM Migrations / Placement Changes
    ax5 = axes[2, 0]
    mig_col = 'placement_changes_from_previous_state' if 'placement_changes_from_previous_state' in epoch_df.columns else 'migration_cost'
    ax5.plot(epochs, epoch_df[mig_col], color='#F59E0B', marker='^', markersize=4, linewidth=1.5, label='Placement Changes')
    ax5.set_title("5. Inter-Epoch VM Migrations / Churn", fontweight='bold')
    ax5.set_xlabel("Completed Decision Epoch")
    ax5.set_ylabel("Migration Count / Cost")
    ax5.legend(loc='upper right', frameon=True)
    
    # Panel 6: Optimization Progress Gauge & Summary
    ax6 = axes[2, 1]
    ax6.axis('off')
    
    latest_cand = epoch_df['selected_candidate_id'].iloc[-1] if 'selected_candidate_id' in epoch_df.columns else "N/A"
    latest_fitness = f"{epoch_df['best_fitness'].iloc[-1]:.6f}" if 'best_fitness' in epoch_df.columns else "N/A"
    total_energy_kwh = epoch_df['energy_wh'].sum() / 1000.0
    total_sla = int(epoch_df['sla_violations'].sum())
    mean_active = epoch_df['active_pms'].mean()
    
    summary_text = (
        f"═════════════════════════════════════════════════\n"
        f"  OPTIMIZATION EXECUTION TELEMETRY\n"
        f"═════════════════════════════════════════════════\n"
        f"  • Active Run ID         : {run_id}\n"
        f"  • Progress Completed    : {latest_epoch} / {total_epochs} epochs ({pct_complete:.1f}%)\n"
        f"  • Winning Candidate ID  : {latest_cand}\n"
        f"  • Best Solution Fitness : {latest_fitness}\n"
        f"  • Total Datacenter Energy: {total_energy_kwh:.3f} kWh\n"
        f"  • Total SLA Violations  : {total_sla}\n"
        f"  • Mean Active Hosts     : {mean_active:.2f} PMs / 20\n"
        f"  • Candidates Logged     : {cand_count:,} evaluations\n"
        f"═════════════════════════════════════════════════\n"
        f"  Status: OPTIMIZATION IN PROGRESS (Streaming Live)"
    )
    ax6.text(
        0.05, 0.50, summary_text,
        transform=ax6.transAxes,
        fontsize=11, family='monospace',
        verticalalignment='center',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#F8FAFC', edgecolor='#CBD5E1', linewidth=1.5)
    )
    
    plt.tight_layout()
    
    # Save composite dashboard PNG
    composite_png = os.path.join(plots_dir, "live_dashboard_overview.png")
    fig.savefig(composite_png, dpi=150)
    
    # Save individual high-resolution figures for report use
    fig1, a1 = plt.subplots(figsize=(8, 4))
    a1.plot(epochs, epoch_df['energy_wh'], color='#2563EB', marker='o', markersize=3, linewidth=1.5)
    a1.set_title(f"Datacenter Energy Consumption vs Epoch (Run: {run_id})", fontweight='bold')
    a1.set_xlabel("Epoch"); a1.set_ylabel("Total Energy (Wh)"); a1.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(); fig1.savefig(os.path.join(plots_dir, "energy_per_epoch.png"), dpi=150); plt.close(fig1)
    
    fig2, a2 = plt.subplots(figsize=(8, 4))
    a2.step(epochs, epoch_df['active_pms'], color='#10B981', where='mid', linewidth=2.0)
    a2.set_title(f"Active Physical Machines vs Epoch (Run: {run_id})", fontweight='bold')
    a2.set_xlabel("Epoch"); a2.set_ylabel("Active PMs"); a2.set_ylim(0, 21); a2.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(); fig2.savefig(os.path.join(plots_dir, "active_pms.png"), dpi=150); plt.close(fig2)
    
    fig3, a3 = plt.subplots(figsize=(8, 4))
    a3.bar(epochs, epoch_df['sla_violations'], color='#EF4444', alpha=0.7)
    a3.set_title(f"SLA Violations vs Epoch (Run: {run_id})", fontweight='bold')
    a3.set_xlabel("Epoch"); a3.set_ylabel("Violations / Epoch"); a3.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(); fig3.savefig(os.path.join(plots_dir, "sla_violations.png"), dpi=150); plt.close(fig3)
    
    fig4, a4 = plt.subplots(figsize=(8, 4))
    a4.plot(epochs, epoch_df[mig_col], color='#F59E0B', marker='^', markersize=3, linewidth=1.5)
    a4.set_title(f"VM Migrations vs Epoch (Run: {run_id})", fontweight='bold')
    a4.set_xlabel("Epoch"); a4.set_ylabel("Migrations"); a4.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(); fig4.savefig(os.path.join(plots_dir, "migrations.png"), dpi=150); plt.close(fig4)
    
    return fig, axes

def main():
    args = parse_args()
    
    plots_dir = os.path.abspath(args.plots_dir)
    os.makedirs(plots_dir, exist_ok=True)
    
    print("=" * 75)
    print("  VM PLACEMENT WA-AHO LIVE RESULTS DASHBOARD & WATCHER")
    print("=" * 75)
    print(f"• Monitoring Metrics File   : {os.path.abspath(args.metrics_file)}")
    print(f"• Monitoring Candidate File : {os.path.abspath(args.candidates_file)}")
    print(f"• Refresh Interval          : {args.interval} seconds")
    print(f"• Target Total Epochs       : {args.total_epochs}")
    print(f"• Figures Save Directory    : {plots_dir}")
    print(f"• Headless Mode             : {args.headless or not check_gui_available()}")
    print("=" * 75)
    
    interactive = (not args.headless) and check_gui_available()
    fig, axes = None, None
    if interactive:
        plt.ion()
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        try:
            plt.show(block=False)
        except Exception:
            interactive = False
    
    last_completed_count = -1
    
    try:
        while True:
            metrics_df = robust_read_csv(
                args.metrics_file,
                required_cols=['epoch', 'energy_wh', 'active_pms', 'sla_violations', 'completed']
            )
            cand_df = robust_read_csv(args.candidates_file)
            
            if metrics_df is None or metrics_df.empty:
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Waiting for completed epochs in '{args.metrics_file}'...")
            else:
                filtered_df, run_id = filter_active_run(metrics_df, args.run_id)
                if filtered_df is None or filtered_df.empty:
                    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] No completed records found for run '{args.run_id}'.")
                else:
                    curr_completed = len(filtered_df)
                    latest_epoch = filtered_df['epoch'].iloc[-1]
                    latest_energy = filtered_df['energy_wh'].iloc[-1]
                    latest_pms = filtered_df['active_pms'].iloc[-1]
                    latest_sla = filtered_df['sla_violations'].iloc[-1]
                    
                    if curr_completed != last_completed_count:
                        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [UPDATE] Run '{run_id}' | Completed Epoch {latest_epoch}/{args.total_epochs} (Total: {curr_completed}) | Energy: {latest_energy:.2f} Wh | Active PMs: {latest_pms} | SLA: {latest_sla}")
                        last_completed_count = curr_completed
                    
                    fig, axes = plot_dashboard(
                        filtered_df, cand_df, run_id, args.total_epochs, plots_dir, fig=fig, axes=axes
                    )
                    
                    if interactive:
                        plt.draw()
                        plt.pause(0.1)

            if args.once:
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Single-pass execution completed. Exiting.")
                break
            
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\n[Dashboard] Gracefully stopped live watcher.")

if __name__ == "__main__":
    main()
