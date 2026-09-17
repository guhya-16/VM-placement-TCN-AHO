"""
Extracts one decision epoch's VM->PM placement from Person 3's
experiment_results.csv and vm_manifest.csv into placement.csv with
full VM sizing specifications:
(vm_id, host_id, pes, mips, ram_mb, storage_mb)

Usage:
    python extract_placement.py <epoch_number> <standard|adaptive> <output_path>

Example:
    python extract_placement.py 8 standard placement_standard.csv
    python extract_placement.py 8 adaptive placement_adaptive.csv
"""
import csv
import ast
import sys
import os

def find_file(candidates):
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]

def extract(epoch_number, algorithm, output_path,
            input_path=None, manifest_path=None):
    if not input_path:
        input_path = find_file([
            "experiment_results.csv",
            "../person3-optimization-FINAL-v2/person3_optimization/experiment_results.csv",
            "person3-optimization-FINAL-v2/person3_optimization/experiment_results.csv",
            "../results/experiment_results.csv"
        ])

    if not manifest_path:
        manifest_path = find_file([
            "vm_manifest.csv",
            "../person3-optimization-FINAL-v2/person3_optimization/vm_manifest.csv",
            "person3-optimization-FINAL-v2/person3_optimization/vm_manifest.csv",
            "../results/vm_manifest.csv"
        ])

    col = f"{algorithm}_placement"  # "standard_placement" or "adaptive_placement"

    with open(input_path, newline="") as f:
        reader = csv.DictReader(f)
        row = next((r for r in reader if int(r["epoch"]) == epoch_number), None)

    if row is None:
        raise ValueError(f"Epoch {epoch_number} not found in {input_path}")

    # e.g. "[16, 16, 16, 16, 16]" -> [16, 16, 16, 16, 16] (0-indexed host IDs)
    pm_ids = ast.literal_eval(row[col])

    # Load VM manifest rows for this epoch
    vm_specs = []
    if os.path.exists(manifest_path):
        with open(manifest_path, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if int(r["epoch"]) == epoch_number:
                    vm_specs.append(r)

    # Ensure parent directory of output_path exists if specified
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["vm_id", "host_id", "pes", "mips", "ram_mb", "storage_mb"])
        for vm_id, pm_id in enumerate(pm_ids):
            host_id = pm_id  # Track 3 optimizer outputs 0-indexed PM indices (0 to 19)
            if vm_id < len(vm_specs):
                spec = vm_specs[vm_id]
                writer.writerow([
                    vm_id,
                    host_id,
                    spec.get("pes", "1"),
                    spec.get("mips", "1000"),
                    spec.get("ram_mb", "1024"),
                    spec.get("storage_mb", "61440")
                ])
            else:
                writer.writerow([vm_id, host_id, 1, 1000, 1024, 61440])

    print(f"Wrote {len(pm_ids)} VM placements (epoch {epoch_number}, {algorithm}) to {output_path}")

if __name__ == "__main__":
    epoch = int(sys.argv[1])
    algorithm = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "placement.csv"
    extract(epoch, algorithm, output_path)
