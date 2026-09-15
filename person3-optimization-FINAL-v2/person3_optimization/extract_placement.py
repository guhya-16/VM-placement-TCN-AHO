"""
Extracts one decision epoch's VM->PM placement from Person 3's
experiment_results.csv into the simple placement.csv format
(vm_id,host_id) that Person 2's VmAllocationPolicyFromFile reads.

Usage:
    python extract_placement.py <epoch_number> <standard|adaptive> <output_path>

Example:
    python extract_placement.py 1 standard ../../cloudsim/placement.csv
"""
import csv
import ast
import sys
import os

def extract(epoch_number, algorithm, output_path,
            input_path="experiment_results.csv"):
    col = f"{algorithm}_placement"  # "standard_placement" or "adaptive_placement"

    with open(input_path, newline="") as f:
        reader = csv.DictReader(f)
        row = next((r for r in reader if int(r["epoch"]) == epoch_number), None)

    if row is None:
        raise ValueError(f"Epoch {epoch_number} not found in {input_path}")

    # e.g. "[17, 17, 17, 17, 17]" -> [17, 17, 17, 17, 17]
    pm_ids_1_indexed = ast.literal_eval(row[col])

    # Ensure parent directory of output_path exists if specified
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["vm_id", "host_id"])
        for vm_id, pm_id in enumerate(pm_ids_1_indexed):
            # Person 3's PM_017 style numbering -> your CloudSim's 0-indexed host list
            host_id = pm_id - 1
            writer.writerow([vm_id, host_id])

    print(f"Wrote {len(pm_ids_1_indexed)} VM placements "
          f"(epoch {epoch_number}, {algorithm}) to {output_path}")

if __name__ == "__main__":
    epoch = int(sys.argv[1])
    algorithm = sys.argv[2]
    output_path = sys.argv[3]
    extract(epoch, algorithm, output_path)
