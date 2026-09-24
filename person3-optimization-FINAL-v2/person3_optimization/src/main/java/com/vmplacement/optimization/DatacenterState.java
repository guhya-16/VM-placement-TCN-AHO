package com.vmplacement.optimization;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Stateful Datacenter Tracker holding the current VM-to-Host (PM) allocation mapping,
 * carried across decision epochs in rolling experiments.
 */
public class DatacenterState {

    // Current VM ID (e.g. "VM_001") -> Host/PM index (0-indexed, 0..19)
    private final Map<String, Integer> vmToHostMap = new LinkedHashMap<>();
    private int currentEpoch = 0;
    private long currentTimestamp = 0L;

    public DatacenterState() {
    }

    /**
     * Resets the datacenter state to empty (clean slate for new experiment run).
     */
    public void reset() {
        vmToHostMap.clear();
        currentEpoch = 0;
        currentTimestamp = 0L;
    }

    public boolean isEmpty() {
        return vmToHostMap.isEmpty();
    }

    public int size() {
        return vmToHostMap.size();
    }

    public Integer getHostForVm(String vmId) {
        return vmToHostMap.get(vmId);
    }

    public Map<String, Integer> getPlacementMap() {
        return Collections.unmodifiableMap(vmToHostMap);
    }

    public int getCurrentEpoch() {
        return currentEpoch;
    }

    public long getCurrentTimestamp() {
        return currentTimestamp;
    }

    /**
     * Updates the state with this epoch's winning placement.
     *
     * @param epoch Epoch number (1-based)
     * @param timestamp Unix epoch seconds
     * @param vms List of active VMs placed this epoch
     * @param solution Winning PlacementSolution
     */
    public void updateState(int epoch, long timestamp, List<VM> vms, PlacementSolution solution) {
        this.currentEpoch = epoch;
        this.currentTimestamp = timestamp;
        int[] pmIndices = solution.getVmToPm();
        for (int i = 0; i < vms.size(); i++) {
            String vmId = vms.get(i).getId();
            vmToHostMap.put(vmId, pmIndices[i]);
        }
    }

    /**
     * Updates the state from a Map of VM ID -> Host ID.
     */
    public void updateState(int epoch, long timestamp, Map<String, Integer> newPlacement) {
        this.currentEpoch = epoch;
        this.currentTimestamp = timestamp;
        vmToHostMap.putAll(newPlacement);
    }

    /**
     * Computes the number of placement changes (migrations) between the previous state
     * and the winning placement of the current epoch.
     *
     * Invariant: For the very first epoch (empty previous state), migrations = 0.
     * For subsequent epochs, migrations = number of active VMs whose PM changed.
     */
    public int computePlacementChanges(List<VM> currentVms, PlacementSolution currentSolution) {
        if (vmToHostMap.isEmpty() || currentSolution == null) {
            return 0;
        }
        int changes = 0;
        int[] placement = currentSolution.getVmToPm();
        for (int i = 0; i < currentVms.size(); i++) {
            String vmId = currentVms.get(i).getId();
            Integer prevHost = vmToHostMap.get(vmId);
            if (prevHost != null && prevHost != placement[i]) {
                changes++;
            }
        }
        return changes;
    }

    public int computePlacementChanges(Map<String, Integer> newPlacement) {
        if (vmToHostMap.isEmpty() || newPlacement == null) {
            return 0;
        }
        int changes = 0;
        for (Map.Entry<String, Integer> entry : newPlacement.entrySet()) {
            String vmId = entry.getKey();
            Integer prevHost = vmToHostMap.get(vmId);
            if (prevHost != null && !prevHost.equals(entry.getValue())) {
                changes++;
            }
        }
        return changes;
    }

    /**
     * Creates a warm-start PlacementSolution for the current epoch using previous placement.
     * If previous placement is feasible for the new epoch workloads, it is returned.
     * If not feasible (e.g. host overload), it is repaired to the nearest feasible solution.
     * If state is empty, returns null.
     */
    public PlacementSolution createWarmStartSolution(List<VM> vms, List<PM> pms, Random random) {
        if (vmToHostMap.isEmpty() || vms == null || vms.isEmpty()) {
            return null;
        }

        int[] placement = new int[vms.size()];

        for (int i = 0; i < vms.size(); i++) {
            String vmId = vms.get(i).getId();
            Integer prevHost = vmToHostMap.get(vmId);
            if (prevHost != null && prevHost >= 0 && prevHost < pms.size()) {
                placement[i] = prevHost;
            } else {
                placement[i] = random.nextInt(pms.size());
            }
        }

        PlacementSolution candidate = new PlacementSolution(placement);
        if (PlacementValidator.isFeasible(vms, pms, candidate)) {
            return candidate;
        }

        // Attempt repair to keep as many previous assignments as possible while restoring feasibility
        PlacementSolution repaired = PlacementRepair.repair(vms, pms, candidate, random);
        return repaired;
    }
}
