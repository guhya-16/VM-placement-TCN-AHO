package org.vmplacement;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Stateful Datacenter Tracker for Person 2 / CloudSim.
 *
 * Tracks the finalized previous-epoch placement state (epoch t-1) to compute
 * inter-epoch VM placement changes (migrations) for candidate placements at epoch t.
 *
 * Guarantees:
 * 1. Intermediate candidate evaluations do NOT modify previousFinalPlacement.
 * 2. Only commitFinalPlacement(...) commits the winning placement and advances state.
 * 3. Operates purely on standard Java primitives with zero optimizer dependencies.
 */
public class DatacenterState {

    // Finalized VM ID (e.g. "VM_001") -> Host/PM index (0-indexed, 0..19)
    private final Map<String, Integer> previousFinalPlacement = new LinkedHashMap<>();
    private int currentEpoch = 0;
    private long currentTimestamp = 0L;

    public DatacenterState() {
    }

    /**
     * Resets the datacenter state to empty (clean slate for new experiment run).
     */
    public synchronized void reset() {
        previousFinalPlacement.clear();
        currentEpoch = 0;
        currentTimestamp = 0L;
    }

    public synchronized boolean isEmpty() {
        return previousFinalPlacement.isEmpty();
    }

    public synchronized int size() {
        return previousFinalPlacement.size();
    }

    public synchronized Integer getHostForVm(String vmId) {
        return previousFinalPlacement.get(vmId);
    }

    public synchronized Map<String, Integer> getFinalPlacementMap() {
        return Collections.unmodifiableMap(new LinkedHashMap<>(previousFinalPlacement));
    }

    public synchronized int getCurrentEpoch() {
        return currentEpoch;
    }

    public synchronized long getCurrentTimestamp() {
        return currentTimestamp;
    }

    /**
     * Computes the number of placement changes (migrations) between the previous finalized state
     * and the candidate placement for the current epoch.
     *
     * Invariant: For the very first epoch (empty previous state), placement changes = 0.
     * For subsequent epochs, placement changes = number of active VMs whose PM changed.
     *
     * Note: This is a read-only comparison and does NOT alter previousFinalPlacement.
     *
     * @param candidatePlacement Candidate mapping of VM ID to PM index
     * @return Number of VM placement changes compared to finalized previous epoch state
     */
    public synchronized int computePlacementChanges(Map<String, Integer> candidatePlacement) {
        if (previousFinalPlacement.isEmpty() || candidatePlacement == null) {
            return 0;
        }
        int changes = 0;
        for (Map.Entry<String, Integer> entry : candidatePlacement.entrySet()) {
            String vmId = entry.getKey();
            Integer prevHost = previousFinalPlacement.get(vmId);
            if (prevHost != null && !prevHost.equals(entry.getValue())) {
                changes++;
            }
        }
        return changes;
    }

    /**
     * Commits the winning/final placement of an epoch into the datacenter state.
     * This updates the baseline for subsequent epoch comparisons.
     *
     * @param epoch Epoch number (1-based)
     * @param timestamp Unix epoch seconds
     * @param finalPlacement Final selected mapping of VM ID to PM index
     */
    public synchronized void commitFinalPlacement(int epoch, long timestamp, Map<String, Integer> finalPlacement) {
        this.currentEpoch = epoch;
        this.currentTimestamp = timestamp;
        if (finalPlacement != null) {
            previousFinalPlacement.clear();
            previousFinalPlacement.putAll(finalPlacement);
        }
    }
}
