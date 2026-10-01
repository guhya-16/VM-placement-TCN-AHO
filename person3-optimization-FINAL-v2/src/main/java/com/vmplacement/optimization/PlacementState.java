package com.vmplacement.optimization;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class PlacementState {
    private final long timestamp;
    private final Map<String, String> vmToPmId;

    private PlacementState(long timestamp, Map<String, String> vmToPmId) {
        this.timestamp = timestamp;
        this.vmToPmId = Collections.unmodifiableMap(new LinkedHashMap<>(vmToPmId));
    }

    public static PlacementState empty() {
        return new PlacementState(Long.MIN_VALUE, Map.of());
    }

    public static PlacementState fromSolution(long timestamp, List<VM> vms,
                                               List<PM> pms, PlacementSolution solution) {
        int[] mapping = solution.getVmToPm();
        if (mapping.length != vms.size()) throw new IllegalArgumentException("Placement/VM size mismatch");
        Map<String, String> result = new LinkedHashMap<>();
        for (int i = 0; i < vms.size(); i++) {
            int pmIndex = mapping[i];
            if (pmIndex < 0 || pmIndex >= pms.size()) throw new IllegalArgumentException("Invalid PM index");
            result.put(vms.get(i).getId(), pms.get(pmIndex).getId());
        }
        return new PlacementState(timestamp, result);
    }

    public long getTimestamp() { return timestamp; }
    public Map<String, String> getVmToPmId() { return vmToPmId; }
    public String getPmForVm(String vmId) { return vmToPmId.get(vmId); }
    public boolean hasVm(String vmId) { return vmToPmId.containsKey(vmId); }
    public boolean isEmpty() { return vmToPmId.isEmpty(); }
}
