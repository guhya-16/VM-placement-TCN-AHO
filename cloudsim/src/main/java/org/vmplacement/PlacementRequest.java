package org.vmplacement;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Clean data transfer structure representing a candidate VM-to-PM placement request
 * sent from the Person 3 Optimizer to the Person 2 CloudSim evaluator.
 */
public class PlacementRequest {

    private final int epoch;
    private final int iteration;
    private final String candidateId;
    private final long timestamp;
    private final String method;
    private final double slaThreshold;
    private final List<VmPlacementSpec> vmPlacements;
    private final List<PmSpec> pmSpecs;

    public PlacementRequest(int epoch, int iteration, String candidateId, long timestamp,
                            String method, double slaThreshold,
                            List<VmPlacementSpec> vmPlacements,
                            List<PmSpec> pmSpecs) {
        this.epoch = epoch;
        this.iteration = iteration;
        this.candidateId = candidateId != null ? candidateId : "CANDIDATE_0";
        this.timestamp = timestamp;
        this.method = method != null ? method : "UNKNOWN";
        this.slaThreshold = slaThreshold > 0.0 ? slaThreshold : 0.05;
        this.vmPlacements = vmPlacements != null ? new ArrayList<>(vmPlacements) : new ArrayList<>();
        this.pmSpecs = pmSpecs != null ? new ArrayList<>(pmSpecs) : new ArrayList<>();
    }

    /**
     * Convenience constructor constructing VM placement specs from a map with standard 4-tier resources.
     */
    public PlacementRequest(int epoch, int iteration, String candidateId, long timestamp,
                            String method, Map<String, Integer> vmToPmMap) {
        this(epoch, iteration, candidateId, timestamp, method, 0.05, buildDefaultSpecs(vmToPmMap), Collections.emptyList());
    }

    /**
     * Backward-compatible constructor for basic epoch-level requests.
     */
    public PlacementRequest(int epoch, long timestamp, String method, Map<String, Integer> vmToPmMap) {
        this(epoch, 0, "CANDIDATE_DEFAULT", timestamp, method, 0.05, buildDefaultSpecs(vmToPmMap), Collections.emptyList());
    }

    private static List<VmPlacementSpec> buildDefaultSpecs(Map<String, Integer> vmToPmMap) {
        List<VmPlacementSpec> list = new ArrayList<>();
        if (vmToPmMap != null) {
            int idx = 0;
            for (Map.Entry<String, Integer> entry : vmToPmMap.entrySet()) {
                String vmId = entry.getKey();
                int pmId = entry.getValue();
                // Deterministic 4-tier resource template matching standard dataset setup
                int tier = idx % 4;
                int pes = tier + 1;
                double mips = (tier + 1) * 500.0;
                double ram = (tier == 0) ? 512.0 : (tier == 1) ? 1024.0 : (tier == 2) ? 2048.0 : 3072.0;
                double storage = (40.0 + tier * 20.0) * 1024.0;
                list.add(new VmPlacementSpec(vmId, pmId, 0.0, 0.0, null, pes, mips, ram, storage));
                idx++;
            }
        }
        return list;
    }

    public int getEpoch() {
        return epoch;
    }

    public int getIteration() {
        return iteration;
    }

    public String getCandidateId() {
        return candidateId;
    }

    public long getTimestamp() {
        return timestamp;
    }

    public String getMethod() {
        return method;
    }

    public double getSlaThreshold() {
        return slaThreshold;
    }

    public List<VmPlacementSpec> getVmPlacements() {
        return Collections.unmodifiableList(vmPlacements);
    }

    public List<PmSpec> getPmSpecs() {
        return Collections.unmodifiableList(pmSpecs);
    }

    public int getVmCount() {
        return vmPlacements.size();
    }

    public Map<String, Integer> getVmToPmMap() {
        Map<String, Integer> map = new LinkedHashMap<>();
        for (VmPlacementSpec spec : vmPlacements) {
            map.put(spec.getVmId(), spec.getPmId());
        }
        return Collections.unmodifiableMap(map);
    }
}
