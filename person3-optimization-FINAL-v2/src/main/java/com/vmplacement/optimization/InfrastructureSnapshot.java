package com.vmplacement.optimization;

import java.util.List;
import java.util.Map;

/**
 * Infrastructure visible to Person 3 for one decision timestamp.
 * The authoritative implementation is supplied by the CloudSim/infrastructure side;
 * Person 3 does not invent VM/PM resources or maintain a duplicate CSV manifest.
 */
public record InfrastructureSnapshot(Map<String, VmResourceSpec> vmSpecs, List<PM> pms) {
    public InfrastructureSnapshot {
        vmSpecs = Map.copyOf(vmSpecs);
        pms = List.copyOf(pms);
        if (pms.isEmpty()) throw new IllegalArgumentException("Infrastructure must contain at least one PM");
    }
}
