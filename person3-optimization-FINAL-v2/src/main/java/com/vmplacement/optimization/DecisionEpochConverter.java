package com.vmplacement.optimization;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public final class DecisionEpochConverter {
    private DecisionEpochConverter() {}

    public static List<VM> toVMs(DecisionEpoch epoch, Map<String, VmResourceSpec> specs, double cpuScale) {
        List<VM> result = new ArrayList<>();
        for (RiskState state : epoch.riskStates()) {
            VmResourceSpec p = specs.get(state.vmId());
            if (p == null) {
                throw new IllegalArgumentException("Infrastructure has no resource specification for active VM "
                        + state.vmId() + " at timestamp " + epoch.timestamp());
            }
            result.add(new VM(p.vmId(), p.pes(), p.mips(), p.ramMb(), p.storageMb(), state, cpuScale));
        }
        return result;
    }
}
