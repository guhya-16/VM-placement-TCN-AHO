package com.vmplacement.optimization;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

public final class HandoffValidator {
    private HandoffValidator() {}
    public static void validate(List<DecisionEpoch> epochs) {
        Set<String> seenVmTimestamp = new HashSet<>();
        long previous = Long.MIN_VALUE;
        for (DecisionEpoch epoch : epochs) {
            if (epoch.timestamp() < previous) throw new IllegalArgumentException("Decision epochs are not chronological");
            previous = epoch.timestamp();
            Set<String> vms = new HashSet<>();
            for (RiskState r : epoch.riskStates()) {
                if (!vms.add(r.vmId())) throw new IllegalArgumentException(
                        "Duplicate VM " + r.vmId() + " at timestamp " + epoch.timestamp());
                String key = r.vmId()+"@"+epoch.timestamp();
                if (!seenVmTimestamp.add(key)) throw new IllegalArgumentException("Duplicate VM/timestamp pair: "+key);
                if (!Double.isFinite(r.riskScore()) || !Double.isFinite(r.volatilityScore()))
                    throw new IllegalArgumentException("Non-finite Person 1 risk/volatility for "+key);
            }
        }
    }
}
