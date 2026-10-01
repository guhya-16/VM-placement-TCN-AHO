package com.vmplacement.optimization;

import java.util.List;

public record DecisionEpoch(long timestamp, List<RiskState> riskStates) {
    public DecisionEpoch {
        riskStates = List.copyOf(riskStates);
    }
}
