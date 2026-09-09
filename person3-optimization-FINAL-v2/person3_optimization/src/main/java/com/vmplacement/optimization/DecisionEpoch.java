package com.vmplacement.optimization;

import java.util.ArrayList;
import java.util.List;

public class DecisionEpoch {

    private long decisionTimestamp;
    private List<RiskState> riskStates;

    public DecisionEpoch(
            long decisionTimestamp,
            List<RiskState> riskStates) {

        this.decisionTimestamp = decisionTimestamp;
        this.riskStates = new ArrayList<>(riskStates);
    }

    public long getDecisionTimestamp() {
        return decisionTimestamp;
    }

    public List<RiskState> getRiskStates() {
        return riskStates;
    }

    public int getVmCount() {
        return riskStates.size();
    }

    @Override
    public String toString() {

        return "DecisionEpoch{"
                + "timestamp=" + decisionTimestamp
                + ", vmCount=" + riskStates.size()
                + '}';
    }
}