package com.vmplacement.optimization;

import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

public final class EpochState {
    private final long timestamp;
    private final List<RiskState> riskStates;
    private final List<VM> activeVms;
    private final List<PM> pms;
    private final PlacementState previousPlacement;
    private final Map<String, RiskState> riskByVm;

    public EpochState(long timestamp, List<RiskState> riskStates, List<VM> activeVms,
                      List<PM> pms, PlacementState previousPlacement) {
        this.timestamp = timestamp;
        this.riskStates = List.copyOf(riskStates);
        this.activeVms = List.copyOf(activeVms);
        this.pms = List.copyOf(pms);
        this.previousPlacement = previousPlacement;
        this.riskByVm = this.riskStates.stream().collect(Collectors.toUnmodifiableMap(
                RiskState::vmId, Function.identity(), (a, b) -> b));
    }

    public long getTimestamp() { return timestamp; }
    public List<RiskState> getRiskStates() { return riskStates; }
    public List<VM> getActiveVms() { return activeVms; }
    public List<PM> getPms() { return pms; }
    public PlacementState getPreviousPlacement() { return previousPlacement; }
    public RiskState getRiskState(String vmId) { return riskByVm.get(vmId); }
}
