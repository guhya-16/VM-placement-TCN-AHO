package com.vmplacement.optimization;

/** Fails closed until the team's real infrastructure/CloudSim configuration is connected. */
public final class UnconfiguredInfrastructureProvider implements InfrastructureProvider {
    @Override
    public InfrastructureSnapshot snapshot(DecisionEpoch epoch) {
        throw new IllegalStateException(
                "No real InfrastructureProvider is configured. Person 2 must expose the authoritative VM/PM configuration before optimization can run.");
    }
}
