package com.vmplacement.optimization;

import java.util.List;

/** Supplies authoritative VM/PM infrastructure configuration to Person 3. */
public interface InfrastructureProvider {
    InfrastructureSnapshot snapshot(DecisionEpoch epoch);
}
