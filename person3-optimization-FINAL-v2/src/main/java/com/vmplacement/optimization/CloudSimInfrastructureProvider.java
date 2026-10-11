package com.vmplacement.optimization;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Supplies authoritative 20-PM datacenter infrastructure and 4-tier VM specifications to Person 3.
 */
public final class CloudSimInfrastructureProvider implements InfrastructureProvider {

    private final List<PM> pms;
    private final Map<String, VmResourceSpec> vmSpecs;

    public CloudSimInfrastructureProvider() {
        this.pms = new ArrayList<>();
        // 10 x PM1 (Hosts 0-9)
        for (int i = 0; i < 10; i++) {
            pms.add(new PM(String.format("PM_%03d", i + 1), 2, 2660.0, 4.0 * 1024.0, 160.0 * 1024.0, 135.0, 93.7));
        }
        // 6 x PM2 (Hosts 10-15)
        for (int i = 10; i < 16; i++) {
            pms.add(new PM(String.format("PM_%03d", i + 1), 4, 3067.0, 8.0 * 1024.0, 250.0 * 1024.0, 113.0, 42.3));
        }
        // 4 x PM3 (Hosts 16-19)
        for (int i = 16; i < 20; i++) {
            pms.add(new PM(String.format("PM_%03d", i + 1), 12, 3067.0, 16.0 * 1024.0, 500.0 * 1024.0, 222.0, 58.4));
        }

        this.vmSpecs = new LinkedHashMap<>();
        // Standard Bitbrains VM catalog with 4 heterogeneous resource tiers
        for (int id = 1; id <= 100; id++) {
            String vmId = String.format("VM_%03d", id);
            int tier = (id - 1) % 4;
            int pes = tier + 1;
            double mips = (tier + 1) * 500.0;
            double ram = (tier == 0) ? 512.0 : (tier == 1) ? 1024.0 : (tier == 2) ? 2048.0 : 3072.0;
            double storage = (40.0 + tier * 20.0) * 1024.0;
            vmSpecs.put(vmId, new VmResourceSpec(vmId, pes, mips, ram, storage));
        }
    }

    @Override
    public InfrastructureSnapshot snapshot(DecisionEpoch epoch) {
        return new InfrastructureSnapshot(vmSpecs, pms);
    }
}
