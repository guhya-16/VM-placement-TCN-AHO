package com.vmplacement.optimization;

/** Static VM resource specification supplied by the CloudSim/infrastructure side. */
public record VmResourceSpec(String vmId, int pes, double mips, double ramMb, double storageMb) {
    public VmResourceSpec {
        if (vmId == null || vmId.isBlank()) throw new IllegalArgumentException("VM id is blank");
        if (pes <= 0 || mips <= 0 || ramMb < 0 || storageMb < 0)
            throw new IllegalArgumentException("Invalid VM resource specification for " + vmId);
    }
}
