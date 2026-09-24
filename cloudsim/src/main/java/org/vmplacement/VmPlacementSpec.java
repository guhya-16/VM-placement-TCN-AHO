package org.vmplacement;

/**
 * Specification and candidate PM assignment for a single VM within a PlacementRequest.
 */
public class VmPlacementSpec {

    private final String vmId;
    private final int pmId;
    private final double currentCpuUtilization;   // CPU demand percentage (0..100) or fraction (0..1)
    private final double predictedCpuUtilization; // Predicted CPU demand
    private final Integer previousPmId;           // Previous PM index (null if initial epoch / new VM)
    private final int peCount;
    private final double mips;
    private final double ramMb;
    private final double storageMb;

    public VmPlacementSpec(String vmId, int pmId, double currentCpuUtilization,
                          double predictedCpuUtilization, Integer previousPmId,
                          int peCount, double mips, double ramMb, double storageMb) {
        this.vmId = vmId;
        this.pmId = pmId;
        this.currentCpuUtilization = currentCpuUtilization;
        this.predictedCpuUtilization = predictedCpuUtilization;
        this.previousPmId = previousPmId;
        this.peCount = peCount;
        this.mips = mips;
        this.ramMb = ramMb;
        this.storageMb = storageMb;
    }

    public String getVmId() {
        return vmId;
    }

    public int getPmId() {
        return pmId;
    }

    public double getCurrentCpuUtilization() {
        return currentCpuUtilization;
    }

    public double getPredictedCpuUtilization() {
        return predictedCpuUtilization;
    }

    public Integer getPreviousPmId() {
        return previousPmId;
    }

    public int getPeCount() {
        return peCount;
    }

    public double getMips() {
        return mips;
    }

    public double getRamMb() {
        return ramMb;
    }

    public double getStorageMb() {
        return storageMb;
    }

    @Override
    public String toString() {
        return String.format("VmPlacementSpec[%s -> PM_%03d, PEs=%d, MIPS=%.0f, RAM=%.0fMB]",
                vmId, pmId + 1, peCount, mips, ramMb);
    }
}
