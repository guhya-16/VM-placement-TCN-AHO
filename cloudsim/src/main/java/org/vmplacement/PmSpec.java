package org.vmplacement;

/**
 * Specification for a Physical Machine (PM / Host) in CloudSim.
 */
public class PmSpec {

    private final int pmId;
    private final String name;
    private final int peCount;
    private final double mipsPerPe;
    private final double ramMb;
    private final double storageMb;
    private final double maxPowerWatts;
    private final double staticPowerWatts;

    public PmSpec(int pmId, String name, int peCount, double mipsPerPe,
                  double ramMb, double storageMb, double maxPowerWatts, double staticPowerWatts) {
        this.pmId = pmId;
        this.name = name != null ? name : String.format("PM_%03d", pmId + 1);
        this.peCount = peCount;
        this.mipsPerPe = mipsPerPe;
        this.ramMb = ramMb;
        this.storageMb = storageMb;
        this.maxPowerWatts = maxPowerWatts;
        this.staticPowerWatts = staticPowerWatts;
    }

    public int getPmId() {
        return pmId;
    }

    public String getName() {
        return name;
    }

    public int getPeCount() {
        return peCount;
    }

    public double getMipsPerPe() {
        return mipsPerPe;
    }

    public double getRamMb() {
        return ramMb;
    }

    public double getStorageMb() {
        return storageMb;
    }

    public double getMaxPowerWatts() {
        return maxPowerWatts;
    }

    public double getStaticPowerWatts() {
        return staticPowerWatts;
    }

    public double getTotalMips() {
        return peCount * mipsPerPe;
    }

    @Override
    public String toString() {
        return String.format("PmSpec[%s (ID %d): PEs=%d, MIPS=%.0f, RAM=%.0fMB, Pmax=%.1fW]",
                name, pmId, peCount, getTotalMips(), ramMb, maxPowerWatts);
    }
}
