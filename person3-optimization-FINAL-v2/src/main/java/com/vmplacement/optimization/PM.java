package com.vmplacement.optimization;

public final class PM {
    private final String id;
    private final int pes;
    private final double mipsPerPe;
    private final double ramMb;
    private final double storageMb;
    private final double maxPowerWatts;
    private final double staticPowerWatts;

    public PM(String id, int pes, double mipsPerPe, double ramMb, double storageMb,
              double maxPowerWatts, double staticPowerWatts) {
        this.id = id;
        this.pes = pes;
        this.mipsPerPe = mipsPerPe;
        this.ramMb = ramMb;
        this.storageMb = storageMb;
        this.maxPowerWatts = maxPowerWatts;
        this.staticPowerWatts = staticPowerWatts;
        if (pes <= 0 || mipsPerPe <= 0 || ramMb < 0 || storageMb < 0
                || maxPowerWatts < 0 || staticPowerWatts < 0
                || staticPowerWatts > maxPowerWatts) {
            throw new IllegalArgumentException("Invalid PM profile: " + id);
        }
    }

    public String getId() { return id; }
    public int getPes() { return pes; }
    public double getMipsPerPe() { return mipsPerPe; }
    public double getTotalMips() { return pes * mipsPerPe; }
    public double getCpuCapacity() { return getTotalMips(); }
    public double getRamMb() { return ramMb; }
    public double getRamCapacity() { return ramMb; }
    public double getStorageMb() { return storageMb; }
    public double getMaxPowerWatts() { return maxPowerWatts; }
    public double getStaticPowerWatts() { return staticPowerWatts; }
}
