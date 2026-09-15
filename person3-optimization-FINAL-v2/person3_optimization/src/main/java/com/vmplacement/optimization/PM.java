package com.vmplacement.optimization;

public class PM {

    private String id;

    private int pes;
    private double mipsPerPe;
    private double ramMb;
    private double storageMb;

    private double maxPowerWatts;
    private double staticPowerWatts;

    /*
     * Main CloudSim-compatible constructor.
     */
    public PM(
            String id,
            int pes,
            double mipsPerPe,
            double ramMb,
            double storageMb,
            double maxPowerWatts,
            double staticPowerWatts) {

        this.id = id;
        this.pes = pes;
        this.mipsPerPe = mipsPerPe;
        this.ramMb = ramMb;
        this.storageMb = storageMb;

        this.maxPowerWatts =
                maxPowerWatts;

        this.staticPowerWatts =
                staticPowerWatts;
    }

    /*
     * Synthetic experiment constructor.
     *
     * Example:
     *
     * new PM("PM1", 100, 16384)
     *
     * Total CPU = 100
     * RAM       = 16384 MB
     */
    public PM(
            String id,
            double cpuCapacity,
            double ramCapacity) {

        this.id = id;

        /*
         * Give the PM enough PEs so multiple
         * synthetic VMs can share it.
         */
        this.pes = 100;

        /*
         * Total MIPS remains exactly cpuCapacity.
         */
        this.mipsPerPe =
                cpuCapacity / 100.0;

        this.ramMb =
                ramCapacity;

        this.storageMb = 0.0;

        /*
         * Synthetic power model.
         */
        this.maxPowerWatts = 200.0;
        this.staticPowerWatts = 100.0;
    }

    public String getId() {
        return id;
    }

    public int getPes() {
        return pes;
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
        return pes * mipsPerPe;
    }

    public double getCpuCapacity() {
        return getTotalMips();
    }

    public double getRamCapacity() {
        return ramMb;
    }
}