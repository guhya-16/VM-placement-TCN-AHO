package com.vmplacement.optimization;

public class VM {

    private String id;

    private int pes;
    private double mips;
    private double ramMb;
    private double storageMb;

    private double currentCpuUtilization;
    private double predictedCpuUtilization;

    private double predictedMeanCpu;
    private double predictedPeakCpu;
    private double predictedStdCpu;

    /*
     * Main constructor.
     */
    public VM(
            String id,
            int pes,
            double mips,
            double ramMb,
            double storageMb,
            double currentCpuUtilization,
            double predictedCpuUtilization) {

        this.id = id;
        this.pes = pes;
        this.mips = mips;
        this.ramMb = ramMb;
        this.storageMb = storageMb;

        this.currentCpuUtilization =
                currentCpuUtilization;

        this.predictedCpuUtilization =
                predictedCpuUtilization;

        this.predictedMeanCpu =
                predictedCpuUtilization;

        this.predictedPeakCpu =
                predictedCpuUtilization;

        this.predictedStdCpu = 0.0;
    }

    /*
     * Constructor including prediction statistics.
     */
    public VM(
            String id,
            int pes,
            double mips,
            double ramMb,
            double storageMb,
            double currentCpuUtilization,
            double predictedCpuUtilization,
            double predictedMeanCpu,
            double predictedPeakCpu,
            double predictedStdCpu) {

        this.id = id;
        this.pes = pes;
        this.mips = mips;
        this.ramMb = ramMb;
        this.storageMb = storageMb;

        this.currentCpuUtilization =
                currentCpuUtilization;

        this.predictedCpuUtilization =
                predictedCpuUtilization;

        this.predictedMeanCpu =
                predictedMeanCpu;

        this.predictedPeakCpu =
                predictedPeakCpu;

        this.predictedStdCpu =
                predictedStdCpu;
    }

    /*
     * Synthetic experiment constructor.
     *
     * CPU demand is represented as a percentage.
     *
     * Example:
     * cpuDemand = 40
     * means approximately 40% CPU utilization.
     */
    public VM(
            String id,
            double cpuDemand,
            double predictedCpuDemand,
            double ramDemand) {

        this.id = id;

        this.pes = 1;

        /*
         * Synthetic PM total CPU capacity = 100.
         */
        this.mips = 100.0;

        this.ramMb = ramDemand;
        this.storageMb = 0.0;

        this.currentCpuUtilization =
                cpuDemand;

        this.predictedCpuUtilization =
                predictedCpuDemand;

        this.predictedMeanCpu =
                predictedCpuDemand;

        this.predictedPeakCpu =
                predictedCpuDemand;

        this.predictedStdCpu = 0.0;
    }

    /*
     * Integer compatibility constructor.
     */
    public VM(
            String id,
            int cpuDemand,
            int predictedCpuDemand,
            int ramDemand) {

        this(
                id,
                (double) cpuDemand,
                (double) predictedCpuDemand,
                (double) ramDemand
        );
    }

    public String getId() {
        return id;
    }

    public int getPes() {
        return pes;
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

    public double getCurrentCpuUtilization() {
        return currentCpuUtilization;
    }

    public double getPredictedCpuUtilization() {
        return predictedCpuUtilization;
    }

    public double getPredictedMeanCpu() {
        return predictedMeanCpu;
    }

    public double getPredictedPeakCpu() {
        return predictedPeakCpu;
    }

    public double getPredictedStdCpu() {
        return predictedStdCpu;
    }

    /*
     * Backward-compatible methods.
     */
    public double getCpuDemand() {
        return currentCpuUtilization;
    }

    public double getPredictedCpuDemand() {
        return predictedCpuUtilization;
    }

    public double getRamDemand() {
        return ramMb;
    }
}