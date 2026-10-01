package com.vmplacement.optimization;

public final class VM {
    private final String id;
    private final int pes;
    private final double mips;
    private final double ramMb;
    private final double storageMb;
    private final double currentCpuPercent;
    private final double predictedMeanCpuPercent;
    private final double predictedPeakCpuPercent;
    private final double predictedStdCpuPercent;
    private final double volatilityScore;
    private final double riskScore;
    private final String riskState;

    public VM(String id, int pes, double mips, double ramMb, double storageMb,
              RiskState state, double cpuScale) {
        this.id = requireText(id, "id");
        this.pes = pes;
        this.mips = mips;
        this.ramMb = ramMb;
        this.storageMb = storageMb;
        this.currentCpuPercent = cpuPercent(state.currentCpu(), cpuScale);
        this.predictedMeanCpuPercent = cpuPercent(state.predictedMeanCpu(), cpuScale);
        this.predictedPeakCpuPercent = cpuPercent(state.predictedPeakCpu(), cpuScale);
        this.predictedStdCpuPercent = cpuPercent(state.predictedStdCpu(), cpuScale);
        this.volatilityScore = state.volatilityScore();
        this.riskScore = state.riskScore();
        this.riskState = state.riskState();
        if (pes <= 0 || mips <= 0 || ramMb < 0 || storageMb < 0) {
            throw new IllegalArgumentException("Invalid VM resource profile for " + id);
        }
    }

    private static String requireText(String value, String field) {
        if (value == null || value.isBlank()) throw new IllegalArgumentException(field + " is blank");
        return value.trim();
    }

    private static double cpuPercent(double value, double scale) {
        // scale < 0 means per-value auto detection, matching the Person 1 handoff
        // convention used by the original branch: values in [0,1] are fractions;
        // larger values are already percentages.
        double v = scale < 0 ? (value >= 0.0 && value <= 1.0 ? value * 100.0 : value) : value * scale;
        if (!Double.isFinite(v) || v < 0 || v > 100.0 + 1e-9) {
            throw new IllegalArgumentException("CPU value outside [0,100] after configured scaling: " + v);
        }
        return Math.max(0.0, Math.min(100.0, v));
    }

    public String getId() { return id; }
    public int getPes() { return pes; }
    public double getMips() { return mips; }
    public double getRamMb() { return ramMb; }
    public double getStorageMb() { return storageMb; }
    public double getCurrentCpuUtilization() { return currentCpuPercent; }
    public double getPredictedMeanCpu() { return predictedMeanCpuPercent; }
    public double getPredictedPeakCpu() { return predictedPeakCpuPercent; }
    public double getPredictedStdCpu() { return predictedStdCpuPercent; }
    public double getVolatilityScore() { return volatilityScore; }
    public double getRiskScore() { return riskScore; }
    public String getRiskState() { return riskState; }
}
