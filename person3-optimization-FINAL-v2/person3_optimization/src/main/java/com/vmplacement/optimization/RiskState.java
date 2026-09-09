package com.vmplacement.optimization;

public class RiskState {

    private String vmId;
    private long decisionTimestamp;

    private double currentCpu;

    // TCN predictions
    private double predictedCpuT5m;
    private double predictedCpuT10m;
    private double predictedCpuT15m;
    private double predictedCpuT30m;
    private double predictedCpuT45m;
    private double predictedCpuT60m;

    // Aggregated prediction information
    private double predictedMeanCpu;
    private double predictedPeakCpu;
    private double predictedStdCpu;

    // Risk information
    private double volatilityScore;
    private double riskScore;
    private String riskState;

    public RiskState(
            String vmId,
            long decisionTimestamp,
            double currentCpu,
            double predictedCpuT5m,
            double predictedCpuT10m,
            double predictedCpuT15m,
            double predictedCpuT30m,
            double predictedCpuT45m,
            double predictedCpuT60m,
            double predictedMeanCpu,
            double predictedPeakCpu,
            double predictedStdCpu,
            double volatilityScore,
            double riskScore,
            String riskState) {

        this.vmId = vmId;
        this.decisionTimestamp = decisionTimestamp;
        this.currentCpu = currentCpu;

        this.predictedCpuT5m = predictedCpuT5m;
        this.predictedCpuT10m = predictedCpuT10m;
        this.predictedCpuT15m = predictedCpuT15m;
        this.predictedCpuT30m = predictedCpuT30m;
        this.predictedCpuT45m = predictedCpuT45m;
        this.predictedCpuT60m = predictedCpuT60m;

        this.predictedMeanCpu = predictedMeanCpu;
        this.predictedPeakCpu = predictedPeakCpu;
        this.predictedStdCpu = predictedStdCpu;

        this.volatilityScore = volatilityScore;
        this.riskScore = riskScore;
        this.riskState = riskState;
    }

    public String getVmId() {
        return vmId;
    }

    public long getDecisionTimestamp() {
        return decisionTimestamp;
    }

    public double getCurrentCpu() {
        return currentCpu;
    }

    public double getPredictedCpuT5m() {
        return predictedCpuT5m;
    }

    public double getPredictedCpuT10m() {
        return predictedCpuT10m;
    }

    public double getPredictedCpuT15m() {
        return predictedCpuT15m;
    }

    public double getPredictedCpuT30m() {
        return predictedCpuT30m;
    }

    public double getPredictedCpuT45m() {
        return predictedCpuT45m;
    }

    public double getPredictedCpuT60m() {
        return predictedCpuT60m;
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

    public double getVolatilityScore() {
        return volatilityScore;
    }

    public double getRiskScore() {
        return riskScore;
    }

    public String getRiskState() {
        return riskState;
    }

    @Override
    public String toString() {

        return vmId
                + " | Timestamp = " + decisionTimestamp
                + " | Current CPU = " + currentCpu
                + " | Predicted 5m = " + predictedCpuT5m
                + " | Predicted Mean = " + predictedMeanCpu
                + " | Predicted Peak = " + predictedPeakCpu
                + " | Std = " + predictedStdCpu
                + " | Volatility = " + volatilityScore
                + " | Risk = " + riskScore
                + " | Risk State = " + riskState;
    }
}