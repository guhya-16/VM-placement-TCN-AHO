package com.vmplacement.optimization;

public record RiskState(
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
}
