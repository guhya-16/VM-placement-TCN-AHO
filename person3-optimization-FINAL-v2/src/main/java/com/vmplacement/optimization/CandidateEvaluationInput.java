package com.vmplacement.optimization;

import java.util.List;

/** Exact Person 3 -> CloudSim candidate-evaluation handoff. One candidate contains one row per active VM. */
public record CandidateEvaluationInput(int epoch, int iteration, String candidateId, long timestamp,
                                       List<Row> rows) {
    public CandidateEvaluationInput {
        rows = List.copyOf(rows);
        if (epoch <= 0 || iteration <= 0 || candidateId == null || candidateId.isBlank())
            throw new IllegalArgumentException("Invalid candidate identity");
    }

    public record Row(String vmId, String pmId, double currentCpuUtilization,
                      double predictedCpuUtilization, String previousPmId,
                      double vmMips, int vmPeCount, double vmRam, double vmStorage,
                      double pmMipsPerPe, int pmPeCount, double pmRam, double pmStorage,
                      double pmMaxPower, double pmStaticPower, double slaThreshold) {}
}
