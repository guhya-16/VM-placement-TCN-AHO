package com.vmplacement.optimization;

/** Exact CloudSim -> Person 3 candidate-evaluation handoff. */
public record CandidateEvaluationOutput(int epoch, int iteration, String candidateId,
                                        boolean feasible, double energyWh, long slaViolations,
                                        int activePms, double meanCpuUtilization,
                                        int placementChangesFromPreviousState, double migrationCost) {}
