package org.vmplacement;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Clean data transfer structure representing the physical simulation metrics
 * returned from the Person 2 CloudSim evaluator to the Person 3 Optimizer.
 */
public class PlacementEvaluationResult {

    private final int epoch;
    private final int iteration;
    private final String candidateId;
    private final boolean feasible;
    private final double energyWh;
    private final int slaViolations;
    private final double slaViolationRate;
    private final int activePms;
    private final double meanCpuUtilization;
    private final int placementChangesFromPreviousState;
    private final double migrationCost;
    private final Map<Integer, Double> hostCpuUtilizationMap;
    private final String message;

    public PlacementEvaluationResult(int epoch, int iteration, String candidateId,
                                     boolean feasible, double energyWh,
                                     int slaViolations, double slaViolationRate,
                                     int activePms, double meanCpuUtilization,
                                     int placementChangesFromPreviousState,
                                     double migrationCost,
                                     Map<Integer, Double> hostCpuUtilizationMap,
                                     String message) {
        this.epoch = epoch;
        this.iteration = iteration;
        this.candidateId = candidateId != null ? candidateId : "CANDIDATE_0";
        this.feasible = feasible;
        this.energyWh = energyWh;
        this.slaViolations = slaViolations;
        this.slaViolationRate = slaViolationRate;
        this.activePms = activePms;
        this.meanCpuUtilization = meanCpuUtilization;
        this.placementChangesFromPreviousState = placementChangesFromPreviousState;
        this.migrationCost = migrationCost;
        this.hostCpuUtilizationMap = (hostCpuUtilizationMap != null)
                ? new LinkedHashMap<>(hostCpuUtilizationMap)
                : new LinkedHashMap<>();
        this.message = message != null ? message : "SUCCESS";
    }

    public static PlacementEvaluationResult infeasible(int epoch, int iteration, String candidateId, String message) {
        return new PlacementEvaluationResult(
                epoch, iteration, candidateId, false,
                Double.POSITIVE_INFINITY, -1, 1.0, 0, 0.0,
                0, 0.0, Collections.emptyMap(), message
        );
    }

    public int getEpoch() {
        return epoch;
    }

    public int getIteration() {
        return iteration;
    }

    public String getCandidateId() {
        return candidateId;
    }

    public boolean isFeasible() {
        return feasible;
    }

    public double getEnergyWh() {
        return energyWh;
    }

    public int getSlaViolations() {
        return slaViolations;
    }

    public double getSlaViolationRate() {
        return slaViolationRate;
    }

    public int getActivePms() {
        return activePms;
    }

    public double getMeanCpuUtilization() {
        return meanCpuUtilization;
    }

    public int getPlacementChangesFromPreviousState() {
        return placementChangesFromPreviousState;
    }

    public double getMigrationCost() {
        return migrationCost;
    }

    public Map<Integer, Double> getHostCpuUtilizationMap() {
        return Collections.unmodifiableMap(hostCpuUtilizationMap);
    }

    public String getMessage() {
        return message;
    }

    @Override
    public String toString() {
        return String.format(
                "PlacementEvaluationResult[epoch=%d, iter=%d, id=%s, feasible=%b, energyWh=%.4f, slaViolations=%d (%.2f%%), activePms=%d, meanCpu=%.4f, changes=%d, migCost=%.2f, msg='%s']",
                epoch, iteration, candidateId, feasible, energyWh, slaViolations, slaViolationRate * 100.0,
                activePms, meanCpuUtilization, placementChangesFromPreviousState, migrationCost, message
        );
    }
}
