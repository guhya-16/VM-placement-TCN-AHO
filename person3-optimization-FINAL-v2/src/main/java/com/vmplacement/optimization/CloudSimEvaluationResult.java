package com.vmplacement.optimization;

/** In-memory form of the CloudSim -> Person 3 candidate output. */
public final class CloudSimEvaluationResult {
    private final CandidateEvaluationOutput output;
    public CloudSimEvaluationResult(CandidateEvaluationOutput output) { this.output = output; }
    public boolean isFeasible(){return output.feasible();}
    public double getEnergyWh(){return output.energyWh();}
    public long getSlaViolations(){return output.slaViolations();}
    public int getActivePms(){return output.activePms();}
    public double getMeanUtilization(){return output.meanCpuUtilization();}
    public int getPlacementChanges(){return output.placementChangesFromPreviousState();}
    public double getMigrationCost(){return output.migrationCost();}
    public CandidateEvaluationOutput getOutput(){return output;}
}
