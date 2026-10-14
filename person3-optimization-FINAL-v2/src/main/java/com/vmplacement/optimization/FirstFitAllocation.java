package com.vmplacement.optimization;

import java.util.List;

/** Deterministic First Fit placement baseline using the existing feasibility rules. */
public final class FirstFitAllocation {
    private final EpochState epoch;
    private final CandidateEvaluationBridge bridge;
    private final FitnessFunction fitnessFunction;
    private int currentEpochIndex;
    private CloudSimEvaluationResult evaluation;
    private PlacementSolution bestSolution;
    private double bestFitness = Double.POSITIVE_INFINITY;

    public FirstFitAllocation(EpochState epoch, CandidateEvaluationBridge bridge, FitnessFunction fitnessFunction) {
        if (epoch.getActiveVms().isEmpty()) {
            throw new IllegalArgumentException("Epoch has no active VMs: " + epoch.getTimestamp());
        }
        this.epoch = epoch;
        this.bridge = bridge;
        this.fitnessFunction = fitnessFunction;
    }

    public PlacementSolution optimize() {
        List<VM> vms = epoch.getActiveVms();
        List<PM> pms = epoch.getPms();
        int[] mapping = new int[vms.size()];
        double[] cpu = new double[pms.size()];
        double[] ram = new double[pms.size()];
        double[] storage = new double[pms.size()];
        int[] pes = new int[pms.size()];

        // First Fit: preserve the input VM order and choose the first PM that fits.
        for (int vmIndex = 0; vmIndex < vms.size(); vmIndex++) {
            VM vm = vms.get(vmIndex);
            int selectedPm = -1;
            double requiredCpu = vm.getMips() * vm.getCurrentCpuUtilization() / 100.0;
            for (int pmIndex = 0; pmIndex < pms.size(); pmIndex++) {
                PM pm = pms.get(pmIndex);
                if (pes[pmIndex] + vm.getPes() <= pm.getPes()
                        && cpu[pmIndex] + requiredCpu <= pm.getTotalMips() + 1e-9
                        && ram[pmIndex] + vm.getRamMb() <= pm.getRamMb() + 1e-9
                        && storage[pmIndex] + vm.getStorageMb() <= pm.getStorageMb() + 1e-9) {
                    selectedPm = pmIndex;
                    break;
                }
            }
            if (selectedPm < 0) {
                throw new IllegalStateException("First Fit could not place VM " + vm.getId()
                        + " at timestamp " + epoch.getTimestamp());
            }
            mapping[vmIndex] = selectedPm;
            cpu[selectedPm] += requiredCpu;
            ram[selectedPm] += vm.getRamMb();
            storage[selectedPm] += vm.getStorageMb();
            pes[selectedPm] += vm.getPes();
        }

        PlacementSolution solution = new PlacementSolution(mapping);
        if (!PlacementValidator.isFeasible(vms, pms, solution)) {
            throw new IllegalStateException("First Fit produced a placement that failed existing validation at timestamp "
                    + epoch.getTimestamp());
        }
        bestSolution = solution.copy();
        evaluation = bridge.evaluate(currentEpochIndex, 1, 1, epoch, solution);
        bestFitness = fitnessFunction.score(List.of(evaluation)).get(0);
        return solution;
    }

    public void setEpochIndex(int epochIndex) {
        if (epochIndex <= 0) throw new IllegalArgumentException("epochIndex must be positive");
        this.currentEpochIndex = epochIndex;
    }

    public double getBestFitness() { return bestFitness; }
    public PlacementSolution getBestSolution() { return bestSolution == null ? null : bestSolution.copy(); }
    public CloudSimEvaluationResult getBestEvaluation() { return evaluation; }
    public int getBestIteration() { return 1; }
}
