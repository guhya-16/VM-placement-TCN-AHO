package com.vmplacement.optimization;

import java.util.ArrayList;
import java.util.List;

/** Builds the exact candidate handoff, calls Person 2 synchronously, and records the returned output. */
public final class CandidateEvaluationBridge {
    private final CloudSimEvaluator evaluator;
    private final CandidateHandoffWriter inputWriter;
    private final CandidateEvaluationOutputWriter outputWriter;
    private final double slaThreshold;

    public CandidateEvaluationBridge(CloudSimEvaluator evaluator, CandidateHandoffWriter inputWriter,
                                     CandidateEvaluationOutputWriter outputWriter, double slaThreshold) {
        this.evaluator=evaluator; this.inputWriter=inputWriter; this.outputWriter=outputWriter; this.slaThreshold=slaThreshold;
    }

    public CloudSimEvaluationResult evaluate(int epochIndex, int iteration, int candidateNumber,
                                             EpochState state, PlacementSolution candidate) {
        String candidateId = "E"+epochIndex+"-I"+iteration+"-C"+candidateNumber;
        List<CandidateEvaluationInput.Row> rows = new ArrayList<>();
        int[] mapping=candidate.getVmToPm();
        for(int i=0;i<state.getActiveVms().size();i++){
            VM vm=state.getActiveVms().get(i); PM pm=state.getPms().get(mapping[i]);
            rows.add(new CandidateEvaluationInput.Row(vm.getId(), pm.getId(), vm.getCurrentCpuUtilization(),
                    vm.getPredictedMeanCpu(), state.getPreviousPlacement().getPmForVm(vm.getId()),
                    vm.getMips(), vm.getPes(), vm.getRamMb(), vm.getStorageMb(), pm.getMipsPerPe(),
                    pm.getPes(), pm.getRamMb(), pm.getStorageMb(), pm.getMaxPowerWatts(),
                    pm.getStaticPowerWatts(), slaThreshold));
        }
        CandidateEvaluationInput input=new CandidateEvaluationInput(epochIndex,iteration,candidateId,state.getTimestamp(),rows);
        inputWriter.write(input);
        CandidateEvaluationOutput output=evaluator.evaluate(input);
        if(output==null) throw new IllegalStateException("CloudSim evaluator returned null for "+candidateId);
        validateIdentity(input,output);
        outputWriter.write(output);
        return new CloudSimEvaluationResult(output);
    }
    private static void validateIdentity(CandidateEvaluationInput in, CandidateEvaluationOutput out){
        if(out.epoch()!=in.epoch() || out.iteration()!=in.iteration() || !out.candidateId().equals(in.candidateId()))
            throw new IllegalStateException("CloudSim output identity does not match candidate input "+in.candidateId());
    }
}
