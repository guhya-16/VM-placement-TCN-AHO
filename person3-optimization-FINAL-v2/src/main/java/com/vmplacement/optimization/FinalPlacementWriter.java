package com.vmplacement.optimization;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/** Exact Person 3 -> Person 2 final committed-placement handoff. */
public final class FinalPlacementWriter implements AutoCloseable {
    private final BufferedWriter out;
    public FinalPlacementWriter(Path path) throws IOException {
        if (path.getParent() != null) Files.createDirectories(path.getParent());
        out = Files.newBufferedWriter(path);
        out.write("epoch,timestamp,vm_id,pm_id,current_cpu_utilization,predicted_cpu_utilization,prediction_risk,risk_state,best_iteration");
        out.newLine();
    }
    public void write(int epochIndex, EpochState state, PlacementSolution solution, int bestIteration) {
        int[] mapping = solution.getVmToPm();
        try {
            for (int i=0;i<state.getActiveVms().size();i++) {
                VM vm = state.getActiveVms().get(i);
                String pm = state.getPms().get(mapping[i]).getId();
                out.write(epochIndex+","+state.getTimestamp()+","+Csv.escape(vm.getId())+","+Csv.escape(pm)+","+
                        vm.getCurrentCpuUtilization()+","+vm.getPredictedMeanCpu()+","+vm.getRiskScore()+","+
                        Csv.escape(vm.getRiskState())+","+bestIteration);
                out.newLine();
            }
        } catch (IOException e) { throw new RuntimeException(e); }
    }
    public void flush() throws IOException { out.flush(); }
    @Override public void close() throws IOException { out.close(); }
}
