package com.vmplacement.optimization;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/** Writes the exact CloudSim -> Person 3 candidate output contract. */
public final class CandidateEvaluationOutputWriter implements AutoCloseable {
    private final BufferedWriter out;
    public CandidateEvaluationOutputWriter(Path path) throws IOException {
        if (path.getParent() != null) Files.createDirectories(path.getParent());
        out = Files.newBufferedWriter(path);
        out.write("epoch,iteration,candidate_id,feasible,energy_wh,sla_violations,active_pms,mean_cpu_utilization,placement_changes_from_previous_state,migration_cost");
        out.newLine();
    }
    public synchronized void write(CandidateEvaluationOutput r) {
        try {
            out.write(r.epoch()+","+r.iteration()+","+Csv.escape(r.candidateId())+","+r.feasible()+","+
                    r.energyWh()+","+r.slaViolations()+","+r.activePms()+","+r.meanCpuUtilization()+","+
                    r.placementChangesFromPreviousState()+","+r.migrationCost());
            out.newLine();
            out.flush();
        } catch (IOException e) { throw new RuntimeException(e); }
    }
    @Override public void close() throws IOException { out.close(); }
}
