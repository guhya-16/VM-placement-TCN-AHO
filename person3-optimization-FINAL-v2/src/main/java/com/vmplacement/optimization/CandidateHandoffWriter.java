package com.vmplacement.optimization;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/** Writes the exact Person 3 -> CloudSim candidate input contract. */
public final class CandidateHandoffWriter implements AutoCloseable {
    private final BufferedWriter out;
    public CandidateHandoffWriter(Path path) throws IOException {
        if (path.getParent() != null) Files.createDirectories(path.getParent());
        out = Files.newBufferedWriter(path);
        out.write("epoch,iteration,candidate_id,timestamp,vm_id,pm_id,current_cpu_utilization,predicted_cpu_utilization,previous_pm_id,vm_mips,vm_pe_count,vm_ram,vm_storage,pm_mips_per_pe,pm_pe_count,pm_ram,pm_storage,pm_max_power,pm_static_power,sla_threshold");
        out.newLine();
    }
    public synchronized void write(CandidateEvaluationInput input) {
        try {
            for (CandidateEvaluationInput.Row r : input.rows()) {
                out.write(input.epoch()+","+input.iteration()+","+input.candidateId()+","+input.timestamp()+","+
                        Csv.escape(r.vmId())+","+Csv.escape(r.pmId())+","+r.currentCpuUtilization()+","+
                        r.predictedCpuUtilization()+","+Csv.escape(r.previousPmId()==null?"":r.previousPmId())+","+
                        r.vmMips()+","+r.vmPeCount()+","+r.vmRam()+","+r.vmStorage()+","+r.pmMipsPerPe()+","+
                        r.pmPeCount()+","+r.pmRam()+","+r.pmStorage()+","+r.pmMaxPower()+","+r.pmStaticPower()+","+
                        r.slaThreshold());
                out.newLine();
            }
            out.flush();
        } catch (IOException e) { throw new RuntimeException(e); }
    }
    @Override public void close() throws IOException { out.close(); }
}
