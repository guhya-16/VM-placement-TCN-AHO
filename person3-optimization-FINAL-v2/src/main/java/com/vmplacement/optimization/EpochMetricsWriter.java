package com.vmplacement.optimization;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.util.Locale;

/**
 * Thread-safe incremental writer for completed decision epoch metrics.
 *
 * Flushes each record immediately so external dashboard monitors can read
 * real-time optimization metrics without waiting for the full multi-epoch run to finish.
 */
public final class EpochMetricsWriter implements AutoCloseable {

    private final File file;
    private final String runId;
    private final PrintWriter writer;

    public EpochMetricsWriter(File file, String runId) throws IOException {
        this.file = file;
        this.runId = (runId == null || runId.isBlank()) ? "default_run" : runId.trim();

        File parent = file.getParentFile();
        if (parent != null && !parent.exists()) {
            parent.mkdirs();
        }

        boolean writeHeader = !file.exists() || file.length() == 0;
        this.writer = new PrintWriter(
                new OutputStreamWriter(new FileOutputStream(file, true), StandardCharsets.UTF_8),
                true
        );

        if (writeHeader) {
            writer.println("run_id,epoch,timestamp,selected_candidate_id,energy_wh,sla_violations,active_pms,mean_cpu_utilization,placement_changes_from_previous_state,migration_cost,best_fitness,completed");
            writer.flush();
        }
    }

    public synchronized void write(int epoch, long timestamp, CloudSimEvaluationResult eval, double bestFitness, boolean completed) {
        String candId = (eval != null && eval.getOutput() != null && eval.getOutput().candidateId() != null)
                ? eval.getOutput().candidateId()
                : "UNKNOWN";

        double energy = eval != null ? eval.getEnergyWh() : Double.NaN;
        long sla = eval != null ? eval.getSlaViolations() : 0;
        int activePms = eval != null ? eval.getActivePms() : 0;
        double meanUtil = eval != null ? eval.getMeanUtilization() : 0.0;
        int placementChanges = eval != null ? eval.getPlacementChanges() : 0;
        double migrationCost = eval != null ? eval.getMigrationCost() : 0.0;

        writer.printf(Locale.US,
                "%s,%d,%d,%s,%.4f,%d,%d,%.6f,%d,%.4f,%.8f,%b%n",
                runId,
                epoch,
                timestamp,
                candId,
                energy,
                sla,
                activePms,
                meanUtil,
                placementChanges,
                migrationCost,
                bestFitness,
                completed
        );
        writer.flush();
    }

    public String getRunId() {
        return runId;
    }

    public File getFile() {
        return file;
    }

    @Override
    public void close() {
        if (writer != null) {
            writer.flush();
            writer.close();
        }
    }
}
