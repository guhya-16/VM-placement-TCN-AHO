package com.vmplacement.optimization;

import org.vmplacement.DatacenterState;
import org.vmplacement.PlacementEvaluationResult;
import org.vmplacement.PlacementEvaluator;
import org.vmplacement.PlacementRequest;
import org.vmplacement.PmSpec;
import org.vmplacement.VmPlacementSpec;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Bridges Person 3 candidate evaluations directly to CloudSim Plus physical simulation.
 * Records each candidate evaluation to CloudSim's authoritative results file.
 */
public final class CloudSimCandidateEvaluator implements CloudSimEvaluator {

    private final DatacenterState datacenterState;
    private final List<PmSpec> pmSpecs;
    private final File candidateResultsFile;

    public CloudSimCandidateEvaluator() {
        this(new DatacenterState(), new File("../results/candidate_evaluation_results.csv"));
    }

    public CloudSimCandidateEvaluator(DatacenterState datacenterState, File candidateResultsFile) {
        this.datacenterState = datacenterState;
        this.candidateResultsFile = candidateResultsFile;
        this.pmSpecs = new ArrayList<>();
        // 10 x PM1 (Hosts 0-9)
        for (int i = 0; i < 10; i++) {
            pmSpecs.add(new PmSpec(i, String.format("PM_%03d", i + 1), 2, 2660L, 4L * 1024, 160L * 1024, 135.0, 93.7));
        }
        // 6 x PM2 (Hosts 10-15)
        for (int i = 10; i < 16; i++) {
            pmSpecs.add(new PmSpec(i, String.format("PM_%03d", i + 1), 4, 3067L, 8L * 1024, 250L * 1024, 113.0, 42.3));
        }
        // 4 x PM3 (Hosts 16-19)
        for (int i = 16; i < 20; i++) {
            pmSpecs.add(new PmSpec(i, String.format("PM_%03d", i + 1), 12, 3067L, 16L * 1024, 500L * 1024, 222.0, 58.4));
        }

        // Initialize candidate results file in results directory if not exists
        if (candidateResultsFile != null) {
            File parent = candidateResultsFile.getParentFile();
            if (parent != null && !parent.exists()) {
                parent.mkdirs();
            }
            if (!candidateResultsFile.exists() || candidateResultsFile.length() == 0) {
                try (FileWriter fw = new FileWriter(candidateResultsFile, false)) {
                    fw.write("epoch,iteration,candidate_id,timestamp,feasible,energy_wh,sla_violations,active_pms,mean_cpu_utilization,placement_changes_from_previous_state,migration_cost\n");
                } catch (IOException e) {
                    System.err.println("Failed to write header to " + candidateResultsFile + ": " + e.getMessage());
                }
            }
        }
    }

    @Override
    public CandidateEvaluationOutput evaluate(CandidateEvaluationInput input) {
        List<VmPlacementSpec> vmPlacements = new ArrayList<>();
        for (CandidateEvaluationInput.Row row : input.rows()) {
            int pmIndex = parsePmIndex(row.pmId());
            Integer prevPmIndex = parseNullablePmIndex(row.previousPmId());
            vmPlacements.add(new VmPlacementSpec(
                    row.vmId(),
                    pmIndex,
                    row.currentCpuUtilization(),
                    row.predictedCpuUtilization(),
                    prevPmIndex,
                    row.vmPeCount(),
                    row.vmMips(),
                    row.vmRam(),
                    row.vmStorage()
            ));
        }

        PlacementRequest request = new PlacementRequest(
                input.epoch(),
                input.iteration(),
                input.candidateId(),
                input.timestamp(),
                "AHO",
                input.rows().get(0).slaThreshold(),
                vmPlacements,
                pmSpecs
        );

        PlacementEvaluationResult result = PlacementEvaluator.evaluate(request, datacenterState);

        // Record candidate evaluation in CloudSim results file
        if (candidateResultsFile != null) {
            try (FileWriter fw = new FileWriter(candidateResultsFile, true)) {
                fw.write(String.format(java.util.Locale.US,
                        "%d,%d,%s,%d,%b,%.4f,%d,%d,%.6f,%d,%.4f\n",
                        result.getEpoch(),
                        result.getIteration(),
                        result.getCandidateId(),
                        input.timestamp(),
                        result.isFeasible(),
                        result.getEnergyWh(),
                        result.getSlaViolations(),
                        result.getActivePms(),
                        result.getMeanCpuUtilization(),
                        result.getPlacementChangesFromPreviousState(),
                        result.getMigrationCost()
                ));
            } catch (IOException e) {
                System.err.println("Failed to record candidate result: " + e.getMessage());
            }
        }

        return new CandidateEvaluationOutput(
                result.getEpoch(),
                result.getIteration(),
                result.getCandidateId(),
                result.isFeasible(),
                result.getEnergyWh(),
                result.getSlaViolations(),
                result.getActivePms(),
                result.getMeanCpuUtilization(),
                result.getPlacementChangesFromPreviousState(),
                result.getMigrationCost()
        );
    }

    public void commitFinalPlacement(int epoch, long timestamp, Map<String, Integer> finalPlacement) {
        datacenterState.commitFinalPlacement(epoch, timestamp, finalPlacement);
        if (finalPlacement != null) {
            File timelineFile = new File("../results/final_placement_timeline.csv");
            boolean isNew = !timelineFile.exists() || timelineFile.length() == 0;
            try (FileWriter fw = new FileWriter(timelineFile, true)) {
                if (isNew) {
                    fw.write("epoch,timestamp,vm_id,pm_id\n");
                }
                for (Map.Entry<String, Integer> e : finalPlacement.entrySet()) {
                    fw.write(String.format("%d,%d,%s,PM_%03d\n", epoch, timestamp, e.getKey(), e.getValue() + 1));
                }
            } catch (IOException e) {
                System.err.println("Failed to write timeline: " + e.getMessage());
            }

            File finalEpochFile = new File("../results/final_epoch_placement.csv");
            try (FileWriter fw = new FileWriter(finalEpochFile, false)) {
                fw.write("epoch,timestamp,vm_id,pm_id\n");
                for (Map.Entry<String, Integer> e : finalPlacement.entrySet()) {
                    fw.write(String.format("%d,%d,%s,PM_%03d\n", epoch, timestamp, e.getKey(), e.getValue() + 1));
                }
            } catch (IOException e) {
                System.err.println("Failed to write final epoch placement: " + e.getMessage());
            }
        }
    }

    private static int parsePmIndex(String pmId) {
        if (pmId == null) return 0;
        String clean = pmId.toUpperCase().replace("PM_", "").replace("PM", "").trim();
        try {
            int id = Integer.parseInt(clean);
            return (id > 0) ? id - 1 : 0;
        } catch (NumberFormatException e) {
            return 0;
        }
    }

    private static Integer parseNullablePmIndex(String pmId) {
        if (pmId == null || pmId.isBlank()) return null;
        String clean = pmId.toUpperCase().replace("PM_", "").replace("PM", "").trim();
        try {
            int id = Integer.parseInt(clean);
            return (id > 0) ? id - 1 : 0;
        } catch (NumberFormatException e) {
            return null;
        }
    }
}
