package com.vmplacement.optimization;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.List;

public class Main {

    private static final String INPUT_FILE =
            "input/risk_state.csv";

    private static final String OUTPUT_FILE =
            "experiment_results.csv";

    private static final String VM_MANIFEST_FILE =
            "vm_manifest.csv";

    private static final int POPULATION_SIZE = 20;

    private static final int MAX_ITERATIONS = 50;

    public static void main(String[] args) {

        System.out.println("==============================================");
        System.out.println("   PERSON 3 - VM PLACEMENT EXPERIMENT");
        System.out.println("   STANDARD HO VS ADAPTIVE HO");
        System.out.println("==============================================");

        // =========================================================
        // 1. LOAD RISK STATE DATA
        // =========================================================

        List<DecisionEpoch> epochs;

        try {

            epochs = CSVWorkloadReader.readDecisionEpochs(
                    INPUT_FILE
            );

        } catch (Exception e) {

            System.out.println(
                    "Error loading risk_state.csv: "
                            + e.getMessage()
            );

            return;
        }

        if (epochs.isEmpty()) {

            System.out.println(
                    "No decision epochs found."
            );

            return;
        }

        System.out.println();
        System.out.println("[1] DATA LOADED");
        System.out.println("----------------------------------------------");

        System.out.println(
                "Total decision epochs : "
                        + epochs.size()
        );

        System.out.println(
                "First timestamp       : "
                        + epochs.get(0)
                        .getDecisionTimestamp()
        );

        System.out.println(
                "Last timestamp        : "
                        + epochs.get(epochs.size() - 1)
                        .getDecisionTimestamp()
        );

        // =========================================================
        // 2. CREATE PHYSICAL MACHINES
        // =========================================================

        // Full 20-PM configuration matching Person 2's CloudSim host manifest.
        // Hosts 0-9:  2 PEs, 2660 MIPS/PE, 4 GB RAM
        // Hosts 10-15: 4 PEs, 3067 MIPS/PE, 8 GB RAM
        // Hosts 16-19: 12 PEs, 3067 MIPS/PE, 16 GB RAM
        java.util.ArrayList<PM> pmList = new java.util.ArrayList<>();

        for (int i = 1; i <= 10; i++) {
            pmList.add(new PM(
                    String.format("PM_%03d", i),
                    2, 2660.0, 4096.0, 163840.0, 135.0, 93.7));
        }

        for (int i = 11; i <= 16; i++) {
            pmList.add(new PM(
                    String.format("PM_%03d", i),
                    4, 3067.0, 8192.0, 256000.0, 113.0, 42.3));
        }

        for (int i = 17; i <= 20; i++) {
            pmList.add(new PM(
                    String.format("PM_%03d", i),
                    12, 3067.0, 16384.0, 512000.0, 222.0, 58.4));
        }

        List<PM> pms = pmList;

        System.out.println();
        System.out.println("[2] PHYSICAL MACHINES");
        System.out.println("----------------------------------------------");

        for (PM pm : pms) {

            System.out.println(
                    pm.getId()
                            + " | CPU = "
                            + pm.getCpuCapacity()
                            + "% | RAM = "
                            + pm.getRamCapacity()
                            + " MB"
            );
        }

        // =========================================================
        // 3. PREPARE OUTPUT FILE
        // =========================================================

        File outputFile = new File(OUTPUT_FILE);
        File manifestFile = new File(VM_MANIFEST_FILE);

        try (
                PrintWriter writer =
                        new PrintWriter(
                                new FileWriter(outputFile)
                        );
                PrintWriter manifestWriter =
                        new PrintWriter(
                                new FileWriter(manifestFile)
                        )
        ) {

            manifestWriter.println(
                    "epoch,"
                            + "vm_id,"
                            + "pes,"
                            + "mips,"
                            + "ram_mb,"
                            + "storage_mb"
            );

            writer.println(
                    "epoch,"
                            + "timestamp,"
                            + "vm_count,"
                            + "workload_variation,"
                            + "prediction_risk,"
                            + "adaptive_signal,"
                            + "adaptive_mode,"
                            + "exploration_probability,"
                            + "standard_fitness,"
                            + "adaptive_fitness,"
                            + "standard_time_ms,"
                            + "adaptive_time_ms,"
                            + "standard_feasible,"
                            + "adaptive_feasible,"
                            + "standard_placement,"
                            + "adaptive_placement"
            );

            // =====================================================
            // 4. PROCESS ALL DECISION EPOCHS
            // =====================================================

            System.out.println();
            System.out.println("[3] RUNNING EXPERIMENT");
            System.out.println("----------------------------------------------");

            int totalEpochs = epochs.size();

            for (int i = 0; i < totalEpochs; i++) {

                DecisionEpoch epoch =
                        epochs.get(i);

                List<VM> epochVMs =
                        DecisionEpochConverter.toVMs(
                                epoch
                        );

                // =================================================
                // STANDARD HO
                // =================================================

                long standardStart =
                        System.nanoTime();

                long epochSeed = 100_000L + i;

                List<PlacementSolution> commonPopulation =
                        createCommonInitialPopulation(
                                epochVMs,
                                pms,
                                POPULATION_SIZE,
                                epochSeed
                        );

                if (commonPopulation.isEmpty()) {
                    System.out.println(
                            "Skipping epoch " + (i + 1)
                                    + ": unable to create a feasible initial population."
                    );
                    continue;
                }

                HippopotamusOptimization standardHO =
                        new HippopotamusOptimization(
                                epochVMs,
                                pms,
                                POPULATION_SIZE,
                                MAX_ITERATIONS,
                                epochSeed + 1,
                                commonPopulation,
                                false
                        );

                PlacementSolution standardSolution =
                        standardHO.optimize();

                long standardEnd =
                        System.nanoTime();

                double standardTimeMs =
                        (standardEnd - standardStart)
                                / 1_000_000.0;

                double standardFitness =
                        standardHO.getBestFitness();

                // =================================================
                // ADAPTIVE HO
                // =================================================

                long adaptiveStart =
                        System.nanoTime();

                AdaptiveHippopotamusOptimization adaptiveHO =
                        new AdaptiveHippopotamusOptimization(
                                epochVMs,
                                pms,
                                POPULATION_SIZE,
                                MAX_ITERATIONS,
                                epochSeed + 2,
                                epoch.getRiskStates(),
                                commonPopulation,
                                false
                        );

                PlacementSolution adaptiveSolution =
                        adaptiveHO.optimize();

                long adaptiveEnd =
                        System.nanoTime();

                double adaptiveTimeMs =
                        (adaptiveEnd - adaptiveStart)
                                / 1_000_000.0;

                double adaptiveFitness =
                        adaptiveHO.getBestFitness();

                // =================================================
                // FEASIBILITY CHECK
                // =================================================

                // Current capacity is a hard constraint.
                // Predicted overload is intentionally handled as a
                // soft SLA/risk penalty inside FitnessFunction.
                boolean standardFeasible =
                        PlacementValidator.isFeasible(
                                epochVMs,
                                pms,
                                standardSolution
                        );

                boolean adaptiveFeasible =
                        PlacementValidator.isFeasible(
                                epochVMs,
                                pms,
                                adaptiveSolution
                        );

                // =================================================
                // ADAPTIVE INFORMATION
                // =================================================

                double workloadVariation =
        adaptiveHO.getWorkloadVariation();

double predictionRisk =
        adaptiveHO.getPredictionRisk();

                
                 
                double adaptiveSignal =
                        adaptiveHO.getAdaptiveSignal();

                String adaptiveMode =
                        adaptiveHO.getCurrentAdaptiveMode();

                double explorationProbability =
                        adaptiveHO
                                .getCurrentExplorationProbability();

                // =================================================
                // WRITE RESULT
                // =================================================

                writer.println(
                        (i + 1)
                                + ","
                                + epoch.getDecisionTimestamp()
                                + ","
                                + epoch.getVmCount()
                                + ","
                                + String.format(
                                        "%.6f",
                                        workloadVariation
                                )
                                + ","
                                + String.format(
        "%.6f",
        predictionRisk
)
                                + ","
                                + String.format(
                                        "%.6f",
                                        adaptiveSignal
                                )
                                + ","
                                + adaptiveMode
                                + ","
                                + String.format(
                                        "%.6f",
                                        explorationProbability
                                )
                                + ","
                                + String.format(
                                        "%.10f",
                                        standardFitness
                                )
                                + ","
                                + String.format(
                                        "%.10f",
                                        adaptiveFitness
                                )
                                + ","
                                + String.format(
                                        "%.4f",
                                        standardTimeMs
                                )
                                + ","
                                + String.format(
                                        "%.4f",
                                        adaptiveTimeMs
                                )
                                + ","
                                + standardFeasible
                                + ","
                                + adaptiveFeasible
                                + ","
                                + quote(
                                        standardSolution.toString()
                                )
                                + ","
                                + quote(
                                        adaptiveSolution.toString()
                                )
                );

                for (VM vm : epochVMs) {
                    manifestWriter.println(
                            (i + 1)
                                    + ","
                                    + vm.getId()
                                    + ","
                                    + vm.getPes()
                                    + ","
                                    + String.format(java.util.Locale.US, "%.0f", vm.getMips())
                                    + ","
                                    + String.format(java.util.Locale.US, "%.0f", vm.getRamMb())
                                    + ","
                                    + String.format(java.util.Locale.US, "%.0f", vm.getStorageMb())
                    );
                }

                // =================================================
                // PROGRESS
                // =================================================

                if (
                        i == 0
                                || (i + 1) % 100 == 0
                                || i == totalEpochs - 1
                ) {

                    System.out.println(
                            "Processed "
                                    + (i + 1)
                                    + " / "
                                    + totalEpochs
                                    + " epochs"
                    );
                }
            }

            writer.flush();
            manifestWriter.flush();

        } catch (Exception e) {

            System.out.println(
                    "Error writing experiment results: "
                            + e.getMessage()
            );

            e.printStackTrace();

            return;
        }

        // =========================================================
        // 5. COMPLETE
        // =========================================================

        System.out.println();
        System.out.println("==============================================");
        System.out.println("   EXPERIMENT COMPLETE");
        System.out.println("==============================================");

        System.out.println(
                "Results saved to : "
                        + OUTPUT_FILE
        );

        System.out.println(
                "Total epochs     : "
                        + epochs.size()
        );

        System.out.println(
                "Population size   : "
                        + POPULATION_SIZE
        );

        System.out.println(
                "Iterations        : "
                        + MAX_ITERATIONS
        );

        System.out.println();
        System.out.println(
                "Next step: analyze experiment_results.csv"
        );
    }

    // =============================================================
    // COMMON INITIAL POPULATION
    // =============================================================

    private static List<PlacementSolution> createCommonInitialPopulation(
            List<VM> vms,
            List<PM> pms,
            int size,
            long seed) {

        List<PlacementSolution> population = new java.util.ArrayList<>();
        java.util.Random random = new java.util.Random(seed);

        int attempts = 0;
        int maxAttempts = size * 200;

        while (population.size() < size && attempts < maxAttempts) {
            attempts++;

            PlacementSolution solution =
                    PlacementRepair.createRandomFeasibleSolution(
                            vms,
                            pms,
                            random,
                            0.65
                    );

            if (solution == null
                    || !PlacementValidator.isFeasible(vms, pms, solution)) {
                continue;
            }

            boolean duplicate = false;

            for (PlacementSolution existing : population) {
                if (java.util.Arrays.equals(
                        existing.getVmToPm(),
                        solution.getVmToPm())) {
                    duplicate = true;
                    break;
                }
            }

            if (!duplicate) {
                population.add(solution.copy());
            }
        }

        return population;
    }

    // =============================================================
    // CSV QUOTE HELPER
    // =============================================================

    private static String quote(String value) {

        return "\""
                + value.replace("\"", "\"\"")
                + "\"";
    }
}