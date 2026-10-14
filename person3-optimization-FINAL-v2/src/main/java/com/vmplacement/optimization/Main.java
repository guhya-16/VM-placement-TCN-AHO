package com.vmplacement.optimization;

import java.nio.file.Path;
import java.util.List;

/**
 * Person 3 orchestration entry point.
 *
 * Uses the CloudSim evaluator and infrastructure-provider interfaces.
 * The current unconfigured implementations fail closed until Person 2
 * supplies the real CloudSim integration.
 */
public final class Main {

    private Main() {}

    public static void main(String[] args) throws Exception {

        Path configPath = Path.of(
                args.length > 0
                        ? args[0]
                        : "config/application.properties"
        );

        Config cfg = Config.load(configPath);
System.out.println("\n===== VM Placement Algorithm Selection =====");
System.out.println("1. WA_AHO - Adaptive Hippopotamus Optimization");
System.out.println("2. STANDARD_HO - Standard Hippopotamus Optimization");
System.out.println("3. FIRST_FIT - First Fit Allocation");
System.out.print("Enter your choice (1-3): ");

java.util.Scanner scanner = new java.util.Scanner(System.in);
String choice = scanner.nextLine().trim();

String algorithm = switch (choice) {
    case "1", "WA_AHO" -> "WA_AHO";
    case "2", "STANDARD_HO" -> "STANDARD_HO";
    case "3", "FIRST_FIT" -> "FIRST_FIT";
    default -> throw new IllegalArgumentException(
        "Invalid choice: " + choice + ". Please enter 1, 2, or 3."
    );
};

System.out.println("Selected algorithm: " + algorithm);
System.out.println("Starting VM placement workflow...");


        List<DecisionEpoch> epochs =
                CSVWorkloadReader.readDecisionEpochs(
                        cfg.path("input.risk_state")
                );

        if (epochs.isEmpty()) {
            throw new IllegalStateException(
                    "Person 1 handoff contains no decision epochs."
            );
        }

        HandoffValidator.validate(epochs);

        /*
         * Production integration points.
         *
         * Person 2 will later replace these unconfigured implementations
         * with the real CloudSim-backed implementations.
         */
        InfrastructureProvider infrastructure =
                new CloudSimInfrastructureProvider();

        String powerModelStr = cfg.getOrDefault("energy.power_state_model", "ACTIVE_SERVERS_ONLY");
        org.vmplacement.HostPowerStateModel powerStateModel = powerModelStr.equalsIgnoreCase("FACILITY_ALWAYS_ON")
                ? org.vmplacement.HostPowerStateModel.FACILITY_ALWAYS_ON
                : org.vmplacement.HostPowerStateModel.ACTIVE_SERVERS_ONLY;
        double standbyWatts = cfg.decimalOrDefault("energy.standby_power_watts", 0.0);

        CloudSimCandidateEvaluator evaluator =
                new CloudSimCandidateEvaluator(
                        new org.vmplacement.DatacenterState(),
                        resolveResultsFile(cfg.getOrDefault("output.candidate_results", "candidate_evaluation_results.csv")),
                        powerStateModel,
                        standbyWatts
                );
        evaluator.setAlgorithmName(algorithm);

        FitnessFunction fitness = new FitnessFunction(
                cfg.decimal("fitness.energy_weight"),
                cfg.decimal("fitness.sla_weight"),
                cfg.decimal("fitness.active_pm_weight"),
                cfg.decimal("fitness.mean_utilization_weight"),
                cfg.decimal("fitness.migration_weight"),
                cfg.decimal("fitness.energy_scale_wh"),
                cfg.decimal("fitness.sla_scale"),
                cfg.decimal("fitness.active_pm_scale"),
                cfg.decimal("fitness.mean_utilization_scale"),
                cfg.decimal("fitness.migration_scale")
        );

        AdaptiveSignalCalculator signal =
                new AdaptiveSignalCalculator(
                        cfg.decimal("adaptive.volatility_weight"),
                        cfg.decimal("adaptive.risk_weight"),
                        cfg.decimal("adaptive.min_exploration"),
                        cfg.decimal("adaptive.max_exploration")
                );

        PlacementState previous = PlacementState.empty();

        int maxEpochs = (args.length > 1) ? Integer.parseInt(args[1]) : epochs.size();
        int maxIterations = (args.length > 2) ? Integer.parseInt(args[2]) : cfg.integer("aho.max_iterations");
        int populationSize = (args.length > 3) ? Integer.parseInt(args[3]) : cfg.integer("aho.population_size");

        String runId = System.getProperty("run.id");
        if (runId == null || runId.isBlank()) {
            runId = System.getenv("RUN_ID");
        }
        if (runId == null || runId.isBlank()) {
            runId = cfg.getOrDefault("run.id", "");
        }
        if (args.length > 4 && args[4] != null && !args[4].isBlank()) {
            runId = args[4];
        }
        if (runId == null || runId.isBlank()) {
            runId = new java.text.SimpleDateFormat("yyyyMMdd_HHmmss").format(new java.util.Date());
        }

        java.io.File epochMetricsFile = resolveResultsFile(cfg.getOrDefault("output.epoch_metrics", "epoch_metrics.csv"));

        try (
                CandidateHandoffWriter candidateIn =
                        new CandidateHandoffWriter(
                                cfg.path("output.candidate_input")
                        );

                CandidateEvaluationOutputWriter candidateOut =
                        new CandidateEvaluationOutputWriter(
                                cfg.path("output.candidate_output")
                        );

                FinalPlacementWriter finalWriter =
                        new FinalPlacementWriter(
                                cfg.path("output.final_placement")
                        );

                EpochMetricsWriter epochMetricsWriter =
                        new EpochMetricsWriter(
                                epochMetricsFile,
                                runId
                        )
        ) {

            CandidateEvaluationBridge bridge =
                    new CandidateEvaluationBridge(
                            evaluator,
                            candidateIn,
                            candidateOut,
                            cfg.decimal("sla.threshold")
                    );

            long seedBase = cfg.longValue("aho.seed");

            for (
                    int epochIndex = 0;
                    epochIndex < Math.min(epochs.size(), maxEpochs);
                    epochIndex++
            ) {

                DecisionEpoch epoch = epochs.get(epochIndex);

                InfrastructureSnapshot infra =
                        infrastructure.snapshot(epoch);

                List<VM> vms =
                        DecisionEpochConverter.toVMs(
                                epoch,
                                infra.vmSpecs(),
                                cpuScale(cfg.get("cpu.scale"))
                        );

                EpochState state =
                        new EpochState(
                                epoch.timestamp(),
                                epoch.riskStates(),
                                vms,
                                infra.pms(),
                                previous
                        );

                PlacementSolution best;
int bestIteration;
double bestFitness;
CloudSimEvaluationResult bestEval;

switch (algorithm) {
    case "WA_AHO" -> {
        AdaptiveHippopotamusOptimization optimizer =
                new AdaptiveHippopotamusOptimization(
                        state,
                        populationSize,
                        maxIterations,
                        seedBase + epochIndex,
                        cfg.bool("aho.verbose"),
                        bridge,
                        fitness,
                        signal,
                        cfg.decimal("adaptive.mutation_base"),
                        cfg.decimal("adaptive.mutation_range")
                );

        optimizer.setEpochIndex(epochIndex + 1);
        best = optimizer.optimize();
        bestIteration = optimizer.getBestIteration();
        bestFitness = optimizer.getBestFitness();
        bestEval = optimizer.getBestEvaluation();
    }

    case "STANDARD_HO" -> {
        StandardHippopotamusOptimization optimizer =
                new StandardHippopotamusOptimization(
                        state,
                        populationSize,
                        maxIterations,
                        seedBase + epochIndex,
                        cfg.bool("aho.verbose"),
                        bridge,
                        fitness,
                        cfg.decimalOrDefault("standard_ho.exploration_probability", 0.50),
                        cfg.decimalOrDefault("standard_ho.mutation_probability", 0.15),
                        cfg.decimalOrDefault("standard_ho.attraction_probability", 0.80),
                        cfg.decimalOrDefault("standard_ho.defense_fraction", 0.10)
                );

        optimizer.setEpochIndex(epochIndex + 1);
        best = optimizer.optimize();
        bestIteration = optimizer.getBestIteration();
        bestFitness = optimizer.getBestFitness();
        bestEval = optimizer.getBestEvaluation();
    }

    case "FIRST_FIT" -> {
        FirstFitAllocation optimizer =
                new FirstFitAllocation(state, bridge, fitness);

        optimizer.setEpochIndex(epochIndex + 1);
        best = optimizer.optimize();
        bestIteration = optimizer.getBestIteration();
        bestFitness = optimizer.getBestFitness();
        bestEval = optimizer.getBestEvaluation();
    }

    default -> throw new IllegalArgumentException(
            "Unsupported algorithm: " + algorithm
    );
}

                if (!PlacementValidator.isFeasible(
                        vms,
                        infra.pms(),
                        best
                )) {
                    throw new IllegalStateException(
                            algorithm + " returned an infeasible committed placement at timestamp "
                                     + epoch.timestamp()
                    );
                }

                finalWriter.write(
                        epochIndex + 1,
                        state,
                        best,
                        bestIteration
                );

                java.util.Map<String, Integer> finalPlacementMap = new java.util.LinkedHashMap<>();
                for (int i = 0; i < vms.size(); i++) {
                    finalPlacementMap.put(vms.get(i).getId(), best.getPmIndex(i));
                }
                evaluator.commitFinalPlacement(epochIndex + 1, epoch.timestamp(), finalPlacementMap);

                if (bestEval != null) {
                    epochMetricsWriter.write(
                            epochIndex + 1,
                            epoch.timestamp(),
                            bestEval,
                            bestFitness,
                            true
                    );
                }

                previous =
                        PlacementState.fromSolution(
                                epoch.timestamp(),
                                vms,
                                infra.pms(),
                                best
                        );

                System.out.printf(
                        "epoch=%d timestamp=%d activeVMs=%d "
                                + "bestIteration=%d bestFitness=%s%n",
                        epochIndex + 1,
                        epoch.timestamp(),
                        vms.size(),
                        bestIteration,
                        Double.toString(bestFitness)
                );
            }

            finalWriter.flush();
        }

        System.out.println(
                "Completed all decision timestamps supplied by Person 1."
        );
    }

    private static double cpuScale(String mode) {

        return switch (mode.toLowerCase()) {

            case "fraction" -> 100.0;

            case "percent" -> 1.0;

            case "auto" -> -1.0;

            default -> throw new IllegalArgumentException(
                    "cpu.scale must be auto, fraction, or percent"
            );
        };
    }

    private static java.io.File resolveResultsFile(String relativePath) {
        if (relativePath == null || relativePath.isBlank()) {
            relativePath = "epoch_metrics.csv";
        }
        java.io.File direct = new java.io.File(relativePath);
        if (direct.isAbsolute()) {
            return direct;
        }
        String cleanName = relativePath.startsWith("results/") || relativePath.startsWith("results\\")
                ? relativePath.substring(8)
                : relativePath;
        java.io.File inResults = new java.io.File("results", cleanName);
        if (new java.io.File("results").isDirectory()) {
            return inResults;
        }
        java.io.File inParentResults = new java.io.File("../results", cleanName);
        if (new java.io.File("../results").isDirectory()) {
            return inParentResults;
        }
        return inResults;
    }
}