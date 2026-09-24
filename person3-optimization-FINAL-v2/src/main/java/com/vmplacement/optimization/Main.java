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
                new UnconfiguredInfrastructureProvider();

        CloudSimEvaluator evaluator =
                new UnconfiguredCloudSimEvaluator();

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
                    epochIndex < epochs.size();
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

                AdaptiveHippopotamusOptimization aho =
                        new AdaptiveHippopotamusOptimization(
                                state,
                                cfg.integer("aho.population_size"),
                                cfg.integer("aho.max_iterations"),
                                seedBase + epochIndex,
                                cfg.bool("aho.verbose"),
                                bridge,
                                fitness,
                                signal,
                                cfg.decimal("adaptive.mutation_base"),
                                cfg.decimal("adaptive.mutation_range")
                        );

                aho.setEpochIndex(epochIndex + 1);

                PlacementSolution best = aho.optimize();

                if (!PlacementValidator.isFeasible(
                        vms,
                        infra.pms(),
                        best
                )) {
                    throw new IllegalStateException(
                            "AHO returned an infeasible committed placement at timestamp "
                                    + epoch.timestamp()
                    );
                }

                finalWriter.write(
                        epochIndex + 1,
                        state,
                        best,
                        aho.getBestIteration()
                );

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
                        aho.getBestIteration(),
                        Double.toString(aho.getBestFitness())
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
}