package com.vmplacement.optimization;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class AdaptiveHOStressExperiment {

    public static void main(String[] args) {

        System.out.println();
        System.out.println("================================================");
        System.out.println("       ADAPTIVE HO STRESS EXPERIMENT");
        System.out.println("================================================");

        runScenario("STABLE");
        runScenario("MODERATE");
        runScenario("HIGH");

        System.out.println();
        System.out.println("================================================");
        System.out.println("          STRESS EXPERIMENT COMPLETE");
        System.out.println("================================================");
    }

    private static void runScenario(String scenario) {

        System.out.println();
        System.out.println();
        System.out.println("################################################");
        System.out.println("SCENARIO: " + scenario);
        System.out.println("################################################");

        List<VM> vms = createVMs(scenario);

        List<PM> pms = Arrays.asList(
                new PM("PM1", 100, 16384),
                new PM("PM2", 100, 16384),
                new PM("PM3", 100, 16384),
                new PM("PM4", 100, 16384),
                new PM("PM5", 100, 16384),
                new PM("PM6", 100, 16384),
                new PM("PM7", 100, 16384),
                new PM("PM8", 100, 16384),
                new PM("PM9", 100, 16384),
                new PM("PM10", 100, 16384)
        );

        System.out.println();
        System.out.println("------------------------------------------------");
        System.out.println("WORKLOAD CHARACTERISTICS");
        System.out.println("------------------------------------------------");

        double variation =
                WorkloadVariation.calculateAverageNormalized(vms);

        double risk =
                PredictionRisk.calculateAverage(vms);

        System.out.println(
                "Average Workload Variation : "
                + (variation * 100) + "%"
        );

        System.out.println(
                "Average Prediction Risk    : "
                + risk
        );

        System.out.println(
                "Variation Level             : "
                + WorkloadVariation.classify(variation * 100)
        );

        System.out.println(
                "Risk Level                  : "
                + PredictionRisk.classify(risk)
        );

        // ------------------------------------------------
        // STANDARD HO
        // ------------------------------------------------

        System.out.println();
        System.out.println("================================================");
        System.out.println("             STANDARD HO");
        System.out.println("================================================");

        HippopotamusOptimization standardOptimizer =
                new HippopotamusOptimization(
                        vms,
                        pms,
                        20,
                        50
                );

        PlacementSolution standardBest =
                standardOptimizer.optimize();

        double standardFitness =
                FitnessFunction.calculate(
                        vms,
                        pms,
                        standardBest
                );

        System.out.println();
        System.out.println("-----------------------------------------------");
        System.out.println("STANDARD HO RESULT");
        System.out.println("-----------------------------------------------");

        System.out.println(
                "Best Placement : "
                + standardBest
        );

        System.out.println(
                "Fitness        : "
                + standardFitness
        );

        if (PlacementValidator.isFeasible(
                vms,
                pms,
                standardBest)
                &&
                PredictedPlacementValidator.isPredictedFeasible(
                        vms,
                        pms,
                        standardBest)) {

            FitnessFunction.printBreakdown(
                    vms,
                    pms,
                    standardBest
            );

        } else {

            System.out.println(
                    "Placement is INFEASIBLE."
            );
        }

        // ------------------------------------------------
        // ADAPTIVE HO
        // ------------------------------------------------

        System.out.println();
        System.out.println("================================================");
        System.out.println("             ADAPTIVE HO");
        System.out.println("================================================");

        AdaptiveHippopotamusOptimization adaptiveOptimizer =
                new AdaptiveHippopotamusOptimization(
                        vms,
                        pms,
                        20,
                        50
                );

        PlacementSolution adaptiveBest =
                adaptiveOptimizer.optimize();

        double adaptiveFitness =
                FitnessFunction.calculate(
                        vms,
                        pms,
                        adaptiveBest
                );

        System.out.println();
        System.out.println("-----------------------------------------------");
        System.out.println("ADAPTIVE HO RESULT");
        System.out.println("-----------------------------------------------");

        System.out.println(
                "Best Placement : "
                + adaptiveBest
        );

        System.out.println(
                "Fitness        : "
                + adaptiveFitness
        );

        if (PlacementValidator.isFeasible(
                vms,
                pms,
                adaptiveBest)
                &&
                PredictedPlacementValidator.isPredictedFeasible(
                        vms,
                        pms,
                        adaptiveBest)) {

            FitnessFunction.printBreakdown(
                    vms,
                    pms,
                    adaptiveBest
            );

        } else {

            System.out.println(
                    "Placement is INFEASIBLE."
            );
        }

        // ------------------------------------------------
        // ADAPTIVE CONTROLLER
        // ------------------------------------------------

        System.out.println();
        System.out.println("-----------------------------------------------");
        System.out.println("ADAPTIVE CONTROLLER");
        System.out.println("-----------------------------------------------");

        System.out.println(
                "Adaptive Signal          : "
                + adaptiveOptimizer.getAdaptiveSignal()
        );

        System.out.println(
                "Adaptive Mode            : "
                + adaptiveOptimizer.getCurrentAdaptiveMode()
        );

        System.out.println(
                "Exploration Probability  : "
                + adaptiveOptimizer.getCurrentExplorationProbability()
        );

        // ------------------------------------------------
        // COMPARISON
        // ------------------------------------------------

        System.out.println();
        System.out.println("================================================");
        System.out.println("        STANDARD HO vs ADAPTIVE HO");
        System.out.println("================================================");

        System.out.println(
                "Standard HO Fitness : "
                + standardFitness
        );

        System.out.println(
                "Adaptive HO Fitness : "
                + adaptiveFitness
        );

        if (standardFitness != Double.MAX_VALUE
                && adaptiveFitness != Double.MAX_VALUE) {

            double improvement =
                    ((standardFitness - adaptiveFitness)
                    / standardFitness) * 100.0;

            System.out.println(
                    "Fitness Improvement : "
                    + improvement + "%"
            );

            if (adaptiveFitness < standardFitness) {

                System.out.println(
                        "Result              : Adaptive HO performed better."
                );

            } else if (adaptiveFitness > standardFitness) {

                System.out.println(
                        "Result              : Standard HO performed better."
                );

            } else {

                System.out.println(
                        "Result              : Both algorithms performed equally."
                );
            }

        } else {

            System.out.println(
                    "Fitness comparison unavailable because one or both solutions are infeasible."
            );
        }
    }

    // ====================================================
    // CREATE SCENARIO DATA
    // ====================================================

    private static List<VM> createVMs(String scenario) {

    List<VM> vms = new ArrayList<>();

    // Current CPU values remain the same for all scenarios.
    double[] currentCpu = {
        20, 30, 15, 40, 25,
        35, 10, 45, 20, 30,
        15, 40, 25, 20, 35,
        10, 45, 30, 20, 25
    };

    double[] predictedCpu;

    if (scenario.equals("STABLE")) {

        predictedCpu = new double[] {
            22, 32, 17, 42, 27,
            37, 12, 47, 22, 32,
            17, 42, 27, 22, 37,
            12, 47, 32, 22, 27
        };

    } else if (scenario.equals("MODERATE")) {

        predictedCpu = new double[] {
            35, 45, 30, 55, 40,
            50, 25, 60, 35, 45,
            30, 55, 40, 35, 50,
            25, 60, 45, 35, 40
        };

    } else { // HIGH

        predictedCpu = new double[] {
            55, 0, 50, 10, 60,
            5, 45, 15, 55, 0,
            50, 10, 60, 0, 70,
            0, 80, 0, 55, 0
        };
    }

    double[] ram = {
        4096, 4096, 2048, 8192, 4096,
        4096, 2048, 8192, 4096, 4096,
        2048, 8192, 4096, 4096, 4096,
        2048, 8192, 4096, 4096, 4096
    };

    for (int i = 0; i < currentCpu.length; i++) {

        vms.add(
            new VM(
                String.format("VM_%03d", i + 1),
                currentCpu[i],
                predictedCpu[i],
                ram[i]
            )
        );
    }

    return vms;
}
}