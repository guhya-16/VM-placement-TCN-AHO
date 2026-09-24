package com.vmplacement.optimization;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class LargeOptimizationExperiment {

    private static final int POPULATION_SIZE = 20;
    private static final int MAX_ITERATIONS = 50;

    public static void main(String[] args) {

        System.out.println();
        System.out.println("================================================");
        System.out.println("       LARGE VM PLACEMENT EXPERIMENT");
        System.out.println("================================================");

        // ========================================================
        // VMs
        // ========================================================

        List<VM> vms = new ArrayList<>();

        vms.add(new VM("VM_001", 20, 25, 4096));
        vms.add(new VM("VM_002", 30, 40, 4096));
        vms.add(new VM("VM_003", 15, 20, 2048));
        vms.add(new VM("VM_004", 40, 55, 8192));
        vms.add(new VM("VM_005", 25, 30, 4096));
        vms.add(new VM("VM_006", 35, 45, 4096));
        vms.add(new VM("VM_007", 10, 15, 2048));
        vms.add(new VM("VM_008", 45, 60, 8192));
        vms.add(new VM("VM_009", 20, 35, 4096));
        vms.add(new VM("VM_010", 30, 35, 4096));
        vms.add(new VM("VM_011", 15, 25, 2048));
        vms.add(new VM("VM_012", 40, 50, 8192));
        vms.add(new VM("VM_013", 25, 40, 4096));
        vms.add(new VM("VM_014", 20, 25, 4096));
        vms.add(new VM("VM_015", 35, 50, 4096));
        vms.add(new VM("VM_016", 10, 20, 2048));
        vms.add(new VM("VM_017", 45, 55, 8192));
        vms.add(new VM("VM_018", 30, 45, 4096));
        vms.add(new VM("VM_019", 20, 30, 4096));
        vms.add(new VM("VM_020", 25, 35, 4096));

        // ========================================================
        // PMs
        // ========================================================

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

        // ========================================================
        // EXPERIMENT INFORMATION
        // ========================================================

        System.out.println();

        System.out.println("Number of VMs : " + vms.size());
        System.out.println("Number of PMs : " + pms.size());
        System.out.println("Population    : " + POPULATION_SIZE);
        System.out.println("Iterations    : " + MAX_ITERATIONS);

        // ========================================================
        // WORKLOAD CHARACTERISTICS
        // ========================================================

        double averageVariation =
                WorkloadVariation.calculateAverage(vms);

        double normalizedVariation =
                WorkloadVariation.calculateAverageNormalized(vms);

        double averageRisk =
                PredictionRisk.calculateAverage(vms);

        System.out.println();

        System.out.println("------------------------------------------------");
        System.out.println("WORKLOAD CHARACTERISTICS");
        System.out.println("------------------------------------------------");

        System.out.println(
                "Average Workload Variation : "
                        + averageVariation
                        + "%"
        );

        System.out.println(
                "Normalized Variation       : "
                        + normalizedVariation
        );

        System.out.println(
                "Average Prediction Risk    : "
                        + averageRisk
        );

        System.out.println(
                "Variation Level            : "
                        + WorkloadVariation.classify(
                                averageVariation
                        )
        );

        System.out.println(
                "Risk Level                 : "
                        + PredictionRisk.classify(
                                averageRisk
                        )
        );

        // ========================================================
        // STANDARD HO
        // ========================================================

        System.out.println();

        System.out.println("================================================");
        System.out.println("             STANDARD HO");
        System.out.println("================================================");

        HippopotamusOptimization standard =
                new HippopotamusOptimization(
                        vms,
                        pms,
                        POPULATION_SIZE,
                        MAX_ITERATIONS,
                        true
                );

        PlacementSolution standardSolution =
                standard.optimize();

        double standardFitness =
                FitnessFunction.calculate(
                        vms,
                        pms,
                        standardSolution
                );

        System.out.println();

        System.out.println("------------------------------------------------");
        System.out.println("STANDARD HO RESULT");
        System.out.println("------------------------------------------------");

        System.out.println(
                "Best Placement : "
                        + standardSolution
        );

        System.out.println(
                "Fitness        : "
                        + standardFitness
        );

        FitnessFunction.printBreakdown(
                vms,
                pms,
                standardSolution
        );

        // ========================================================
        // ADAPTIVE HO
        // ========================================================

        System.out.println();

        System.out.println("================================================");
        System.out.println("             ADAPTIVE HO");
        System.out.println("================================================");

        /*
         * We explicitly provide an empty RiskState list here.
         * Adaptive HO will fall back to VM-based workload
         * variation and prediction-risk calculations.
         */

        AdaptiveHippopotamusOptimization adaptive =
                new AdaptiveHippopotamusOptimization(
                        vms,
                        pms,
                        POPULATION_SIZE,
                        MAX_ITERATIONS,
                        new ArrayList<RiskState>(),
                        false
                );

        PlacementSolution adaptiveSolution =
                adaptive.optimize();

        double adaptiveFitness =
                FitnessFunction.calculate(
                        vms,
                        pms,
                        adaptiveSolution
                );

        System.out.println();

        System.out.println("------------------------------------------------");
        System.out.println("ADAPTIVE HO RESULT");
        System.out.println("------------------------------------------------");

        System.out.println(
                "Best Placement : "
                        + adaptiveSolution
        );

        System.out.println(
                "Fitness        : "
                        + adaptiveFitness
        );

        FitnessFunction.printBreakdown(
                vms,
                pms,
                adaptiveSolution
        );

        // ========================================================
        // ADAPTIVE CONTROLLER
        // ========================================================

        System.out.println();

        System.out.println("------------------------------------------------");
        System.out.println("ADAPTIVE CONTROLLER");
        System.out.println("------------------------------------------------");

        System.out.println(
                "Workload Variation       : "
                        + adaptive.getWorkloadVariation()
        );

        System.out.println(
                "Prediction Risk          : "
                        + adaptive.getPredictionRisk()
        );

        System.out.println(
                "Adaptive Signal          : "
                        + adaptive.getAdaptiveSignal()
        );

        System.out.println(
                "Adaptive Mode            : "
                        + adaptive.getCurrentAdaptiveMode()
        );

        System.out.println(
                "Exploration Probability  : "
                        + adaptive.getCurrentExplorationProbability()
        );

        // ========================================================
        // COMPARISON
        // ========================================================

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

        double improvement = 0.0;

        if (standardFitness != Double.MAX_VALUE
                && adaptiveFitness != Double.MAX_VALUE
                && standardFitness != 0.0) {

            improvement =
                    (
                            (standardFitness - adaptiveFitness)
                                    / standardFitness
                    ) * 100.0;
        }

        System.out.println(
                "Fitness Improvement : "
                        + improvement
                        + "%"
        );

        if (adaptiveFitness < standardFitness) {

            System.out.println(
                    "Result              : "
                            + "Adaptive HO performed better."
            );

        } else if (adaptiveFitness > standardFitness) {

            System.out.println(
                    "Result              : "
                            + "Standard HO performed better."
            );

        } else {

            System.out.println(
                    "Result              : "
                            + "Both produced the same fitness."
            );
        }

        // ========================================================
        // COMPLETE
        // ========================================================

        System.out.println();

        System.out.println("================================================");
        System.out.println("             EXPERIMENT COMPLETE");
        System.out.println("================================================");
    }
}

