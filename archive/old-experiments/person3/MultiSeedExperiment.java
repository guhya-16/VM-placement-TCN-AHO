package com.vmplacement.optimization;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class MultiSeedExperiment {

    private static final int POPULATION_SIZE = 20;

    private static final int MAX_ITERATIONS = 50;

    private static final long[] SEEDS = {
            101,
            201,
            301,
            401,
            501,
            601,
            701,
            801,
            901,
            1001
    };

    public static void main(String[] args) {

        List<VM> vms =
                createVMs();

        List<PM> pms =
                createPMs();

        printFeasibility(
                vms,
                pms
        );

        System.out.println();

        System.out.println(
                "============================================================"
        );

        System.out.println(
                " FAIR STANDARD HO vs ADAPTIVE HO EXPERIMENT"
        );

        System.out.println(
                " SAME INITIAL POPULATION FOR EVERY SEED"
        );

        System.out.println(
                "============================================================"
        );

        System.out.printf(
                "%-8s %-14s %-14s %-14s%n",
                "Seed",
                "Standard HO",
                "Adaptive HO",
                "Improvement"
        );

        System.out.println(
                "------------------------------------------------------------"
        );

        List<Double> standardResults =
                new ArrayList<>();

        List<Double> adaptiveResults =
                new ArrayList<>();

        List<Double> improvements =
                new ArrayList<>();

        int adaptiveWins = 0;
        int standardWins = 0;
        int ties = 0;

        for (long seed : SEEDS) {

            /*
             * Generate ONE population.
             */
            List<PlacementSolution>
                    commonPopulation =
                    createCommonInitialPopulation(
                            vms,
                            pms,
                            seed
                    );

            if (commonPopulation.isEmpty()) {

                System.out.println(
                        "Seed "
                                + seed
                                + " skipped: no feasible population."
                );

                continue;
            }

            /*
             * Standard HO receives a copy.
             */
            HippopotamusOptimization standard =
                    new HippopotamusOptimization(
                            vms,
                            pms,
                            POPULATION_SIZE,
                            MAX_ITERATIONS,
                            seed,
                            deepCopyPopulation(
                                    commonPopulation
                            ),
                            false
                    );

            standard.optimize();

            double standardFitness =
                    standard.getBestFitness();

            /*
             * Adaptive HO receives the SAME population.
             */
            AdaptiveHippopotamusOptimization adaptive =
                    new AdaptiveHippopotamusOptimization(
                            vms,
                            pms,
                            POPULATION_SIZE,
                            MAX_ITERATIONS,
                            seed,
                            new ArrayList<RiskState>(),
                            deepCopyPopulation(
                                    commonPopulation
                            ),
                            false
                    );

            adaptive.optimize();

            double adaptiveFitness =
                    adaptive.getBestFitness();

            /*
             * Lower fitness is better.
             */
            double improvement =
                    standardFitness != 0.0
                            ? (
                            (
                                    standardFitness
                                            - adaptiveFitness
                            )
                                    / standardFitness
                    ) * 100.0
                            : 0.0;

            standardResults.add(
                    standardFitness
            );

            adaptiveResults.add(
                    adaptiveFitness
            );

            improvements.add(
                    improvement
            );

            double tolerance =
                    1e-9;

            if (adaptiveFitness
                    < standardFitness
                    - tolerance) {

                adaptiveWins++;

            } else if (standardFitness
                    < adaptiveFitness
                    - tolerance) {

                standardWins++;

            } else {

                ties++;
            }

            System.out.printf(
                    "%-8d %-14.8f %-14.8f %+.4f%%%n",
                    seed,
                    standardFitness,
                    adaptiveFitness,
                    improvement
            );
        }

        System.out.println(
                "------------------------------------------------------------"
        );

        printSummary(
                standardResults,
                adaptiveResults,
                improvements,
                adaptiveWins,
                standardWins,
                ties
        );
    }

    // ============================================================
    // VM creation
    // ============================================================

    private static List<VM> createVMs() {

        List<VM> vms =
                new ArrayList<>();

        /*
         * Current CPU utilization (%)
         */
        int[] currentCpu = {
                20, 25, 30, 35, 40,
                45, 50, 55, 60, 65,
                70, 30, 35, 40, 45,
                50, 55, 60, 65, 70
        };

        /*
         * Predicted CPU utilization (%)
         */
        int[] predictedCpu = {
                30, 35, 40, 45, 50,
                55, 60, 65, 70, 75,
                80, 40, 45, 50, 55,
                60, 65, 70, 75, 80
        };

        /*
         * RAM in MB.
         */
        int[] ram = {
                4096, 4096, 4096, 4096, 4096,
                4096, 4096, 4096, 4096, 4096,
                4096, 4096, 4096, 4096, 4096,
                4096, 4096, 4096, 4096, 4096
        };

        for (int i = 0;
             i < currentCpu.length;
             i++) {

            vms.add(
                    new VM(
                            "VM_" + String.format(
                                    "%03d",
                                    i + 1
                            ),
                            currentCpu[i],
                            predictedCpu[i],
                            ram[i]
                    )
            );
        }

        return vms;
    }

    // ============================================================
    // PM creation
    // ============================================================

    private static List<PM> createPMs() {

        List<PM> pms =
                new ArrayList<>();

        for (int i = 0;
             i < 10;
             i++) {

            pms.add(
                    new PM(
                            "PM" + (i + 1),
                            130,
                            16384
                    )
            );
        }

        return pms;
    }

    // ============================================================
    // Common initial population
    // ============================================================

    private static List<PlacementSolution>
    createCommonInitialPopulation(
            List<VM> vms,
            List<PM> pms,
            long seed) {

        List<PlacementSolution>
                population =
                new ArrayList<>();

        /*
         * This Random object is used ONLY to generate
         * the common initial population.
         */
        Random sharedRandom =
                new Random(seed);

        int attempts = 0;

        int maxAttempts =
                POPULATION_SIZE * 200;

        while (population.size()
                < POPULATION_SIZE
                && attempts < maxAttempts) {

            attempts++;

            PlacementSolution solution =
                    PlacementRepair
                            .createRandomFeasibleSolution(
                                    vms,
                                    pms,
                                    sharedRandom,
                                    0.65
                            );

            if (solution == null) {
                continue;
            }

            if (!isFeasible(
                    vms,
                    pms,
                    solution
            )) {
                continue;
            }

            if (containsSolution(
                    population,
                    solution
            )) {
                continue;
            }

            population.add(
                    copySolution(solution)
            );
        }

        if (population.size()
                < POPULATION_SIZE) {

            System.out.printf(
                    "Warning: seed %d created %d/%d "
                            + "initial solutions.%n",
                    seed,
                    population.size(),
                    POPULATION_SIZE
            );
        }

        return population;
    }

    // ============================================================
    // Feasibility
    // ============================================================

    private static boolean isFeasible(
            List<VM> vms,
            List<PM> pms,
            PlacementSolution solution) {

        return PlacementValidator.isFeasible(
                vms,
                pms,
                solution
        );
    }

    // ============================================================
    // Copy
    // ============================================================

    private static PlacementSolution
    copySolution(
            PlacementSolution solution) {

        return new PlacementSolution(
                solution
                        .getVmToPm()
                        .clone()
        );
    }

    private static List<PlacementSolution>
    deepCopyPopulation(
            List<PlacementSolution> source) {

        List<PlacementSolution>
                copy =
                new ArrayList<>();

        for (PlacementSolution solution
                : source) {

            copy.add(
                    copySolution(solution)
            );
        }

        return copy;
    }

    // ============================================================
    // Duplicate check
    // ============================================================

    private static boolean containsSolution(
            List<PlacementSolution> population,
            PlacementSolution candidate) {

        for (PlacementSolution existing
                : population) {

            if (Arrays.equals(
                    existing.getVmToPm(),
                    candidate.getVmToPm()
            )) {

                return true;
            }
        }

        return false;
    }

    // ============================================================
    // Feasibility report
    // ============================================================

    private static void printFeasibility(
            List<VM> vms,
            List<PM> pms) {

        double totalCurrentCpu = 0.0;
        double totalPredictedCpu = 0.0;
        double totalVmRam = 0.0;

        double totalPmCpu = 0.0;
        double totalPmRam = 0.0;

        for (VM vm : vms) {

            totalCurrentCpu +=
                    vm.getMips()
                            * vm.getCurrentCpuUtilization()
                            / 100.0;

            totalPredictedCpu +=
                    vm.getMips()
                            * vm.getPredictedCpuUtilization()
                            / 100.0;

            totalVmRam +=
                    vm.getRamMb();
        }

        for (PM pm : pms) {

            totalPmCpu +=
                    pm.getTotalMips();

            totalPmRam +=
                    pm.getRamMb();
        }

        System.out.println();

        System.out.println(
                "Feasibility:"
        );

        System.out.printf(
                "Total Current CPU     : %.2f%n",
                totalCurrentCpu
        );

        System.out.printf(
                "Total Predicted CPU   : %.2f%n",
                totalPredictedCpu
        );

        System.out.printf(
                "Total PM CPU Capacity : %.2f%n",
                totalPmCpu
        );

        System.out.printf(
                "Total VM RAM          : %.2f MB%n",
                totalVmRam
        );

        System.out.printf(
                "Total PM RAM Capacity : %.2f MB%n",
                totalPmRam
        );

        boolean feasible =
                totalCurrentCpu <= totalPmCpu
                        && totalPredictedCpu <= totalPmCpu
                        && totalVmRam <= totalPmRam;

        System.out.println(
                "Feasibility Status      : "
                        + (
                        feasible
                                ? "FEASIBLE"
                                : "NOT FEASIBLE"
                )
        );
    }

    // ============================================================
    // Summary
    // ============================================================

    private static void printSummary(
            List<Double> standardResults,
            List<Double> adaptiveResults,
            List<Double> improvements,
            int adaptiveWins,
            int standardWins,
            int ties) {

        if (standardResults.isEmpty()) {

            System.out.println(
                    "No valid experiment results."
            );

            return;
        }

        double standardMean =
                mean(standardResults);

        double adaptiveMean =
                mean(adaptiveResults);

        double standardStd =
                standardDeviation(
                        standardResults,
                        standardMean
                );

        double adaptiveStd =
                standardDeviation(
                        adaptiveResults,
                        adaptiveMean
                );

        double meanImprovement =
                mean(improvements);

        double improvementStd =
                standardDeviation(
                        improvements,
                        meanImprovement
                );

        double overallImprovement =
                standardMean != 0.0
                        ? (
                        (
                                standardMean
                                        - adaptiveMean
                        )
                                / standardMean
                ) * 100.0
                        : 0.0;

        System.out.println();

        System.out.println(
                "============================================================"
        );

        System.out.println(
                " SUMMARY"
        );

        System.out.println(
                "============================================================"
        );

        System.out.printf(
                "Standard HO Mean Fitness     : %.8f%n",
                standardMean
        );

        System.out.printf(
                "Standard HO Std. Deviation   : %.8f%n",
                standardStd
        );

        System.out.printf(
                "Adaptive HO Mean Fitness     : %.8f%n",
                adaptiveMean
        );

        System.out.printf(
                "Adaptive HO Std. Deviation   : %.8f%n",
                adaptiveStd
        );

        System.out.printf(
                "Mean Fitness Improvement     : %.4f%%%n",
                meanImprovement
        );

        System.out.printf(
                "Improvement Std. Deviation   : %.4f%%%n",
                improvementStd
        );

        System.out.printf(
                "Adaptive HO Wins             : %d%n",
                adaptiveWins
        );

        System.out.printf(
                "Standard HO Wins             : %d%n",
                standardWins
        );

        System.out.printf(
                "Ties                         : %d%n",
                ties
        );

        System.out.printf(
                "Overall Mean Improvement     : %.4f%%%n",
                overallImprovement
        );

        System.out.println();

        if (adaptiveMean < standardMean) {

            System.out.printf(
                    "Conclusion: Adaptive HO achieved "
                            + "%.4f%% lower mean fitness "
                            + "than Standard HO.%n",
                    overallImprovement
            );

        } else if (standardMean < adaptiveMean) {

            System.out.printf(
                    "Conclusion: Standard HO achieved "
                            + "%.4f%% lower mean fitness "
                            + "than Adaptive HO.%n",
                    -overallImprovement
            );

        } else {

            System.out.println(
                    "Conclusion: Both algorithms achieved "
                            + "the same mean fitness."
            );
        }
    }

    // ============================================================
    // Statistics
    // ============================================================

    private static double mean(
            List<Double> values) {

        double sum = 0.0;

        for (double value : values) {
            sum += value;
        }

        return sum / values.size();
    }

    private static double standardDeviation(
            List<Double> values,
            double mean) {

        double sum = 0.0;

        for (double value : values) {

            double difference =
                    value - mean;

            sum +=
                    difference * difference;
        }

        return Math.sqrt(
                sum / values.size()
        );
    }
}