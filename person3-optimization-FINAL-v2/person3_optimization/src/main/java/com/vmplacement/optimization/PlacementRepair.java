package com.vmplacement.optimization;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class PlacementRepair {

    private PlacementRepair() {
    }

    /*
     * Creates a new feasible placement.
     *
     * Current resource constraints are HARD constraints.
     *
     * Predicted workload is NOT treated as a hard constraint.
     * It is handled by FitnessFunction through SLA/risk penalty.
     */
    public static PlacementSolution createRandomFeasibleSolution(
            List<VM> vms,
            List<PM> pms,
            Random random,
            double consolidationBias) {

        if (vms == null
                || pms == null
                || vms.isEmpty()
                || pms.isEmpty()) {

            return null;
        }

        int maxAttempts = 200;

        for (int attempt = 0;
             attempt < maxAttempts;
             attempt++) {

            PlacementSolution solution =
                    tryCreateSolution(
                            vms,
                            pms,
                            random,
                            consolidationBias
                    );

            if (solution != null
                    && PlacementValidator.isFeasible(
                    vms,
                    pms,
                    solution)) {

                return solution;
            }
        }

        return null;
    }

    /*
     * Creates one randomized placement.
     */
    private static PlacementSolution tryCreateSolution(
            List<VM> vms,
            List<PM> pms,
            Random random,
            double consolidationBias) {

        int[] placement =
                new int[vms.size()];

        for (int i = 0;
             i < placement.length;
             i++) {

            placement[i] = -1;
        }

        double[] currentMips =
                new double[pms.size()];

        double[] ram =
                new double[pms.size()];

        double[] storage =
                new double[pms.size()];

        int[] usedPes =
                new int[pms.size()];

        /*
         * Random VM order.
         */
        List<Integer> vmOrder =
                new ArrayList<>();

        for (int i = 0;
             i < vms.size();
             i++) {

            vmOrder.add(i);
        }

        Collections.shuffle(
                vmOrder,
                random
        );

        /*
         * Sometimes process larger VMs first.
         * This improves feasibility.
         */
        if (random.nextDouble() < 0.65) {

            vmOrder.sort(
                    (a, b) -> {

                        double workloadA =
                                vms.get(a)
                                        .getCurrentCpuUtilization()
                                        * vms.get(a).getMips();

                        double workloadB =
                                vms.get(b)
                                        .getCurrentCpuUtilization()
                                        * vms.get(b).getMips();

                        return Double.compare(
                                workloadB,
                                workloadA
                        );
                    }
            );
        }

        /*
         * Place VMs one by one.
         */
        for (int vmIndex : vmOrder) {

            VM vm =
                    vms.get(vmIndex);

            double currentRequired =
                    vm.getMips()
                            * vm.getCurrentCpuUtilization()
                            / 100.0;

            /*
             * Only CURRENT CPU is a hard constraint.
             */
            List<Integer> feasiblePms =
                    new ArrayList<>();

            for (int pmIndex = 0;
                 pmIndex < pms.size();
                 pmIndex++) {

                PM pm =
                        pms.get(pmIndex);

                /*
                 * PE constraint.
                 */
                if (usedPes[pmIndex]
                        + vm.getPes()
                        > pm.getPes()) {

                    continue;
                }

                /*
                 * Current CPU constraint.
                 */
                if (currentMips[pmIndex]
                        + currentRequired
                        > pm.getTotalMips()
                        + 1e-9) {

                    continue;
                }

                /*
                 * RAM constraint.
                 */
                if (ram[pmIndex]
                        + vm.getRamMb()
                        > pm.getRamMb()
                        + 1e-9) {

                    continue;
                }

                /*
                 * Storage constraint.
                 */
                if (storage[pmIndex]
                        + vm.getStorageMb()
                        > pm.getStorageMb()
                        + 1e-9) {

                    continue;
                }

                feasiblePms.add(
                        pmIndex
                );
            }

            if (feasiblePms.isEmpty()) {
                return null;
            }

            int selectedPm =
                    selectPM(
                            feasiblePms,
                            pms,
                            currentMips,
                            vm,
                            random,
                            consolidationBias
                    );

            placement[vmIndex] =
                    selectedPm;

            usedPes[selectedPm] +=
                    vm.getPes();

            currentMips[selectedPm] +=
                    currentRequired;

            ram[selectedPm] +=
                    vm.getRamMb();

            storage[selectedPm] +=
                    vm.getStorageMb();
        }

        return new PlacementSolution(
                placement
        );
    }

    /*
     * Select PM using a mixture of:
     *
     * 1. Random placement
     * 2. Consolidation
     * 3. Load balancing
     *
     * The random component is important because the
     * optimizer needs a diverse initial population.
     */
    private static int selectPM(
            List<Integer> feasiblePms,
            List<PM> pms,
            double[] currentMips,
            VM vm,
            Random random,
            double consolidationBias) {

        /*
         * Random choice.
         *
         * Higher randomness = more population diversity.
         */
        double randomChoiceProbability =
                0.35
                        + 0.35
                        * (1.0 - consolidationBias);

        if (random.nextDouble()
                < randomChoiceProbability) {

            return feasiblePms.get(
                    random.nextInt(
                            feasiblePms.size()
                    )
            );
        }

        double bestScore =
                Double.MAX_VALUE;

        List<Integer> bestPms =
                new ArrayList<>();

        for (int pmIndex :
                feasiblePms) {

            PM pm =
                    pms.get(pmIndex);

            double currentRequired =
                    vm.getMips()
                            * vm.getCurrentCpuUtilization()
                            / 100.0;

            double resultingUtilization =
                    (
                            currentMips[pmIndex]
                                    + currentRequired
                    )
                            /
                            pm.getTotalMips();

            boolean alreadyActive =
                    currentMips[pmIndex] > 0.0;

            /*
             * Lower score is better.
             */
            double score =
                    resultingUtilization;

            /*
             * Consolidation bonus.
             */
            if (alreadyActive) {

                score -=
                        consolidationBias
                                * 0.20;
            }

            /*
             * Small random noise prevents
             * deterministic identical solutions.
             */
            score +=
                    random.nextDouble()
                            * 0.05;

            if (score < bestScore) {

                bestScore =
                        score;

                bestPms.clear();

                bestPms.add(
                        pmIndex
                );

            } else if (
                    Math.abs(
                            score
                                    - bestScore
                    ) < 0.02) {

                bestPms.add(
                        pmIndex
                );
            }
        }

        if (bestPms.isEmpty()) {

            return feasiblePms.get(
                    random.nextInt(
                            feasiblePms.size()
                    )
            );
        }

        return bestPms.get(
                random.nextInt(
                        bestPms.size()
                )
        );
    }

    /*
     * Generic repair.
     *
     * Only current feasibility is hard.
     */
    public static PlacementSolution repair(
            List<VM> vms,
            List<PM> pms,
            PlacementSolution original,
            Random random) {

        if (original != null
                && PlacementValidator.isFeasible(
                vms,
                pms,
                original)) {

            return original;
        }

        /*
         * Construct a new current-feasible placement.
         */
        return createRandomFeasibleSolution(
                vms,
                pms,
                random,
                0.65
        );
    }
}