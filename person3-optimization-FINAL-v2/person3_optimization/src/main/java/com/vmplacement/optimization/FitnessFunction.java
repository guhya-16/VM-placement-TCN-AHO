package com.vmplacement.optimization;

import java.util.List;

/**
 * Objective function for VM-to-PM placement.
 *
 * Current resource capacity is a hard feasibility constraint.
 * Predicted workload is deliberately a SOFT optimization signal so that
 * the optimizer can compare otherwise-current-feasible placements and
 * prefer placements with lower future SLA risk and better load balance.
 *
 * The final energy evaluation is performed with CloudSim Plus. This class
 * uses a lightweight power approximation only for optimization experiments.
 */
public final class FitnessFunction {

    private static final double ENERGY_WEIGHT = 0.40;
    private static final double SLA_WEIGHT = 0.25;
    private static final double IMBALANCE_WEIGHT = 0.20;
    private static final double ACTIVE_PM_WEIGHT = 0.15;

    private static final double SLA_THRESHOLD = 0.80;
    private static final double PEAK_RISK_THRESHOLD = 0.90;

    private FitnessFunction() {
    }

    public static double calculate(
            List<VM> vms,
            List<PM> pms,
            PlacementSolution solution) {

        if (!PlacementValidator.isFeasible(vms, pms, solution)) {
            return Double.MAX_VALUE;
        }

        int[] placement = solution.getVmToPm();

        double[] currentMips = new double[pms.size()];
        double[] predictedMips = new double[pms.size()];
        double[] predictedPeakMips = new double[pms.size()];
        boolean[] active = new boolean[pms.size()];

        for (int i = 0; i < vms.size(); i++) {
            VM vm = vms.get(i);
            int pmIndex = placement[i];

            currentMips[pmIndex] +=
                    vm.getMips() * vm.getCurrentCpuUtilization() / 100.0;

            predictedMips[pmIndex] +=
                    vm.getMips() * vm.getPredictedMeanCpu() / 100.0;

            predictedPeakMips[pmIndex] +=
                    vm.getMips() * vm.getPredictedPeakCpu() / 100.0;

            active[pmIndex] = true;
        }

        double totalPower = 0.0;
        double totalMaxPower = 0.0;
        int activePmCount = 0;

        for (int i = 0; i < pms.size(); i++) {
            PM pm = pms.get(i);
            totalMaxPower += pm.getMaxPowerWatts();

            if (!active[i]) {
                continue;
            }

            activePmCount++;

            double utilization = clamp(
                    currentMips[i] / pm.getTotalMips(), 0.0, 1.0);

            totalPower += pm.getStaticPowerWatts()
                    + (pm.getMaxPowerWatts() - pm.getStaticPowerWatts())
                    * utilization;
        }

        /*
         * Predicted SLA risk:
         * - mean predicted utilization above 80%
         * - predicted peak above 90%
         *
         * These are soft penalties, not feasibility rejection.
         */
        double slaPenalty = 0.0;

        for (int i = 0; i < pms.size(); i++) {
            if (!active[i]) {
                continue;
            }

            double meanUtil =
                    predictedMips[i] / pms.get(i).getTotalMips();

            double peakUtil =
                    predictedPeakMips[i] / pms.get(i).getTotalMips();

            if (meanUtil > SLA_THRESHOLD) {
                double excess = meanUtil - SLA_THRESHOLD;
                slaPenalty += excess * excess;
            }

            if (peakUtil > PEAK_RISK_THRESHOLD) {
                double excess = peakUtil - PEAK_RISK_THRESHOLD;
                slaPenalty += 0.5 * excess * excess;
            }
        }

        /*
         * Predicted load imbalance among active PMs.
         */
        double averageUtilization = 0.0;
        int activeCount = 0;

        for (int i = 0; i < pms.size(); i++) {
            if (!active[i]) {
                continue;
            }

            averageUtilization +=
                    predictedMips[i] / pms.get(i).getTotalMips();
            activeCount++;
        }

        if (activeCount > 0) {
            averageUtilization /= activeCount;
        }

        double imbalance = 0.0;

        for (int i = 0; i < pms.size(); i++) {
            if (!active[i]) {
                continue;
            }

            double utilization =
                    predictedMips[i] / pms.get(i).getTotalMips();

            double difference = utilization - averageUtilization;
            imbalance += difference * difference;
        }

        if (activeCount > 0) {
            imbalance /= activeCount;
        }

        double normalizedEnergy =
                totalMaxPower > 0.0
                        ? totalPower / totalMaxPower
                        : 1.0;

        double normalizedSla = Math.min(1.0, slaPenalty);
        double normalizedActivePm =
                (double) activePmCount / pms.size();

        return ENERGY_WEIGHT * normalizedEnergy
                + SLA_WEIGHT * normalizedSla
                + IMBALANCE_WEIGHT * imbalance
                + ACTIVE_PM_WEIGHT * normalizedActivePm;
    }

    public static void printBreakdown(
            List<VM> vms,
            List<PM> pms,
            PlacementSolution solution) {

        if (!PlacementValidator.isFeasible(vms, pms, solution)) {
            System.out.println("Placement is not currently feasible.");
            return;
        }

        int[] placement = solution.getVmToPm();

        double[] currentMips = new double[pms.size()];
        double[] predictedMips = new double[pms.size()];
        double[] predictedPeakMips = new double[pms.size()];
        boolean[] active = new boolean[pms.size()];

        for (int i = 0; i < vms.size(); i++) {
            VM vm = vms.get(i);
            int pm = placement[i];

            currentMips[pm] +=
                    vm.getMips() * vm.getCurrentCpuUtilization() / 100.0;
            predictedMips[pm] +=
                    vm.getMips() * vm.getPredictedMeanCpu() / 100.0;
            predictedPeakMips[pm] +=
                    vm.getMips() * vm.getPredictedPeakCpu() / 100.0;

            active[pm] = true;
        }

        double totalPower = 0.0;
        double totalMaxPower = 0.0;
        int activePmCount = 0;

        double slaPenalty = 0.0;
        double averageUtilization = 0.0;
        int activeCount = 0;

        for (int i = 0; i < pms.size(); i++) {
            PM pm = pms.get(i);
            totalMaxPower += pm.getMaxPowerWatts();

            if (!active[i]) {
                continue;
            }

            activePmCount++;

            double currentUtil =
                    clamp(currentMips[i] / pm.getTotalMips(), 0.0, 1.0);

            totalPower += pm.getStaticPowerWatts()
                    + (pm.getMaxPowerWatts() - pm.getStaticPowerWatts())
                    * currentUtil;

            double meanUtil =
                    predictedMips[i] / pm.getTotalMips();
            double peakUtil =
                    predictedPeakMips[i] / pm.getTotalMips();

            if (meanUtil > SLA_THRESHOLD) {
                double excess = meanUtil - SLA_THRESHOLD;
                slaPenalty += excess * excess;
            }

            if (peakUtil > PEAK_RISK_THRESHOLD) {
                double excess = peakUtil - PEAK_RISK_THRESHOLD;
                slaPenalty += 0.5 * excess * excess;
            }

            averageUtilization += meanUtil;
            activeCount++;
        }

        if (activeCount > 0) {
            averageUtilization /= activeCount;
        }

        double imbalance = 0.0;

        for (int i = 0; i < pms.size(); i++) {
            if (!active[i]) {
                continue;
            }

            double utilization =
                    predictedMips[i] / pms.get(i).getTotalMips();

            double difference = utilization - averageUtilization;
            imbalance += difference * difference;
        }

        if (activeCount > 0) {
            imbalance /= activeCount;
        }

        double normalizedEnergy =
                totalMaxPower > 0.0
                        ? totalPower / totalMaxPower
                        : 1.0;

        double normalizedSla = Math.min(1.0, slaPenalty);
        double normalizedActivePm =
                (double) activePmCount / pms.size();

        double finalFitness =
                ENERGY_WEIGHT * normalizedEnergy
                        + SLA_WEIGHT * normalizedSla
                        + IMBALANCE_WEIGHT * imbalance
                        + ACTIVE_PM_WEIGHT * normalizedActivePm;

        System.out.println("Total Power (W)          : " + totalPower);
        System.out.println("Active PMs               : " + activePmCount);
        System.out.println("Average Current Util     : "
                + averageCurrentUtilization(currentMips, pms, active));
        System.out.println("Average Predicted Util   : " + averageUtilization);
        System.out.println("Predicted SLA Penalty    : " + slaPenalty);
        System.out.println("Load Imbalance           : " + imbalance);
        System.out.println("Normalized Energy        : " + normalizedEnergy);
        System.out.println("Normalized Active PM     : " + normalizedActivePm);
        System.out.println("Final Fitness             : " + finalFitness);
    }

    private static double averageCurrentUtilization(
            double[] currentMips,
            List<PM> pms,
            boolean[] active) {

        double total = 0.0;
        int count = 0;

        for (int i = 0; i < pms.size(); i++) {
            if (!active[i]) {
                continue;
            }

            total += currentMips[i] / pms.get(i).getTotalMips();
            count++;
        }

        return count == 0 ? 0.0 : total / count;
    }

    private static double clamp(
            double value,
            double min,
            double max) {

        return Math.max(min, Math.min(max, value));
    }
}
