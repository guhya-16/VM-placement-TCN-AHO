package com.vmplacement.optimization;

import java.util.List;

public class PredictionRisk {

    /**
     * Calculates a prototype risk score for one VM.
     *
     * Risk is based on:
     * 1. Difference between current and predicted CPU
     * 2. Whether predicted CPU approaches/exceeds 100%
     */
    public static double calculate(
            double currentCpu,
            double predictedCpu) {

        // Workload change
        double variation =
                Math.abs(predictedCpu - currentCpu) / 100.0;

        // Predicted overload component
        double overloadRisk = 0.0;

        if (predictedCpu > 80.0) {
            overloadRisk =
                    (predictedCpu - 80.0) / 20.0;
        }

        // Combine the two components
        double risk =
                0.6 * variation
                + 0.4 * overloadRisk;

        // Keep risk between 0 and 1
        return Math.min(
                1.0,
                Math.max(0.0, risk)
        );
    }


    /**
     * Calculates average prediction risk
     * across all VMs.
     */
    public static double calculateAverage(
            List<VM> vms) {

        if (vms == null || vms.isEmpty()) {
            return 0.0;
        }

        double totalRisk = 0.0;

        for (VM vm : vms) {

            totalRisk +=
                    calculate(
                            vm.getCpuDemand(),
                            vm.getPredictedCpuDemand()
                    );
        }

        return totalRisk / vms.size();
    }


    /**
     * Converts numerical risk into a
     * simple risk category.
     */
    public static String classify(
            double risk) {

        if (risk < 0.15) {
            return "LOW";
        }

        if (risk < 0.35) {
            return "MEDIUM";
        }

        return "HIGH";
    }
}