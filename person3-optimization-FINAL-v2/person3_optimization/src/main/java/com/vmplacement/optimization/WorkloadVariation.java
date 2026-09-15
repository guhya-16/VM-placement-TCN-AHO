package com.vmplacement.optimization;

import java.util.List;

public class WorkloadVariation {

    // Calculate variation for one VM
    public static double calculate(
            double currentWorkload,
            double predictedWorkload) {

        return Math.abs(
                predictedWorkload - currentWorkload
        );
    }


    // Calculate normalized variation for one VM
    public static double calculateNormalized(
            double currentWorkload,
            double predictedWorkload) {

        return Math.abs(
                predictedWorkload - currentWorkload
        ) / 100.0;
    }


    // Calculate average workload variation
    // across all VMs
    public static double calculateAverage(
            List<VM> vms) {

        if (vms == null || vms.isEmpty()) {
            return 0.0;
        }

        double totalVariation = 0.0;

        for (VM vm : vms) {

            totalVariation +=
                    calculate(
                            vm.getCpuDemand(),
                            vm.getPredictedCpuDemand()
                    );
        }

        return totalVariation / vms.size();
    }


    // Calculate normalized average variation
    public static double calculateAverageNormalized(
            List<VM> vms) {

        return calculateAverage(vms) / 100.0;
    }


    // Classify workload variation
    public static String classify(
            double variation) {

        if (variation < 10.0) {
            return "LOW";
        }

        if (variation < 25.0) {
            return "MEDIUM";
        }

        return "HIGH";
    }
}