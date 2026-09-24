package com.vmplacement.optimization;

import java.util.List;

public class PredictedUtilization {

    public static double[] calculate(
            List<VM> vms,
            List<PM> pms,
            PlacementSolution solution) {

        double[] predictedCpu =
                new double[pms.size()];

        int[] vmToPm =
                solution.getVmToPm();

        for (int i = 0; i < vms.size(); i++) {

            int pmIndex = vmToPm[i];

            if (pmIndex < 0 ||
                pmIndex >= pms.size()) {
                continue;
            }

            predictedCpu[pmIndex] +=
                    vms.get(i).getPredictedCpuDemand();
        }

        double[] utilization =
                new double[pms.size()];

        for (int i = 0; i < pms.size(); i++) {

            utilization[i] =
                    predictedCpu[i]
                    / pms.get(i).getCpuCapacity();
        }

        return utilization;
    }

    public static void print(
            List<VM> vms,
            List<PM> pms,
            PlacementSolution solution) {

        double[] utilization =
                calculate(vms, pms, solution);

        System.out.println();
        System.out.println(
                "========== PREDICTED PM UTILIZATION =========="
        );

        for (int i = 0; i < pms.size(); i++) {

            System.out.println(
                    pms.get(i).getId()
                    + " | Predicted Utilization = "
                    + (utilization[i] * 100)
                    + "%"
            );
        }

        System.out.println(
                "==============================================="
        );
    }
}