package com.vmplacement.optimization;

import java.util.List;

public class PredictedPlacementValidator {

    public static boolean isPredictedFeasible(
            List<VM> vms,
            List<PM> pms,
            PlacementSolution solution) {

        int[] placement =
                solution.getVmToPm();

        if (placement == null) {
            return false;
        }

        if (placement.length != vms.size()) {
            return false;
        }

        double[] predictedMips =
                new double[pms.size()];

        double[] usedRam =
                new double[pms.size()];

        double[] usedStorage =
                new double[pms.size()];

        int[] usedPes =
                new int[pms.size()];

        for (int i = 0;
             i < vms.size();
             i++) {

            int pmIndex =
                    placement[i];

            if (pmIndex < 0
                    || pmIndex >= pms.size()) {

                return false;
            }

            VM vm =
                    vms.get(i);

            double vmPredictedMips =
                    vm.getMips()
                            *
                            vm.getPredictedCpuUtilization()
                            / 100.0;

            predictedMips[pmIndex] +=
                    vmPredictedMips;

            usedRam[pmIndex] +=
                    vm.getRamMb();

            usedStorage[pmIndex] +=
                    vm.getStorageMb();

            usedPes[pmIndex] +=
                    vm.getPes();
        }

        for (int i = 0;
             i < pms.size();
             i++) {

            PM pm =
                    pms.get(i);

            if (usedPes[i] >
                    pm.getPes()) {

                return false;
            }

            if (predictedMips[i] >
                    pm.getTotalMips()
                            + 1e-9) {

                return false;
            }

            if (usedRam[i] >
                    pm.getRamMb()
                            + 1e-9) {

                return false;
            }

            if (usedStorage[i] >
                    pm.getStorageMb()
                            + 1e-9) {

                return false;
            }
        }

        return true;
    }
}