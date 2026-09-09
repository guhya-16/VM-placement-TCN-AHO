package com.vmplacement.optimization;

import java.util.ArrayList;
import java.util.List;

public class DecisionEpochConverter {

    /*
     * VM resource templates matching Person 2's CloudSim environment.
     *
     * Type 1: 1 PE, 500 MIPS, 512 MB RAM, 40 GB storage
     * Type 2: 2 PE, 1000 MIPS, 1024 MB RAM, 60 GB storage
     * Type 3: 3 PE, 1500 MIPS, 2048 MB RAM, 80 GB storage
     * Type 4: 4 PE, 2000 MIPS, 3072 MB RAM, 100 GB storage
     */

    private static final int[] VM_PES = {
            1, 2, 3, 4
    };

    private static final double[] VM_MIPS = {
            500.0, 1000.0, 1500.0, 2000.0
    };

    private static final double[] VM_RAM_MB = {
            512.0, 1024.0, 2048.0, 3072.0
    };

    private static final double[] VM_STORAGE_MB = {
            40960.0, 61440.0, 81920.0, 102400.0
    };

    public static List<VM> toVMs(DecisionEpoch epoch) {

        List<VM> vms = new ArrayList<>();

        List<RiskState> riskStates = epoch.getRiskStates();

        for (int i = 0; i < riskStates.size(); i++) {

            RiskState risk = riskStates.get(i);

            /*
             * VM types repeat in the same order as Person 2's
             * VM manifest.
             */
            int typeIndex = i % 4;

            int pes = VM_PES[typeIndex];
            double mips = VM_MIPS[typeIndex];
            double ramMb = VM_RAM_MB[typeIndex];
            double storageMb = VM_STORAGE_MB[typeIndex];

            /*
             * Workload information comes from Person 1's
             * RiskState output.
             */
            // Risk-state CSV stores CPU utilization as a fraction (0.0 to 1.0).
            // The optimizer represents utilization internally as a percentage.
            double currentCpu = toPercent(risk.getCurrentCpu());
            double predictedCpu = toPercent(risk.getPredictedMeanCpu());
            double predictedMean = toPercent(risk.getPredictedMeanCpu());
            double predictedPeak = toPercent(risk.getPredictedPeakCpu());
            double predictedStd = toPercent(risk.getPredictedStdCpu());

            VM vm = new VM(
                    risk.getVmId(),
                    pes,
                    mips,
                    ramMb,
                    storageMb,
                    currentCpu,
                    predictedCpu,
                    predictedMean,
                    predictedPeak,
                    predictedStd
            );

            vms.add(vm);
        }

        return vms;
    }

    private static double toPercent(double fraction) {
        // Be defensive if a future CSV is already percentage-based.
        if (fraction >= 0.0 && fraction <= 1.0) {
            return fraction * 100.0;
        }
        return fraction;
    }
}