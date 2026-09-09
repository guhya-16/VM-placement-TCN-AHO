package com.vmplacement.optimization;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.LinkedHashMap;
import java.util.Map;

public class CSVWorkloadReader {

    public static List<VM> readPredictions(
            String filePath) throws IOException {

        List<VM> vms = new ArrayList<>();

        try (BufferedReader br =
                     new BufferedReader(
                             new FileReader(filePath))) {

            // Skip header
            String line = br.readLine();

            while ((line = br.readLine()) != null) {

                if (line.trim().isEmpty()) {
                    continue;
                }

                String[] values = line.split(",");

                String vmId = values[0];

                double currentCpu =
                        Double.parseDouble(values[2]);

                double predictedCpu =
                        Double.parseDouble(values[3]);

                double ram =
                        Double.parseDouble(values[4]);

                VM vm = new VM(
                        vmId,
                        currentCpu,
                        predictedCpu,
                        ram
                );

                vms.add(vm);
            }
        }

        return vms;
    }
    public static List<RiskState> readRiskStates(
        String filePath) throws IOException {

    List<RiskState> riskStates = new ArrayList<>();

    try (BufferedReader br =
                 new BufferedReader(
                         new FileReader(filePath))) {

        // Skip header
        String line = br.readLine();

        while ((line = br.readLine()) != null) {

            if (line.trim().isEmpty()) {
                continue;
            }

            String[] values = line.split(",");

            if (values.length < 15) {
                System.out.println(
                        "Skipping invalid risk_state row: "
                        + line
                );
                continue;
            }

            String vmId =
                    values[0];

            long decisionTimestamp =
                    Long.parseLong(values[1]);

            double currentCpu =
                    Double.parseDouble(values[2]);

            double predictedCpuT5m =
                    Double.parseDouble(values[3]);

            double predictedCpuT10m =
                    Double.parseDouble(values[4]);

            double predictedCpuT15m =
                    Double.parseDouble(values[5]);

            double predictedCpuT30m =
                    Double.parseDouble(values[6]);

            double predictedCpuT45m =
                    Double.parseDouble(values[7]);

            double predictedCpuT60m =
                    Double.parseDouble(values[8]);

            double predictedMeanCpu =
                    Double.parseDouble(values[9]);

            double predictedPeakCpu =
                    Double.parseDouble(values[10]);

            double predictedStdCpu =
                    Double.parseDouble(values[11]);

            double volatilityScore =
                    Double.parseDouble(values[12]);

            double riskScore =
                    Double.parseDouble(values[13]);

            String riskState =
                    values[14];

            RiskState risk =
                    new RiskState(
                            vmId,
                            decisionTimestamp,
                            currentCpu,
                            predictedCpuT5m,
                            predictedCpuT10m,
                            predictedCpuT15m,
                            predictedCpuT30m,
                            predictedCpuT45m,
                            predictedCpuT60m,
                            predictedMeanCpu,
                            predictedPeakCpu,
                            predictedStdCpu,
                            volatilityScore,
                            riskScore,
                            riskState
                    );

            riskStates.add(risk);
        }
    }

    return riskStates;
}
public static List<DecisionEpoch> readDecisionEpochs(
        String filePath) throws IOException {

    List<RiskState> riskStates =
            readRiskStates(filePath);

    Map<Long, List<RiskState>> grouped =
            new LinkedHashMap<>();

    for (RiskState risk : riskStates) {

        grouped
                .computeIfAbsent(
                        risk.getDecisionTimestamp(),
                        key -> new ArrayList<>()
                )
                .add(risk);
    }

    List<DecisionEpoch> epochs =
            new ArrayList<>();

    for (Map.Entry<Long, List<RiskState>> entry
            : grouped.entrySet()) {

        DecisionEpoch epoch =
                new DecisionEpoch(
                        entry.getKey(),
                        entry.getValue()
                );

        epochs.add(epoch);
    }

    return epochs;
}
}