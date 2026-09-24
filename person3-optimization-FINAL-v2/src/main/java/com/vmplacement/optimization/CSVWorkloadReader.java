package com.vmplacement.optimization;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class CSVWorkloadReader {
    private static final List<String> REQUIRED = List.of(
            "vm_id", "decision_timestamp", "current_cpu",
            "predicted_cpu_t5m", "predicted_cpu_t10m", "predicted_cpu_t15m",
            "predicted_cpu_t30m", "predicted_cpu_t45m", "predicted_cpu_t60m",
            "predicted_mean_cpu", "predicted_peak_cpu", "predicted_std_cpu",
            "volatility_score", "risk_score", "risk_state");

    private CSVWorkloadReader() {}

    public static List<DecisionEpoch> readDecisionEpochs(Path path) throws IOException {
        Map<Long, List<RiskState>> grouped = new LinkedHashMap<>();
        try (BufferedReader br = Files.newBufferedReader(path)) {
            String headerLine = br.readLine();
            if (headerLine == null) throw new IllegalArgumentException("Empty risk_state.csv: " + path);
            Map<String,Integer> header = headerMap(Csv.parseLine(headerLine));
            for (String required : REQUIRED) {
                if (!header.containsKey(required)) throw new IllegalArgumentException(
                        "Person 1 handoff is missing required column: " + required);
            }
            String line;
            long lineNumber = 1;
            while ((line = br.readLine()) != null) {
                lineNumber++;
                if (line.isBlank() || line.trim().startsWith("#")) continue;
                List<String> row = Csv.parseLine(line);
                try {
                    RiskState state = parse(row, header);
                    grouped.computeIfAbsent(state.decisionTimestamp(), k -> new ArrayList<>()).add(state);
                } catch (RuntimeException ex) {
                    throw new IllegalArgumentException("Invalid risk_state.csv at line " + lineNumber + ": " + ex.getMessage(), ex);
                }
            }
        }
        List<DecisionEpoch> epochs = new ArrayList<>();
        grouped.entrySet().stream().sorted(Map.Entry.comparingByKey())
                .forEach(e -> epochs.add(new DecisionEpoch(e.getKey(), e.getValue())));
        return epochs;
    }

    private static RiskState parse(List<String> r, Map<String,Integer> h) {
        String vm = Csv.require(r,h,"vm_id");
        long ts = Long.parseLong(Csv.require(r,h,"decision_timestamp"));
        return new RiskState(vm, ts,
                d(r,h,"current_cpu"), d(r,h,"predicted_cpu_t5m"), d(r,h,"predicted_cpu_t10m"),
                d(r,h,"predicted_cpu_t15m"), d(r,h,"predicted_cpu_t30m"), d(r,h,"predicted_cpu_t45m"),
                d(r,h,"predicted_cpu_t60m"), d(r,h,"predicted_mean_cpu"), d(r,h,"predicted_peak_cpu"),
                d(r,h,"predicted_std_cpu"), d(r,h,"volatility_score"), d(r,h,"risk_score"),
                Csv.require(r,h,"risk_state"));
    }

    private static double d(List<String> r, Map<String,Integer> h, String name) {
        return Double.parseDouble(Csv.require(r,h,name));
    }

    private static Map<String,Integer> headerMap(List<String> header) {
        Map<String,Integer> m = new LinkedHashMap<>();
        for (int i=0;i<header.size();i++) m.put(header.get(i).trim(), i);
        return m;
    }
}
