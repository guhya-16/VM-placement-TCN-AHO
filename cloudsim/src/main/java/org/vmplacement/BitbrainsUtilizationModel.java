package org.vmplacement;

import org.cloudsimplus.utilizationmodels.UtilizationModel;
import org.cloudsimplus.utilizationmodels.UtilizationModelAbstract;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Reads a real Bitbrains fastStorage CSV trace and replays its CPU usage [%]
 * column as a CloudSim Plus UtilizationModel, so a Cloudlet's "workload"
 * is driven by real recorded data instead of a made-up constant value.
 *
 * File format expected (Bitbrains fastStorage):
 *   Timestamp [ms];  CPU cores;  CPU capacity provisioned [MHZ];
 *   CPU usage [MHZ]; CPU usage [%]; Memory capacity provisioned [KB]; ...
 *
 * Note: despite the header saying "[ms]", the timestamp values are actually
 * Unix epoch SECONDS (10 digits), not milliseconds (13 digits) - the code
 * below treats them as seconds accordingly.
 */
public class BitbrainsUtilizationModel extends UtilizationModelAbstract implements UtilizationModel {

    private final List<Double> timeOffsetsSeconds = new ArrayList<>();
    private final List<Double> cpuUtilizationFraction = new ArrayList<>();

    /**
     * @param csvFilePath path to a single VM's Bitbrains CSV file
     */
    public BitbrainsUtilizationModel(String csvFilePath) {
        loadTrace(csvFilePath);
    }

    private void loadTrace(String csvFilePath) {
        try (BufferedReader reader = new BufferedReader(new FileReader(csvFilePath))) {
            String line = reader.readLine(); // header row, skip it
            Long firstTimestamp = null;

            while ((line = reader.readLine()) != null) {
                if (line.isBlank()) continue;

                // Split on ';' and trim to also strip any tab/whitespace left over
                String[] parts = line.split(";");
                if (parts.length < 5) continue;

                long timestampSeconds = Long.parseLong(parts[0].replace("\"", "").trim());
                double cpuUsagePercent = Double.parseDouble(parts[4].replace("\"", "").trim());

                if (firstTimestamp == null) {
                    firstTimestamp = timestampSeconds;
                }

                double offsetSeconds = timestampSeconds - firstTimestamp;
                double fraction = cpuUsagePercent / 100.0;

                timeOffsetsSeconds.add(offsetSeconds);
                cpuUtilizationFraction.add(fraction);
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to load Bitbrains trace: " + csvFilePath, e);
        }

        if (timeOffsetsSeconds.isEmpty()) {
            throw new RuntimeException("No data rows loaded from: " + csvFilePath);
        }
    }

    @Override
    protected double getUtilizationInternal(double simulationTimeSeconds) {
        // Find the last recorded sample at or before this simulation time.
        // Since samples are ~5 minutes apart, this holds each value constant
        // until the next real sample point - a simple, defensible approach.
        int index = 0;
        for (int i = 0; i < timeOffsetsSeconds.size(); i++) {
            if (timeOffsetsSeconds.get(i) <= simulationTimeSeconds) {
                index = i;
            } else {
                break;
            }
        }
        double fraction = cpuUtilizationFraction.get(index);
        return Math.max(0.01, Math.min(1.0, fraction));
    }
}
