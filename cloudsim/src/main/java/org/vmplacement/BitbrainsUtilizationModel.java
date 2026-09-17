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
    private final long startTimestampSeconds;

    /**
     * Constructs a trace utilization model starting from the trace's first recorded timestamp.
     *
     * @param csvFilePath path to a single VM's Bitbrains CSV file
     */
    public BitbrainsUtilizationModel(String csvFilePath) {
        this(csvFilePath, 0L);
    }

    /**
     * Constructs a trace utilization model aligned with a specific decision epoch timestamp.
     *
     * @param csvFilePath path to a single VM's Bitbrains CSV file
     * @param startTimestampSeconds Unix epoch timestamp (in seconds) corresponding to the decision epoch
     */
    public BitbrainsUtilizationModel(String csvFilePath, long startTimestampSeconds) {
        this.startTimestampSeconds = startTimestampSeconds;
        loadTrace(csvFilePath, startTimestampSeconds);
    }

    private void loadTrace(String csvFilePath, long targetStartTimestamp) {
        try (BufferedReader reader = new BufferedReader(new FileReader(csvFilePath))) {
            String line = reader.readLine(); // header row, skip it
            Long firstTimestamp = null;
            List<Long> allTimestamps = new ArrayList<>();
            List<Double> allFractions = new ArrayList<>();

            while ((line = reader.readLine()) != null) {
                if (line.isBlank()) continue;

                // Support both ';' and ',' delimiters
                String[] parts = line.contains(";") ? line.split(";") : line.split(",");
                if (parts.length < 5) continue;

                long timestampSeconds = Long.parseLong(parts[0].replace("\"", "").trim());
                double cpuUsagePercent = Double.parseDouble(parts[4].replace("\"", "").trim());

                allTimestamps.add(timestampSeconds);
                allFractions.add(cpuUsagePercent / 100.0);
            }

            if (allTimestamps.isEmpty()) {
                throw new RuntimeException("No data rows loaded from: " + csvFilePath);
            }

            // Determine reference starting timestamp
            long referenceTimestamp = allTimestamps.get(0);
            if (targetStartTimestamp > 0) {
                // Find if targetStartTimestamp is within the trace
                boolean foundTarget = false;
                for (long ts : allTimestamps) {
                    if (ts >= targetStartTimestamp) {
                        referenceTimestamp = targetStartTimestamp;
                        foundTarget = true;
                        break;
                    }
                }
                if (!foundTarget) {
                    System.err.printf("[WARN] Target timestamp %d not found in %s (range %d - %d). Falling back to trace start.%n",
                        targetStartTimestamp, csvFilePath, allTimestamps.get(0), allTimestamps.get(allTimestamps.size() - 1));
                    referenceTimestamp = allTimestamps.get(0);
                }
            }

            long firstAcceptedTs = -1;
            for (int i = 0; i < allTimestamps.size(); i++) {
                long ts = allTimestamps.get(i);
                if (ts >= referenceTimestamp) {
                    if (firstAcceptedTs < 0) {
                        firstAcceptedTs = ts;
                    }
                    timeOffsetsSeconds.add((double) (ts - firstAcceptedTs));
                    cpuUtilizationFraction.add(allFractions.get(i));
                }
            }

            // If slicing left an empty list, load from initial trace start
            if (timeOffsetsSeconds.isEmpty()) {
                long firstTs = allTimestamps.get(0);
                for (int i = 0; i < allTimestamps.size(); i++) {
                    timeOffsetsSeconds.add((double) (allTimestamps.get(i) - firstTs));
                    cpuUtilizationFraction.add(allFractions.get(i));
                }
            }

        } catch (IOException e) {
            throw new RuntimeException("Failed to load Bitbrains trace: " + csvFilePath, e);
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

    /**
     * Computes the mean requested utilization across the trace, within a
     * given time window - used to detect if delivered CPU fell short of demand.
     */
    public double getMeanRequestedUtilization(double windowSeconds) {
        double sum = 0;
        int count = 0;
        for (int i = 0; i < timeOffsetsSeconds.size(); i++) {
            if (timeOffsetsSeconds.get(i) <= windowSeconds) {
                sum += cpuUtilizationFraction.get(i);
                count++;
            }
        }
        return count == 0 ? 0 : sum / count;
    }
}

