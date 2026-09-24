package org.vmplacement;

import java.io.File;

/**
 * Deterministic Trace Path Resolver for Bitbrains VM Workload Telemetry.
 *
 * Maps actual VM identities (e.g., "VM_001" -> "1.csv", "VM_003" -> "3.csv",
 * "VM_025" -> "25.csv") to fastStorage dataset files.
 */
public final class TraceResolver {

    private TraceResolver() {}

    private static final String[] CANDIDATE_DIRS = {
        "../dataset/fastStorage/2013-8",
        "dataset/fastStorage/2013-8",
        "C:/Users/aduru/OneDrive/Desktop/VM Placement/dataset/fastStorage/2013-8",
        "cloudsim/data",
        "data",
        "../cloudsim/data"
    };

    /**
     * Converts a VM ID string into the expected Bitbrains trace file name.
     * Examples:
     *   "VM_001" -> "1.csv"
     *   "VM_003" -> "3.csv"
     *   "VM_025" -> "25.csv"
     *   "1"      -> "1.csv"
     */
    public static String getTraceFileName(String vmIdStr) {
        if (vmIdStr == null || vmIdStr.isBlank()) {
            return "1.csv";
        }
        String clean = vmIdStr.toUpperCase().replace("VM_", "").replace("VM", "").trim();
        try {
            int vmNum = Integer.parseInt(clean);
            return vmNum + ".csv";
        } catch (NumberFormatException e) {
            return clean + ".csv";
        }
    }

    /**
     * Resolves the absolute or relative file path for a given VM ID's trace file.
     * Searches candidate directories and falls back safely if not found.
     *
     * @param vmIdStr VM identifier (e.g. "VM_001")
     * @return Path to existing trace CSV file
     */
    public static String resolveTracePath(String vmIdStr) {
        String traceFileName = getTraceFileName(vmIdStr);

        for (String dir : CANDIDATE_DIRS) {
            File f = new File(dir, traceFileName);
            if (f.exists() && f.isFile()) {
                return f.getPath();
            }
        }

        // Fallback to vm1.csv if the specific trace file is unavailable
        for (String dir : CANDIDATE_DIRS) {
            File f = new File(dir, "vm1.csv");
            if (f.exists() && f.isFile()) {
                return f.getPath();
            }
        }

        return "cloudsim/data/vm1.csv";
    }
}
