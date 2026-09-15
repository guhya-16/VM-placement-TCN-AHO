package org.vmplacement;

import org.cloudsimplus.allocationpolicies.VmAllocationPolicyAbstract;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.vms.Vm;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * =====================================================================================
 * VmAllocationPolicyFromFile
 * =====================================================================================
 * Reads an externally computed VM-to-Host placement (e.g., from Person 3's
 * Standard HO or Adaptive HO optimizer) and maps VMs directly to their assigned
 * Physical Machines (PMs / Hosts).
 *
 * Robust Validation Rules:
 * 1. Placement file must exist, be accessible, and contain valid data rows.
 * 2. Header row must strictly contain "vm_id" and "host_id" column names.
 * 3. VM IDs and Host IDs must be valid non-negative integers.
 * 4. Duplicate VM ID assignments are rejected.
 * 5. Every VM submitted to the simulation must have an explicit placement entry.
 * 6. Assigned Host IDs must exist within the Datacenter host pool (0 <= host_id < num_hosts).
 * =====================================================================================
 */
public class VmAllocationPolicyFromFile extends VmAllocationPolicyAbstract {

    private final String placementCsvPath;
    private final Map<Long, Long> vmIdToHostId = new LinkedHashMap<>();

    /**
     * Constructs the policy and loads/validates the placement CSV.
     *
     * @param placementCsvPath Path to placement.csv file.
     */
    public VmAllocationPolicyFromFile(String placementCsvPath) {
        this.placementCsvPath = placementCsvPath;
        loadAndValidatePlacement(placementCsvPath);
    }

    private void loadAndValidatePlacement(String csvPath) {
        File file = new File(csvPath);
        if (!file.exists() || !file.isFile()) {
            throw new IllegalArgumentException("Placement file does not exist or is not a valid file: " + csvPath);
        }
        if (file.length() == 0) {
            throw new IllegalArgumentException("Placement file is empty: " + csvPath);
        }

        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String headerLine = reader.readLine();
            if (headerLine == null || headerLine.isBlank()) {
                throw new IllegalArgumentException("Placement file header is missing: " + csvPath);
            }

            String[] headers = headerLine.split(",");
            if (headers.length < 2) {
                throw new IllegalArgumentException("Placement header must contain at least 2 columns (vm_id,host_id). Found: " + headerLine);
            }

            String col0 = headers[0].trim().toLowerCase();
            String col1 = headers[1].trim().toLowerCase();
            if (!col0.contains("vm_id") || !col1.contains("host_id")) {
                throw new IllegalArgumentException(
                    String.format("Invalid placement CSV header format. Expected 'vm_id,host_id', found '%s,%s' in %s",
                        headers[0].trim(), headers[1].trim(), csvPath)
                );
            }

            String line;
            int lineNumber = 1;
            while ((line = reader.readLine()) != null) {
                lineNumber++;
                if (line.isBlank()) continue;

                String[] parts = line.split(",");
                if (parts.length < 2) {
                    throw new IllegalArgumentException(
                        String.format("Malformed placement row at line %d in %s: '%s' (expected 'vm_id,host_id')",
                            lineNumber, csvPath, line)
                    );
                }

                long vmId;
                long hostId;

                try {
                    vmId = Long.parseLong(parts[0].trim());
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException(
                        String.format("Invalid non-integer vm_id '%s' at line %d in %s", parts[0].trim(), lineNumber, csvPath), e);
                }

                try {
                    hostId = Long.parseLong(parts[1].trim());
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException(
                        String.format("Invalid non-integer host_id '%s' at line %d in %s", parts[1].trim(), lineNumber, csvPath), e);
                }

                if (vmId < 0) {
                    throw new IllegalArgumentException(
                        String.format("Negative vm_id %d is not allowed at line %d in %s", vmId, lineNumber, csvPath));
                }

                if (vmIdToHostId.containsKey(vmId)) {
                    throw new IllegalArgumentException(
                        String.format("Duplicate placement definition for VM ID %d detected at line %d in %s (already mapped to Host %d)",
                            vmId, lineNumber, csvPath, vmIdToHostId.get(vmId)));
                }

                vmIdToHostId.put(vmId, hostId);
            }

            if (vmIdToHostId.isEmpty()) {
                throw new IllegalArgumentException("No valid placement rows found in: " + csvPath);
            }

        } catch (IOException e) {
            throw new RuntimeException("I/O error while reading placement file: " + csvPath, e);
        }
    }

    /**
     * Validates that all mapped host IDs exist within the datacenter.
     */
    public void validateWithDatacenterHosts(List<Host> hostList) {
        int hostCount = hostList.size();
        for (Map.Entry<Long, Long> entry : vmIdToHostId.entrySet()) {
            long vmId = entry.getKey();
            long hostId = entry.getValue();

            if (hostId < 0 || hostId >= hostCount) {
                throw new IllegalArgumentException(
                    String.format("Invalid Host ID %d for VM %d: Datacenter contains %d hosts (valid IDs: 0 to %d).",
                        hostId, vmId, hostCount, hostCount - 1)
                );
            }
        }
    }

    /**
     * Returns an unmodifiable view of the loaded VM -> Host placement map.
     */
    public Map<Long, Long> getPlacementMap() {
        return Collections.unmodifiableMap(vmIdToHostId);
    }

    public String getPlacementCsvPath() {
        return placementCsvPath;
    }

    @Override
    protected Optional<Host> defaultFindHostForVm(Vm vm) {
        Long targetHostId = vmIdToHostId.get(vm.getId());
        if (targetHostId == null) {
            throw new IllegalStateException(
                String.format("Missing placement entry for VM ID %d in %s. All submitted VMs must be present in placement.csv.",
                    vm.getId(), placementCsvPath)
            );
        }

        if (targetHostId < 0 || targetHostId >= getHostList().size()) {
            throw new IllegalStateException(
                String.format("VM ID %d is assigned to non-existent Host ID %d. Valid host range is [0, %d].",
                    vm.getId(), targetHostId, getHostList().size() - 1)
            );
        }

        return getHostList().stream()
            .filter(h -> h.getId() == targetHostId)
            .filter(h -> h.isSuitableForVm(vm)) // Enforce physical resource constraints
            .findFirst();
    }
}
