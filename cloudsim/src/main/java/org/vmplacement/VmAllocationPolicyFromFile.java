package org.vmplacement;

import org.cloudsimplus.allocationpolicies.VmAllocationPolicyAbstract;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.vms.Vm;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

/**
 * Reads an externally-computed VM-to-Host placement (e.g. from Person 3's
 * HO or WA-AHO optimizer) and applies it directly, instead of computing
 * placement algorithmically like PABFD does.
 */
public class VmAllocationPolicyFromFile extends VmAllocationPolicyAbstract {

    private final Map<Long, Long> vmIdToHostId = new HashMap<>();

    public VmAllocationPolicyFromFile(String placementCsvPath) {
        loadPlacement(placementCsvPath);
    }

    private void loadPlacement(String csvPath) {
        try (BufferedReader reader = new BufferedReader(new FileReader(csvPath))) {
            reader.readLine(); // skip header
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isBlank()) continue;
                String[] parts = line.split(",");
                long vmId = Long.parseLong(parts[0].trim());
                long hostId = Long.parseLong(parts[1].trim());
                vmIdToHostId.put(vmId, hostId);
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to load placement.csv: " + csvPath, e);
        }
    }

    @Override
    protected Optional<Host> defaultFindHostForVm(Vm vm) {
        Long targetHostId = vmIdToHostId.get(vm.getId());
        if (targetHostId == null || targetHostId == -1) {
            return Optional.empty(); // Person 3 flagged this VM as unplaceable
        }
        return getHostList().stream()
            .filter(h -> h.getId() == targetHostId)
            .filter(h -> h.isSuitableForVm(vm)) // still enforce real constraints
            .findFirst();
    }
}
