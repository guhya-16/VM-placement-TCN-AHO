package org.vmplacement;

import org.cloudsimplus.brokers.DatacenterBroker;
import org.cloudsimplus.brokers.DatacenterBrokerSimple;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletSimple;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.datacenters.Datacenter;
import org.cloudsimplus.datacenters.DatacenterSimple;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.hosts.HostSimple;
import org.cloudsimplus.power.models.PowerModelHost;
import org.cloudsimplus.power.models.PowerModelHostSimple;
import org.cloudsimplus.resources.Pe;
import org.cloudsimplus.resources.PeSimple;
import org.cloudsimplus.schedulers.cloudlet.CloudletSchedulerTimeShared;
import org.cloudsimplus.schedulers.vm.VmSchedulerTimeShared;
import org.cloudsimplus.utilizationmodels.UtilizationModel;
import org.cloudsimplus.utilizationmodels.UtilizationModelDynamic;
import org.cloudsimplus.utilizationmodels.UtilizationModelFull;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmSimple;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * =====================================================================================
 * PlacementEvaluator — Pure Physical Simulation & Candidate Evaluator for CloudSim Plus
 * =====================================================================================
 *
 * Responsibilities:
 * 1. Validate physical capacity constraints for the exact candidate mapping.
 * 2. If infeasible, immediately return feasible=false without altering or repairing placements.
 * 3. If feasible, construct the CloudSim datacenter, apply exact VM->PM mapping,
 *    attach real Bitbrains telemetry via BitbrainsUtilizationModel, and run discrete-event simulation.
 * 4. Extract real physical metrics (Energy Wh, SLA violations, active PMs, mean CPU, placement changes, migration cost).
 * 5. Return a clean PlacementEvaluationResult.
 * =====================================================================================
 */
public class PlacementEvaluator {

    public static final double OBSERVATION_WINDOW_SECONDS = 3600.0; // 1-hour observation window

    // PM Hardware Specifications matching PM.java and host_manifest.csv
    private static final int PM1_PES = 2;
    private static final long PM1_MIPS = 2660;
    private static final long PM1_RAM_MB = 4 * 1024;
    private static final long PM1_STORAGE_MB = 160 * 1024;
    private static final double PM1_MAX_POWER = 135;
    private static final double PM1_STATIC_POWER = 93.7;

    private static final int PM2_PES = 4;
    private static final long PM2_MIPS = 3067;
    private static final long PM2_RAM_MB = 8 * 1024;
    private static final long PM2_STORAGE_MB = 250 * 1024;
    private static final double PM2_MAX_POWER = 113;
    private static final double PM2_STATIC_POWER = 42.3;

    private static final int PM3_PES = 12;
    private static final long PM3_MIPS = 3067;
    private static final long PM3_RAM_MB = 16 * 1024;
    private static final long PM3_STORAGE_MB = 500 * 1024;
    private static final double PM3_MAX_POWER = 222;
    private static final double PM3_STATIC_POWER = 58.4;

    private static final long HOST_BW = 10_000;

    /**
     * Evaluates a single candidate placement synchronously against CloudSim Plus physics.
     *
     * @param request PlacementRequest containing the exact candidate mapping and VM/PM specs
     * @param state Current DatacenterState containing the finalized previous epoch state
     * @return PlacementEvaluationResult containing real physical simulation metrics
     */
    public static PlacementEvaluationResult evaluate(PlacementRequest request, DatacenterState state) {
        if (request == null) {
            return PlacementEvaluationResult.infeasible(0, 0, "NULL_REQUEST", "PlacementRequest cannot be null");
        }

        List<VmPlacementSpec> vmPlacements = request.getVmPlacements();
        if (vmPlacements == null || vmPlacements.isEmpty()) {
            return PlacementEvaluationResult.infeasible(
                    request.getEpoch(), request.getIteration(), request.getCandidateId(),
                    "No VMs provided in PlacementRequest");
        }

        // Build PM specs (use request specs if provided, else standard 20 PM configuration)
        List<PmSpec> pmSpecs = request.getPmSpecs();
        if (pmSpecs == null || pmSpecs.isEmpty()) {
            pmSpecs = createDefaultPmSpecs();
        }
        int numPms = pmSpecs.size();

        // 1. FEASIBILITY VALIDATION
        // Check PM bounds and aggregate resource capacity violations
        int[] usedPes = new int[numPms];
        double[] usedRam = new double[numPms];
        double[] usedStorage = new double[numPms];
        double[] usedMips = new double[numPms];

        for (VmPlacementSpec vm : vmPlacements) {
            int pmId = vm.getPmId();
            if (pmId < 0 || pmId >= numPms) {
                return PlacementEvaluationResult.infeasible(
                        request.getEpoch(), request.getIteration(), request.getCandidateId(),
                        String.format("Invalid PM ID %d for VM %s. Total PMs available: %d", pmId, vm.getVmId(), numPms));
            }

            usedPes[pmId] += vm.getPeCount();
            usedRam[pmId] += vm.getRamMb();
            usedStorage[pmId] += vm.getStorageMb();

            double cpuDemandFraction = vm.getCurrentCpuUtilization() > 1.0
                    ? vm.getCurrentCpuUtilization() / 100.0
                    : vm.getCurrentCpuUtilization();
            if (cpuDemandFraction <= 0.0) {
                cpuDemandFraction = 1.0; // Default to full provisioned capacity if not specified
            }
            usedMips[pmId] += vm.getMips() * cpuDemandFraction;
        }

        for (int i = 0; i < numPms; i++) {
            PmSpec pm = pmSpecs.get(i);
            if (usedPes[i] > pm.getPeCount()) {
                return PlacementEvaluationResult.infeasible(
                        request.getEpoch(), request.getIteration(), request.getCandidateId(),
                        String.format("PM %d (%s) PE capacity exceeded: requested %d PEs, capacity %d PEs",
                                i, pm.getName(), usedPes[i], pm.getPeCount()));
            }
            if (usedRam[i] > pm.getRamMb() + 1e-6) {
                return PlacementEvaluationResult.infeasible(
                        request.getEpoch(), request.getIteration(), request.getCandidateId(),
                        String.format("PM %d (%s) RAM capacity exceeded: requested %.0f MB, capacity %.0f MB",
                                i, pm.getName(), usedRam[i], pm.getRamMb()));
            }
            if (usedStorage[i] > pm.getStorageMb() + 1e-6) {
                return PlacementEvaluationResult.infeasible(
                        request.getEpoch(), request.getIteration(), request.getCandidateId(),
                        String.format("PM %d (%s) Storage capacity exceeded: requested %.0f MB, capacity %.0f MB",
                                i, pm.getName(), usedStorage[i], pm.getStorageMb()));
            }
            if (usedMips[i] > pm.getTotalMips() + 1e-6) {
                return PlacementEvaluationResult.infeasible(
                        request.getEpoch(), request.getIteration(), request.getCandidateId(),
                        String.format("PM %d (%s) CPU MIPS capacity exceeded: requested %.1f MIPS, capacity %.1f MIPS",
                                i, pm.getName(), usedMips[i], pm.getTotalMips()));
            }
        }

        // 2. DISCRETE-EVENT SIMULATION WITH CLOUDSIM PLUS
        final CloudSimPlus simulation = new CloudSimPlus();
        final List<Host> hostList = createHostsFromSpecs(pmSpecs);

        // Build in-memory placement mapping (numeric VM ID -> Host ID)
        final Map<Long, Long> inMemoryPlacementMap = new LinkedHashMap<>();
        for (int idx = 0; idx < vmPlacements.size(); idx++) {
            VmPlacementSpec vmSpec = vmPlacements.get(idx);
            long vmNumId = parseVmNumericId(vmSpec.getVmId(), idx);
            inMemoryPlacementMap.put(vmNumId, (long) vmSpec.getPmId());
        }

        final VmAllocationPolicyFromFile allocationPolicy =
                new VmAllocationPolicyFromFile(inMemoryPlacementMap, request.getCandidateId());
        allocationPolicy.validateWithDatacenterHosts(hostList);

        final Datacenter datacenter = new DatacenterSimple(simulation, hostList, allocationPolicy);
        datacenter.setSchedulingInterval(10); // Sample every 10 simulated seconds

        final DatacenterBroker broker = new DatacenterBrokerSimple(simulation);

        // Create VMs and attach Cloudlets with Bitbrains telemetry
        final List<Vm> vmList = new ArrayList<>();
        final List<Cloudlet> cloudletList = new ArrayList<>();
        final Map<Long, Double> vmRequestedUtilizationMap = new HashMap<>();

        final UtilizationModel ramUtilizationModel = new UtilizationModelDynamic(0.3);
        final UtilizationModel bwUtilizationModel = new UtilizationModelFull();

        for (int idx = 0; idx < vmPlacements.size(); idx++) {
            VmPlacementSpec spec = vmPlacements.get(idx);
            long vmNumId = parseVmNumericId(spec.getVmId(), idx);

            Vm vm = new VmSimple(vmNumId, spec.getMips(), spec.getPeCount());
            vm.setRam((long) spec.getRamMb()).setBw(1000).setSize((long) spec.getStorageMb());
            vm.setCloudletScheduler(new CloudletSchedulerTimeShared());
            vm.enableUtilizationStats();
            vmList.add(vm);

            // Deterministic trace resolution
            String tracePath = TraceResolver.resolveTracePath(spec.getVmId());
            BitbrainsUtilizationModel cpuModel =
                    new BitbrainsUtilizationModel(tracePath, request.getTimestamp());

            Cloudlet cloudlet = new CloudletSimple(50_000_000L, 1);
            cloudlet.setUtilizationModelCpu(cpuModel);
            cloudlet.setUtilizationModelRam(ramUtilizationModel);
            cloudlet.setUtilizationModelBw(bwUtilizationModel);
            cloudletList.add(cloudlet);

            vmRequestedUtilizationMap.put(vmNumId, cpuModel.getMeanRequestedUtilization(OBSERVATION_WINDOW_SECONDS));
        }

        broker.submitVmList(vmList);
        broker.submitCloudletList(cloudletList);

        simulation.terminateAt(OBSERVATION_WINDOW_SECONDS);
        simulation.start();

        // 3. PHYSICAL METRICS EXTRACTION
        double totalEnergyWh = 0.0;
        final double observationWindowHours = OBSERVATION_WINDOW_SECONDS / 3600.0; // Strictly 1.0 hour
        final Map<Integer, Double> hostCpuMap = new LinkedHashMap<>();

        for (Host host : datacenter.getHostList()) {
            double meanUtil = host.getCpuUtilizationStats().getMean();
            double meanPowerWatts = host.getPowerModel().getPower(meanUtil);
            double hostEnergyWh = meanPowerWatts * observationWindowHours;
            totalEnergyWh += hostEnergyWh;
            hostCpuMap.put((int) host.getId(), meanUtil);
        }

        int activePmCount = (int) datacenter.getHostList().stream()
                .filter(h -> !h.getVmCreatedList().isEmpty())
                .count();

        double meanActiveCpu = datacenter.getHostList().stream()
                .filter(h -> !h.getVmCreatedList().isEmpty())
                .mapToDouble(h -> h.getCpuUtilizationStats().getMean())
                .average()
                .orElse(0.0);

        // SLA Violations
        int slaViolations = 0;
        double slaThreshold = request.getSlaThreshold();
        for (Vm vm : vmList) {
            double delivered = vm.getCpuUtilizationStats().getMean();
            double requested = vmRequestedUtilizationMap.getOrDefault(vm.getId(), 0.0);
            double gap = requested - delivered;
            if (gap > slaThreshold) {
                slaViolations++;
            }
        }
        double slaViolationRate = vmList.isEmpty() ? 0.0 : (double) slaViolations / vmList.size();

        // Inter-epoch placement changes calculated against finalized previous state
        int placementChanges = (state != null)
                ? state.computePlacementChanges(request.getVmToPmMap())
                : 0;

        // Migration cost: directly derived from placement changes count (or RAM-weighted)
        double migrationCost = (double) placementChanges;

        return new PlacementEvaluationResult(
                request.getEpoch(),
                request.getIteration(),
                request.getCandidateId(),
                true,
                totalEnergyWh,
                slaViolations,
                slaViolationRate,
                activePmCount,
                meanActiveCpu,
                placementChanges,
                migrationCost,
                hostCpuMap,
                "SUCCESS"
        );
    }

    private static List<PmSpec> createDefaultPmSpecs() {
        List<PmSpec> list = new ArrayList<>();
        // 10 x PM1 (Hosts 0-9)
        for (int i = 0; i < 10; i++) {
            list.add(new PmSpec(i, String.format("PM_%03d", i + 1), PM1_PES, PM1_MIPS, PM1_RAM_MB, PM1_STORAGE_MB, PM1_MAX_POWER, PM1_STATIC_POWER));
        }
        // 6 x PM2 (Hosts 10-15)
        for (int i = 10; i < 16; i++) {
            list.add(new PmSpec(i, String.format("PM_%03d", i + 1), PM2_PES, PM2_MIPS, PM2_RAM_MB, PM2_STORAGE_MB, PM2_MAX_POWER, PM2_STATIC_POWER));
        }
        // 4 x PM3 (Hosts 16-19)
        for (int i = 16; i < 20; i++) {
            list.add(new PmSpec(i, String.format("PM_%03d", i + 1), PM3_PES, PM3_MIPS, PM3_RAM_MB, PM3_STORAGE_MB, PM3_MAX_POWER, PM3_STATIC_POWER));
        }
        return list;
    }

    private static List<Host> createHostsFromSpecs(List<PmSpec> pmSpecs) {
        List<Host> hostList = new ArrayList<>();
        for (PmSpec spec : pmSpecs) {
            List<Pe> peList = new ArrayList<>();
            for (int p = 0; p < spec.getPeCount(); p++) {
                peList.add(new PeSimple(spec.getMipsPerPe()));
            }
            Host host = new HostSimple((long) spec.getRamMb(), HOST_BW, (long) spec.getStorageMb(), peList);
            host.setVmScheduler(new VmSchedulerTimeShared());
            PowerModelHost powerModel = new PowerModelHostSimple(spec.getMaxPowerWatts(), spec.getStaticPowerWatts());
            host.setPowerModel(powerModel);
            host.enableUtilizationStats();
            hostList.add(host);
        }
        return hostList;
    }

    private static long parseVmNumericId(String vmIdStr, int fallbackIndex) {
        if (vmIdStr == null || vmIdStr.isBlank()) {
            return fallbackIndex;
        }
        String clean = vmIdStr.toUpperCase().replace("VM_", "").replace("VM", "").trim();
        try {
            return Long.parseLong(clean) - 1; // 0-indexed internal CloudSim VM ID
        } catch (NumberFormatException e) {
            return fallbackIndex;
        }
    }
}
