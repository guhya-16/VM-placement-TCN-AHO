package org.vmplacement;

import org.cloudsimplus.allocationpolicies.VmAllocationPolicy;
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

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * =====================================================================================
 * VM Placement Simulation Engine — CloudSim Plus
 * =====================================================================================
 *
 * Research Title:
 * "Predictive Energy-Efficient VM Placement in Cloud Using ML and Adaptive
 * Hippopotamus Optimization"
 *
 * Supported Execution Modes:
 * 1. MODE A: SYNTHETIC_PABFD_BASELINE
 *    - Creates a fixed cohort of 24 heterogeneous synthetic VMs (Types 1-4).
 *    - Uses Power-Aware Best Fit Decreasing (PABFD) heuristic allocation.
 *    - Primary role: Initial CloudSim framework stress-testing & heuristic baseline.
 *
 * 2. MODE B: REAL_DATA_EXTERNAL_PLACEMENT (Default)
 *    - Dynamically loads active VM count N from the current decision epoch's
 *      placement.csv (e.g., N=5 for Epoch 1, N=6, N=4, etc. up to 7 validated VMs).
 *    - Never hardcodes 24, 5, or 7 VMs; N is completely data-driven.
 *    - Uses VmAllocationPolicyFromFile with strict integrity & constraint validation.
 *    - Primary role: Evaluating real-data optimization placements (Standard HO &
 *      Adaptive HO) driven by multi-step TCN workload predictions.
 *
 * Note on VM Resource Specifications:
 * In Real-Data mode, all dynamically created VMs use the medium specification
 * (2 PEs, 1000 MIPS, 1024 MB RAM, 60 GB storage). This is a temporary modeling
 * simplification due to the absence of validated per-trace physical hardware telemetry
 * at this small dataset scale.
 * =====================================================================================
 */
public class Simulation {

    public enum ExecutionMode {
        SYNTHETIC_PABFD_BASELINE,
        REAL_DATA_EXTERNAL_PLACEMENT
    }

    // Default execution mode (can also be overridden via command-line argument: "pabfd" or "real")
    public static ExecutionMode CURRENT_MODE = ExecutionMode.REAL_DATA_EXTERNAL_PLACEMENT;
    public static String placementFile = "placement.csv";
    public static String algorithmLabel = null;
    public static String epochLabel = "N/A";
    public static String timestampLabel = "N/A";

    private static final int SYNTHETIC_VM_COUNT = 24;
    private static final double OBSERVATION_WINDOW_SECONDS = 3600.0; // Fixed 1-hour observation window

    // PM specifications matching project PM/VM configuration table:
    // PM1 (Hosts 0-9)  : 2 PEs, 2660 MIPS/PE, 4 GB RAM, 160 GB Storage, Pmax=135W, Pstatic=93.7W
    private static final int PM1_PES = 2;
    private static final long PM1_MIPS = 2660;
    private static final long PM1_RAM_MB = 4 * 1024;
    private static final long PM1_STORAGE_MB = 160 * 1024;
    private static final double PM1_MAX_POWER = 135;
    private static final double PM1_STATIC_POWER = 93.7;

    // PM2 (Hosts 10-15): 4 PEs, 3067 MIPS/PE, 8 GB RAM, 250 GB Storage, Pmax=113W, Pstatic=42.3W
    private static final int PM2_PES = 4;
    private static final long PM2_MIPS = 3067;
    private static final long PM2_RAM_MB = 8 * 1024;
    private static final long PM2_STORAGE_MB = 250 * 1024;
    private static final double PM2_MAX_POWER = 113;
    private static final double PM2_STATIC_POWER = 42.3;

    // PM3 (Hosts 16-19): 12 PEs, 3067 MIPS/PE, 16 GB RAM, 500 GB Storage, Pmax=222W, Pstatic=58.4W
    private static final int PM3_PES = 12;
    private static final long PM3_MIPS = 3067;
    private static final long PM3_RAM_MB = 16 * 1024;
    private static final long PM3_STORAGE_MB = 500 * 1024;
    private static final double PM3_MAX_POWER = 222;
    private static final double PM3_STATIC_POWER = 58.4;

    private static final long HOST_BW = 10_000;

    /**
     * Synchronous candidate placement evaluation entry point for Person 3.
     * Evaluates exactly the candidate mapping supplied in PlacementRequest
     * against real CloudSim Plus physical simulation.
     *
     * @param request Candidate placement evaluation request
     * @param state DatacenterState tracking the finalized previous epoch state
     * @return Physical metrics in PlacementEvaluationResult
     */
    public static PlacementEvaluationResult evaluatePlacement(PlacementRequest request, DatacenterState state) {
        return PlacementEvaluator.evaluate(request, state);
    }

    public static void main(String[] args) {
        // Parse CLI arguments: [placement_file_or_mode] [algorithm_label] [epoch] [timestamp]
        if (args != null && args.length > 0) {
            String arg0 = args[0].trim();
            if (arg0.equalsIgnoreCase("pabfd") || arg0.equalsIgnoreCase("synthetic")) {
                CURRENT_MODE = ExecutionMode.SYNTHETIC_PABFD_BASELINE;
                algorithmLabel = "PABFD_SYNTHETIC_BASELINE";
            } else {
                CURRENT_MODE = ExecutionMode.REAL_DATA_EXTERNAL_PLACEMENT;
                placementFile = arg0;
            }

            if (args.length > 1) {
                algorithmLabel = args[1].trim();
            }
            if (args.length > 2) {
                epochLabel = args[2].trim();
            }
            if (args.length > 3) {
                timestampLabel = args[3].trim();
            }
        }

        if (algorithmLabel == null) {
            if (CURRENT_MODE == ExecutionMode.SYNTHETIC_PABFD_BASELINE) {
                algorithmLabel = "PABFD_SYNTHETIC_BASELINE";
            } else if (placementFile.contains("standard")) {
                algorithmLabel = "Standard_HO_Epoch8";
                epochLabel = "8";
                timestampLabel = "1378606200";
            } else if (placementFile.contains("adaptive")) {
                algorithmLabel = "Adaptive_HO_Epoch8";
                epochLabel = "8";
                timestampLabel = "1378606200";
            } else {
                algorithmLabel = "HO_INTEGRATED";
            }
        }

        final CloudSimPlus simulation = new CloudSimPlus();
        final Datacenter datacenter = createDatacenter(simulation);
        final DatacenterBroker broker = new DatacenterBrokerSimple(simulation);

        final Map<Long, Double> vmRequestedUtilizationMap = new HashMap<>();
        final List<Vm> vmList = createVms();
        final List<Cloudlet> cloudletList = createCloudlets(vmList, vmRequestedUtilizationMap);

        // Print Startup Validation Banner
        printStartupBanner(datacenter, vmList);

        if (CURRENT_MODE == ExecutionMode.SYNTHETIC_PABFD_BASELINE) {
            // Sort VMs by descending expected utilization (Decreasing part of PABFD)
            vmList.sort((v1, v2) -> Double.compare(
                v2.getExpectedHostCpuUtilization(0),
                v1.getExpectedHostCpuUtilization(0)
            ));
        }

        broker.submitVmList(vmList);
        broker.submitCloudletList(cloudletList);

        // Fixed observation window: VM placement evaluation under steady workload
        simulation.terminateAt(OBSERVATION_WINDOW_SECONDS);
        simulation.start();

        System.out.println("\n========== UTILIZATION VERIFICATION ==========");
        for (Vm vm : vmList) {
            System.out.printf("VM %2d | CPU Utilization Mean: %.4f | Max: %.4f | Min: %.4f%n",
                vm.getId(),
                vm.getCpuUtilizationStats().getMean(),
                vm.getCpuUtilizationStats().getMax(),
                vm.getCpuUtilizationStats().getMin());
        }

        System.out.println("\n========== SLA VIOLATION CHECK ==========");
        int slaViolations = 0;
        final double SLA_TOLERANCE = 0.05;

        for (Vm vm : vmList) {
            double delivered = vm.getCpuUtilizationStats().getMean();
            double requested = vmRequestedUtilizationMap.getOrDefault(vm.getId(), 0.0);
            double gap = requested - delivered;

            System.out.printf("VM %2d | Requested: %.4f | Delivered: %.4f | Gap: %.4f%n",
                vm.getId(), requested, delivered, gap);

            if (gap > SLA_TOLERANCE) {
                slaViolations++;
                System.out.printf("  -> SLA VIOLATION (gap %.4f exceeds tolerance %.2f)%n", gap, SLA_TOLERANCE);
            }
        }
        double slaViolationRate = vmList.isEmpty() ? 0.0 : (double) slaViolations / vmList.size();
        System.out.printf("Total SLA violations: %d / %d VMs (%.2f%%)%n",
            slaViolations, vmList.size(), slaViolationRate * 100);

        final List<Cloudlet> finishedCloudlets = broker.getCloudletFinishedList();
        System.out.println("\n========== SIMULATION RESULTS ==========");
        finishedCloudlets.forEach(c ->
            System.out.printf(
                "Cloudlet %2d | VM %2d | Status %s | Start %6.2f | Finish %6.2f%n",
                c.getId(), c.getVm().getId(), c.getStatus(),
                c.getStartTime(), c.getFinishTime()
            )
        );

        final int activePmCount = (int) datacenter.getHostList().stream()
            .filter(host -> !host.getVmCreatedList().isEmpty())
            .count();

        System.out.printf("\nActive (utilized) hosts during simulation: %d / %d%n",
            activePmCount, datacenter.getHostList().size());

        System.out.println("\n========== POWER & ENERGY CONSUMPTION ==========");
        double totalEnergyWh = 0;
        final double observationWindowHours = OBSERVATION_WINDOW_SECONDS / 3600.0; // Strictly 1.0 hour

        for (Host host : datacenter.getHostList()) {
            double meanUtilization = host.getCpuUtilizationStats().getMean();
            double meanPowerWatts = host.getPowerModel().getPower(meanUtilization);
            double hostEnergyWh = meanPowerWatts * observationWindowHours;
            int vmsPlaced = host.getVmCreatedList().size();

            if (vmsPlaced > 0 || CURRENT_MODE == ExecutionMode.SYNTHETIC_PABFD_BASELINE) {
                System.out.printf("Host %2d | VMs: %2d | Mean Utilization: %5.2f%% | Mean Power: %6.2f W | Energy: %7.4f Wh%n",
                    host.getId(), vmsPlaced, meanUtilization * 100, meanPowerWatts, hostEnergyWh);
            }

            totalEnergyWh += hostEnergyWh;
        }
        System.out.printf("\nTotal energy consumed across all %d hosts: %.4f Wh (over %.2f hour window)%n",
            datacenter.getHostList().size(), totalEnergyWh, observationWindowHours);

        double avgCpuUtil = datacenter.getHostList().stream()
            .filter(host -> !host.getVmCreatedList().isEmpty())
            .mapToDouble(host -> host.getCpuUtilizationStats().getMean())
            .average()
            .orElse(0.0);

        String algorithmName = (CURRENT_MODE == ExecutionMode.REAL_DATA_EXTERNAL_PLACEMENT)
            ? "HO_INTEGRATED" : "PABFD";

        exportManifests(datacenter, vmList);
        exportResults(algorithmLabel, epochLabel, timestampLabel, datacenter, vmList, totalEnergyWh, slaViolations, avgCpuUtil, activePmCount);
    }

    /**
     * Prints startup verification banner displaying execution mode, placement file details,
     * VM count, host count, and explicit VM -> Host assignments.
     */
    private static void printStartupBanner(Datacenter datacenter, List<Vm> vmList) {
        System.out.println("==================================================");
        System.out.println("CLOUDSIM PLUS SIMULATION INITIALIZATION");
        System.out.println("==================================================");
        System.out.printf("Algorithm Label        : %s%n", algorithmLabel);
        System.out.printf("Mode                   : %s%n",
            CURRENT_MODE == ExecutionMode.REAL_DATA_EXTERNAL_PLACEMENT
                ? "REAL-DATA EXTERNAL PLACEMENT" : "SYNTHETIC 24-VM PABFD BASELINE");

        if (CURRENT_MODE == ExecutionMode.REAL_DATA_EXTERNAL_PLACEMENT) {
            String placementPath = resolvePlacementPath(placementFile);
            System.out.printf("Placement File         : %s%n", placementPath);
            System.out.printf("Placement Rows (N)     : %d%n", vmList.size());
            System.out.printf("VMs Initialized        : %d%n", vmList.size());
            System.out.printf("Datacenter Hosts       : %d PMs%n", datacenter.getHostList().size());
            System.out.printf("Observation Window     : %.2f seconds (%.2f hour)%n",
                OBSERVATION_WINDOW_SECONDS, OBSERVATION_WINDOW_SECONDS / 3600.0);
            System.out.println("==================================================");
            System.out.println("VM -> Host Placement Mapping:");

            if (datacenter.getVmAllocationPolicy() instanceof VmAllocationPolicyFromFile filePolicy) {
                Map<Long, Long> mapping = filePolicy.getPlacementMap();
                for (Map.Entry<Long, Long> entry : mapping.entrySet()) {
                    long vmId = entry.getKey();
                    long hostId = entry.getValue();
                    String pmName = String.format("PM_%03d", hostId + 1); // 1-indexed PM label
                    System.out.printf("  VM %2d -> Host %2d (%s)%n", vmId, hostId, pmName);
                }
            }
        } else {
            System.out.printf("Synthetic VM Cohort    : %d VMs (Types 1-4)%n", vmList.size());
            System.out.printf("Datacenter Hosts       : %d PMs%n", datacenter.getHostList().size());
            System.out.printf("Allocation Policy      : Power-Aware Best Fit Decreasing (PABFD)%n");
            System.out.printf("Observation Window     : %.2f seconds (%.2f hour)%n",
                OBSERVATION_WINDOW_SECONDS, OBSERVATION_WINDOW_SECONDS / 3600.0);
        }
        System.out.println("==================================================\n");
    }

    /**
     * Primary entry point for VM creation:
     * - In REAL_DATA_EXTERNAL_PLACEMENT mode: dynamically creates N VMs matching placement.csv rows.
     * - In SYNTHETIC_PABFD_BASELINE mode: creates the fixed 24-VM synthetic benchmark cohort.
     */
    public static List<Vm> createVms() {
        if (CURRENT_MODE == ExecutionMode.REAL_DATA_EXTERNAL_PLACEMENT) {
            String placementPath = resolvePlacementPath(placementFile);
            return createVmsFromPlacement(placementPath);
        } else {
            return createSyntheticVms(SYNTHETIC_VM_COUNT);
        }
    }

    /**
     * Creates VMs dynamically based on the exact per-VM specifications in placement.csv.
     * Single source of truth: Reads vm_id, host_id, pes, mips, ram_mb, storage_mb.
     */
    public static List<Vm> createVmsFromPlacement(String placementCsvPath) {
        final List<Vm> vmList = new ArrayList<>();
        File file = new File(placementCsvPath);
        if (!file.exists() || !file.isFile()) {
            throw new IllegalArgumentException("Placement file not found: " + placementCsvPath);
        }

        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String header = reader.readLine();
            if (header == null) {
                throw new IllegalArgumentException("Placement file is empty: " + placementCsvPath);
            }

            String line;
            int lineNumber = 1;
            while ((line = reader.readLine()) != null) {
                lineNumber++;
                if (line.isBlank()) continue;
                String[] parts = line.split(",");
                if (parts.length < 2) continue;

                String vmIdStr = parts[0].trim();
                long vmId;
                if (vmIdStr.toUpperCase().startsWith("VM_")) {
                    vmId = Long.parseLong(vmIdStr.substring(3)) - 1;
                } else {
                    vmId = Long.parseLong(vmIdStr);
                }

                int pes = 1;
                long mips = 1000;
                long ramMb = 1024;
                long storageMb = 60 * 1024;

                if (parts.length >= 6) {
                    pes = Integer.parseInt(parts[2].trim());
                    mips = (long) Double.parseDouble(parts[3].trim());
                    ramMb = (long) Double.parseDouble(parts[4].trim());
                    storageMb = (long) Double.parseDouble(parts[5].trim());
                }

                Vm vm = createVm(vmId, pes, mips, ramMb, storageMb);
                vmList.add(vm);
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to read placement file: " + placementCsvPath, e);
        }

        if (vmList.isEmpty()) {
            throw new IllegalArgumentException("Placement file contains no valid VM rows: " + placementCsvPath);
        }

        return vmList;
    }

    /**
     * Synthetic 24-VM baseline generator (preserves original PABFD baseline run mode).
     */
    public static List<Vm> createSyntheticVms(int count) {
        final List<Vm> vmList = new ArrayList<>();
        int[] synPes = { 1, 2, 3, 4 };
        long[] synMips = { 500, 1000, 1500, 2000 };
        long[] synRam = { 512, 1024, 2048, 3072 };
        long[] synStorage = { 40 * 1024, 60 * 1024, 80 * 1024, 100 * 1024 };

        for (int i = 0; i < count; i++) {
            int typeIndex = i % 4;
            Vm vm = createVm(i, synPes[typeIndex], synMips[typeIndex], synRam[typeIndex], synStorage[typeIndex]);
            vmList.add(vm);
        }
        return vmList;
    }

    private static Vm createVm(long id, int pes, long mips, long ramMB, long storageMB) {
        final Vm vm = new VmSimple(id, mips, pes);
        vm.setRam(ramMB).setBw(1000).setSize(storageMB);
        vm.setCloudletScheduler(new CloudletSchedulerTimeShared());
        vm.enableUtilizationStats();
        return vm;
    }

    private static Datacenter createDatacenter(CloudSimPlus simulation) {
        final List<Host> hostList = new ArrayList<>();

        // 10 x PM1, 6 x PM2, 4 x PM3 = 20 PMs total
        for (int i = 0; i < 10; i++) {
            hostList.add(createHost(PM1_PES, PM1_MIPS, PM1_RAM_MB, PM1_STORAGE_MB, PM1_MAX_POWER, PM1_STATIC_POWER));
        }
        for (int i = 0; i < 6; i++) {
            hostList.add(createHost(PM2_PES, PM2_MIPS, PM2_RAM_MB, PM2_STORAGE_MB, PM2_MAX_POWER, PM2_STATIC_POWER));
        }
        for (int i = 0; i < 4; i++) {
            hostList.add(createHost(PM3_PES, PM3_MIPS, PM3_RAM_MB, PM3_STORAGE_MB, PM3_MAX_POWER, PM3_STATIC_POWER));
        }

        final VmAllocationPolicy allocationPolicy;
        if (CURRENT_MODE == ExecutionMode.REAL_DATA_EXTERNAL_PLACEMENT) {
            String placementPath = resolvePlacementPath(placementFile);
            VmAllocationPolicyFromFile filePolicy = new VmAllocationPolicyFromFile(placementPath);
            filePolicy.validateWithDatacenterHosts(hostList);
            allocationPolicy = filePolicy;
        } else {
            allocationPolicy = new VmAllocationPolicyPabfd();
        }

        final Datacenter datacenter = new DatacenterSimple(simulation, hostList, allocationPolicy);
        datacenter.setSchedulingInterval(10); // Sample every 10 simulated seconds
        return datacenter;
    }

    private static Host createHost(int numPes, long mipsPerPe, long ramMB, long storageMB,
                                    double maxPowerWatts, double staticPowerWatts) {
        final List<Pe> peList = new ArrayList<>();
        for (int i = 0; i < numPes; i++) {
            peList.add(new PeSimple(mipsPerPe));
        }
        final Host host = new HostSimple(ramMB, HOST_BW, storageMB, peList);
        host.setVmScheduler(new VmSchedulerTimeShared());

        final PowerModelHost powerModel = new PowerModelHostSimple(maxPowerWatts, staticPowerWatts);
        host.setPowerModel(powerModel);
        host.enableUtilizationStats();

        return host;
    }

    private static final String[] BITBRAINS_TRACE_FILES = {
        "1.csv", "3.csv", "11.csv", "12.csv", "13.csv", "23.csv", "25.csv"
    };

    private static List<Cloudlet> createCloudlets(List<Vm> vmList, Map<Long, Double> vmRequestedUtilizationMap) {
        final List<Cloudlet> cloudletList = new ArrayList<>();
        final UtilizationModel ramUtilizationModel = new UtilizationModelDynamic(0.3);
        final UtilizationModel bwUtilizationModel = new UtilizationModelFull();

        long startEpochTimestamp = 0L;
        try {
            if (timestampLabel != null && !timestampLabel.equalsIgnoreCase("N/A")) {
                startEpochTimestamp = Long.parseLong(timestampLabel.trim());
            }
        } catch (NumberFormatException ignored) {}

        for (int i = 0; i < vmList.size(); i++) {
            Vm vm = vmList.get(i);
            long vmId = vm.getId();

            // Deterministic trace mapping: Maps active VMs to their actual Bitbrains fastStorage trace
            final String tracePath = resolveTracePathForVm(i, vm);
            final BitbrainsUtilizationModel cpuUtilizationModel =
                new BitbrainsUtilizationModel(tracePath, startEpochTimestamp);
            final long length = 50_000_000; // Sufficiently large to execute across the full observation window
            final int pes = 1;

            final Cloudlet cloudlet = new CloudletSimple(length, pes);
            cloudlet.setUtilizationModelCpu(cpuUtilizationModel);
            cloudlet.setUtilizationModelRam(ramUtilizationModel);
            cloudlet.setUtilizationModelBw(bwUtilizationModel);

            cloudletList.add(cloudlet);
            vmRequestedUtilizationMap.put(vmId, cpuUtilizationModel.getMeanRequestedUtilization(OBSERVATION_WINDOW_SECONDS));
        }

        return cloudletList;
    }

    private static String resolveTracePathForVm(int vmIndex, Vm vm) {
        String vmIdStr = "VM_" + String.format("%03d", vm.getId() + 1);
        return TraceResolver.resolveTracePath(vmIdStr);
    }

    private static void exportResults(String algorithmName, String epoch, String timestamp,
                                      Datacenter datacenter, List<Vm> vmList,
                                      double totalEnergyWh, int slaViolations, double avgCpuUtil, int activePmCount) {
        try {
            File resultsDir = resolveDirectory("../results", "results");
            if (!resultsDir.exists()) {
                resultsDir.mkdirs();
            }
            File f = new File(resultsDir, "simulation_results.csv");
            boolean isNew = !f.exists() || f.length() == 0;
            try (java.io.FileWriter fw = new java.io.FileWriter(f, true)) {
                if (isNew) {
                    fw.write("Algorithm,Epoch,Timestamp,VM_Count,Active_PM,Energy_Wh,SLA_Violations,Avg_CPU,Migrations\n");
                }
                fw.write(String.format(java.util.Locale.US, "%s,%s,%s,%d,%d,%.4f,%d,%.4f,0%n",
                    algorithmName, epoch, timestamp, vmList.size(), activePmCount, totalEnergyWh, slaViolations, avgCpuUtil));
            }
        } catch (IOException e) {
            System.err.println("Failed to write results: " + e.getMessage());
        }
    }

    private static void exportManifests(Datacenter datacenter, List<Vm> vmList) {
        try {
            File resultsDir = resolveDirectory("../results", "results");
            if (!resultsDir.exists()) {
                resultsDir.mkdirs();
            }
            try (java.io.FileWriter fw = new java.io.FileWriter(new File(resultsDir, "host_manifest.csv"))) {
                fw.write("host_id,pes,mips_per_pe,ram_mb,storage_mb,max_power_w,static_power_w\n");
                for (Host host : datacenter.getHostList()) {
                    fw.write(String.format(java.util.Locale.US, "%d,%d,%.0f,%d,%d,%.1f,%.1f%n",
                        host.getId(), host.getPesNumber(), (double) host.getPeList().get(0).getCapacity(),
                        host.getRam().getCapacity(), host.getStorage().getCapacity(),
                        ((PowerModelHostSimple) host.getPowerModel()).getMaxPower(),
                        ((PowerModelHostSimple) host.getPowerModel()).getStaticPower()));
                }
            }

            try (java.io.FileWriter fw = new java.io.FileWriter(new File(resultsDir, "vm_manifest.csv"))) {
                fw.write("vm_id,pes,mips,ram_mb,storage_mb\n");
                for (Vm vm : vmList) {
                    fw.write(String.format(java.util.Locale.US, "%d,%d,%.0f,%d,%d%n",
                        vm.getId(), vm.getPesNumber(), vm.getMips(),
                        vm.getRam().getCapacity(), vm.getStorage().getCapacity()));
                }
            }
        } catch (IOException e) {
            System.err.println("Failed to write manifests: " + e.getMessage());
        }
    }

    private static String resolvePlacementPath(String filename) {
        File f1 = new File(filename);
        if (f1.exists()) return f1.getPath();
        File f2 = new File("cloudsim", filename);
        if (f2.exists()) return f2.getPath();
        File f3 = new File("../cloudsim", filename);
        if (f3.exists()) return f3.getPath();
        return filename;
    }

    private static String resolveTracePath(String relPath) {
        File f1 = new File(relPath);
        if (f1.exists()) return f1.getPath();
        File f2 = new File("cloudsim", relPath);
        if (f2.exists()) return f2.getPath();
        File f3 = new File("../cloudsim", relPath);
        if (f3.exists()) return f3.getPath();
        return relPath;
    }

    private static File resolveDirectory(String path1, String path2) {
        File d1 = new File(path1);
        if (d1.exists() || d1.getParentFile() != null && d1.getParentFile().exists()) {
            return d1;
        }
        return new File(path2);
    }
}
