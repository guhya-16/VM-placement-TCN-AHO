package org.vmplacement;

import com.vmplacement.optimization.AdaptiveHippopotamusOptimization;
import com.vmplacement.optimization.CSVWorkloadReader;
import com.vmplacement.optimization.DecisionEpoch;
import com.vmplacement.optimization.DecisionEpochConverter;
import com.vmplacement.optimization.HippopotamusOptimization;
import com.vmplacement.optimization.PM;
import com.vmplacement.optimization.PlacementRepair;
import com.vmplacement.optimization.PlacementSolution;
import com.vmplacement.optimization.PlacementValidator;
import com.vmplacement.optimization.RiskState;
import com.vmplacement.optimization.VM;

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

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;

/**
 * =====================================================================================
 * RollingExperiment — Stateful Rolling Experiment Runner
 * =====================================================================================
 * Executes chronological multi-epoch simulations across PABFD, Standard HO, and WA-AHO.
 *
 * Core Guarantees:
 * 1. Stateful DatacenterState: Preserves VM->Host allocation across consecutive epochs.
 * 2. Warm-Start Optimizer Seeding: Previous epoch placement is injected as a starting
 *    candidate in the optimizer's initial population.
 * 3. Fast In-Loop Evaluation: Optimizer uses lightweight FitnessFunction for all 50
 *    iterations x 20 population candidates (NO CloudSim inside optimizer search).
 * 4. Single CloudSim Run Per Epoch: Real CloudSim Plus discrete-event simulation is
 *    executed EXACTLY ONCE per epoch on the winning placement.
 * 5. Deterministic Trace Identity: Replays fastStorage telemetry using actual VM identity
 *    (VM_001 -> 1.csv, VM_003 -> 3.csv, etc.) via TraceResolver.
 * 6. Migration Tracking: Inter-epoch VM placement changes are precisely computed against DatacenterState.
 * =====================================================================================
 */
public class RollingExperiment {

    public static final double OBSERVATION_WINDOW_SECONDS = 3600.0;
    public static final double SLA_TOLERANCE = 0.05;
    public static final double VOLATILITY_P95 = 4.675352;

    public static final int POPULATION_SIZE = 20;
    public static final int MAX_ITERATIONS = 50;

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

    public static class EpochResult {
        public final int epoch;
        public final long timestamp;
        public final String method;
        public final double energyWh;
        public final int slaViolations;
        public final int activePms;
        public final int placementChanges;
        public final double risk;
        public final double meanPredictedCpuUtilization;
        public final String explorationProbability;
        public final String adaptiveMode;
        public final String bestIteration;
        public final double avgCpu;
        public final double bestFitness;
        public final String placementStr;

        public EpochResult(int epoch, long timestamp, String method, double energyWh, int slaViolations,
                           int activePms, int placementChanges, double risk, double meanPredictedCpuUtilization,
                           String explorationProbability, String adaptiveMode, String bestIteration,
                           double avgCpu, double bestFitness, String placementStr) {
            this.epoch = epoch;
            this.timestamp = timestamp;
            this.method = method;
            this.energyWh = energyWh;
            this.slaViolations = slaViolations;
            this.activePms = activePms;
            this.placementChanges = placementChanges;
            this.risk = risk;
            this.meanPredictedCpuUtilization = meanPredictedCpuUtilization;
            this.explorationProbability = explorationProbability;
            this.adaptiveMode = adaptiveMode;
            this.bestIteration = bestIteration;
            this.avgCpu = avgCpu;
            this.bestFitness = bestFitness;
            this.placementStr = placementStr;
        }
    }

    public static class CloudSimResult {
        public final double energyWh;
        public final int slaViolations;
        public final int activePmCount;
        public final double avgCpu;

        public CloudSimResult(double energyWh, int slaViolations, int activePmCount, double avgCpu) {
            this.energyWh = energyWh;
            this.slaViolations = slaViolations;
            this.activePmCount = activePmCount;
            this.avgCpu = avgCpu;
        }
    }

    public static void main(String[] args) {
        // Suppress verbose SLF4J / Logback logging for clean terminal progress
        try {
            ch.qos.logback.classic.Logger root = (ch.qos.logback.classic.Logger) org.slf4j.LoggerFactory
                    .getLogger(org.slf4j.Logger.ROOT_LOGGER_NAME);
            root.setLevel(ch.qos.logback.classic.Level.WARN);
        } catch (Throwable ignored) {}

        int maxEpochs = 10; // Default to 10 for test run as requested in requirement 5
        if (args != null && args.length > 0) {
            try {
                if (args[0].equalsIgnoreCase("full") || args[0].equalsIgnoreCase("all")) {
                    maxEpochs = 997;
                } else {
                    maxEpochs = Integer.parseInt(args[0].trim());
                }
            } catch (NumberFormatException ignored) {}
        }

        System.out.println("==========================================================================");
        System.out.println("   STATEFUL ROLLING EXPERIMENT RUNNER — CLOUDSIM PLUS & AHO");
        System.out.println("   Methods: PABFD vs Standard HO vs WA-AHO (Workload-Aware Adaptive HO)");
        System.out.println("   Testing Epochs: " + maxEpochs);
        System.out.println("==========================================================================");

        File riskFile = resolveFile(
            "results/stage5/risk/risk_state.csv",
            "../results/stage5/risk/risk_state.csv",
            "person3-optimization-FINAL-v2/person3_optimization/input/risk_state.csv",
            "../person3-optimization-FINAL-v2/person3_optimization/input/risk_state.csv",
            "input/risk_state.csv"
        );

        System.out.println("Risk State Input File : " + riskFile.getPath());

        List<DecisionEpoch> allEpochs;
        try {
            allEpochs = CSVWorkloadReader.readDecisionEpochs(riskFile.getPath());
        } catch (IOException e) {
            System.err.println("Failed to read risk_state.csv: " + e.getMessage());
            return;
        }

        int totalAvailable = allEpochs.size();
        int epochsToRun = Math.min(maxEpochs, totalAvailable);
        List<DecisionEpoch> targetEpochs = allEpochs.subList(0, epochsToRun);

        System.out.printf("Loaded %d total epochs. Executing rolling run on first %d epochs.%n%n",
                totalAvailable, epochsToRun);

        List<PM> pms = createPhysicalMachines();

        // Prepare output file
        File csvOutputFile = resolveOutputFile("results/rolling_experiment_results.csv", "../results/rolling_experiment_results.csv");
        csvOutputFile.getParentFile().mkdirs();

        List<EpochResult> allResults = new ArrayList<>();

        String[] methods = { "PABFD", "Standard_HO", "WA-AHO" };

        for (String method : methods) {
            System.out.printf("--------------------------------------------------------------------------%n");
            System.out.printf(">>> STARTING ROLLING EXPERIMENT RUN: %s (%d Epochs)%n", method, epochsToRun);
            System.out.printf("--------------------------------------------------------------------------%n");

            DatacenterState datacenterState = new DatacenterState();
            long methodStart = System.currentTimeMillis();

            for (int i = 0; i < epochsToRun; i++) {
                int epochNum = i + 1;
                DecisionEpoch epoch = targetEpochs.get(i);
                long timestamp = epoch.getDecisionTimestamp();
                List<RiskState> riskStates = epoch.getRiskStates();
                List<VM> epochVMs = DecisionEpochConverter.toVMs(epoch);

                // Step 2a, 2b, 2c: Workload & Risk Calculations
                double volatilitySum = 0.0;
                double riskSum = 0.0;
                double predMeanSum = 0.0;
                for (RiskState r : riskStates) {
                    volatilitySum += r.getVolatilityScore();
                    riskSum += r.getRiskScore();
                    predMeanSum += r.getPredictedMeanCpu();
                }
                double workloadVar = clamp((volatilitySum / riskStates.size()) / VOLATILITY_P95, 0.0, 1.0);
                double predRisk = clamp(riskSum / riskStates.size(), 0.0, 1.0);
                double compositeRisk = 0.60 * workloadVar + 0.40 * predRisk;
                double avgPrediction = predMeanSum / riskStates.size(); // Arithmetic mean predicted CPU fraction

                // Step 2d: Optimization with warm-start from DatacenterState
                long epochSeed = 100_000L + i;
                PlacementSolution winningSolution;
                String explorationProbStr = "";
                String adaptiveModeStr = "";
                String bestIterationStr = "";
                double bestFitnessVal = 0.0;

                if (method.equals("PABFD")) {
                    winningSolution = computePabfdSolution(epochVMs, pms);
                    bestIterationStr = "";
                } else {
                    List<PlacementSolution> population = createInitialPopulationWithWarmStart(
                            epochVMs, pms, datacenterState, POPULATION_SIZE, epochSeed);

                    if (method.equals("Standard_HO")) {
                        HippopotamusOptimization standardHO = new HippopotamusOptimization(
                                epochVMs, pms, POPULATION_SIZE, MAX_ITERATIONS, epochSeed + 1, population, false);
                        winningSolution = standardHO.optimize();
                        bestIterationStr = String.valueOf(standardHO.getBestIteration());
                        bestFitnessVal = standardHO.getBestFitness();
                    } else { // WA-AHO
                        AdaptiveHippopotamusOptimization adaptiveHO = new AdaptiveHippopotamusOptimization(
                                epochVMs, pms, POPULATION_SIZE, MAX_ITERATIONS, epochSeed + 2, riskStates, population, false);
                        winningSolution = adaptiveHO.optimize();
                        explorationProbStr = String.format(Locale.US, "%.4f", adaptiveHO.getCurrentExplorationProbability());
                        adaptiveModeStr = adaptiveHO.getCurrentAdaptiveMode();
                        bestIterationStr = String.valueOf(adaptiveHO.getBestIteration());
                        bestFitnessVal = adaptiveHO.getBestFitness();
                    }
                }

                // Step 2f: Compare winning placement to PREVIOUS epoch placement to count placement changes
                int placementChanges = datacenterState.computePlacementChanges(epochVMs, winningSolution);

                // Step 2e: Run REAL CloudSim exactly once per epoch on winning placement
                Map<Long, Long> placementMap = new LinkedHashMap<>();
                int[] vmToPm = winningSolution.getVmToPm();
                for (int v = 0; v < vmToPm.length; v++) {
                    placementMap.put((long) v, (long) vmToPm[v]);
                }

                CloudSimResult simRes = runSingleCloudSimEpoch(epochNum, timestamp, method, epochVMs, placementMap);

                // Step 2g: Record Epoch Result
                EpochResult result = new EpochResult(
                        epochNum, timestamp, method, simRes.energyWh, simRes.slaViolations,
                        simRes.activePmCount, placementChanges, compositeRisk, avgPrediction,
                        explorationProbStr, adaptiveModeStr, bestIterationStr, simRes.avgCpu,
                        bestFitnessVal, Arrays.toString(winningSolution.getVmToPm()));
                allResults.add(result);

                // Step 2h: Update DatacenterState with winning placement before next epoch
                datacenterState.updateState(epochNum, timestamp, epochVMs, winningSolution);

                System.out.printf("  [Epoch %2d/%2d] %-11s | Energy: %7.2f Wh | SLA: %d | Active PMs: %2d | Changes: %d | Risk: %.4f | BestIter: %-2s | Map: %s%n",
                        epochNum, epochsToRun, method, simRes.energyWh, simRes.slaViolations,
                        simRes.activePmCount, placementChanges, compositeRisk,
                        bestIterationStr.isEmpty() ? "N/A" : bestIterationStr,
                        Arrays.toString(winningSolution.getVmToPm()));
            }

            long methodElapsed = System.currentTimeMillis() - methodStart;
            System.out.printf("Completed %s in %.2f seconds (%.2f ms/epoch).%n%n",
                    method, methodElapsed / 1000.0, (double) methodElapsed / epochsToRun);
        }

        // Export to CSV
        exportToCsv(csvOutputFile, allResults);
        System.out.println("==========================================================================");
        System.out.println("   RESULTS SAVED TO: " + csvOutputFile.getPath());
        System.out.println("==========================================================================");

        printSummaryTable(allResults, epochsToRun);
        printDetailedCheckpoints(allResults);
    }

    private static List<PlacementSolution> createInitialPopulationWithWarmStart(
            List<VM> vms, List<PM> pms, DatacenterState state, int size, long seed) {

        List<PlacementSolution> population = new ArrayList<>();
        Random random = new Random(seed);

        // Warm-start solution from previous epoch state (if present and feasible/repaired)
        if (!state.isEmpty()) {
            PlacementSolution warmStart = state.createWarmStartSolution(vms, pms, random);
            if (warmStart != null && PlacementValidator.isFeasible(vms, pms, warmStart)) {
                population.add(warmStart.copy());
            }
        }

        int attempts = 0;
        int maxAttempts = size * 200;

        while (population.size() < size && attempts < maxAttempts) {
            attempts++;
            PlacementSolution solution = PlacementRepair.createRandomFeasibleSolution(vms, pms, random, 0.65);
            if (solution == null || !PlacementValidator.isFeasible(vms, pms, solution)) {
                continue;
            }

            boolean duplicate = false;
            for (PlacementSolution existing : population) {
                if (Arrays.equals(existing.getVmToPm(), solution.getVmToPm())) {
                    duplicate = true;
                    break;
                }
            }

            if (!duplicate) {
                population.add(solution.copy());
            }
        }

        return population;
    }

    private static PlacementSolution computePabfdSolution(List<VM> vms, List<PM> pms) {
        int[] usedPes = new int[pms.size()];
        double[] usedRam = new double[pms.size()];
        double[] usedStorage = new double[pms.size()];
        double[] usedMips = new double[pms.size()];
        boolean[] active = new boolean[pms.size()];

        List<Integer> vmIndices = new ArrayList<>();
        for (int i = 0; i < vms.size(); i++) vmIndices.add(i);

        // Sort decreasing by expected compute demand (Decreasing part of PABFD)
        vmIndices.sort((a, b) -> {
            VM va = vms.get(a);
            VM vb = vms.get(b);
            return Double.compare(vb.getPes() * vb.getMips(), va.getPes() * va.getMips());
        });

        int[] placement = new int[vms.size()];

        for (int vmIdx : vmIndices) {
            VM vm = vms.get(vmIdx);
            int bestPm = -1;
            double minPowerDiff = Double.MAX_VALUE;

            for (int p = 0; p < pms.size(); p++) {
                PM pm = pms.get(p);
                double currentReqMips = vm.getMips() * vm.getCurrentCpuUtilization() / 100.0;

                if (usedPes[p] + vm.getPes() <= pm.getPes() &&
                    usedRam[p] + vm.getRamMb() <= pm.getRamMb() &&
                    usedStorage[p] + vm.getStorageMb() <= pm.getStorageMb() &&
                    usedMips[p] + currentReqMips <= pm.getTotalMips()) {

                    double currUtil = active[p] ? Math.min(1.0, usedMips[p] / pm.getTotalMips()) : 0.0;
                    double nextUtil = Math.min(1.0, (usedMips[p] + currentReqMips) / pm.getTotalMips());

                    double currPower = active[p] ? pm.getStaticPowerWatts() + (pm.getMaxPowerWatts() - pm.getStaticPowerWatts()) * currUtil : 0.0;
                    double nextPower = pm.getStaticPowerWatts() + (pm.getMaxPowerWatts() - pm.getStaticPowerWatts()) * nextUtil;
                    double powerDiff = nextPower - currPower;

                    if (powerDiff < minPowerDiff) {
                        minPowerDiff = powerDiff;
                        bestPm = p;
                    }
                }
            }

            if (bestPm == -1) {
                // Fallback to first fit
                for (int p = 0; p < pms.size(); p++) {
                    PM pm = pms.get(p);
                    if (usedPes[p] + vm.getPes() <= pm.getPes() &&
                        usedRam[p] + vm.getRamMb() <= pm.getRamMb()) {
                        bestPm = p;
                        break;
                    }
                }
                if (bestPm == -1) bestPm = 0;
            }

            placement[vmIdx] = bestPm;
            usedPes[bestPm] += vm.getPes();
            usedRam[bestPm] += vm.getRamMb();
            usedStorage[bestPm] += vm.getStorageMb();
            usedMips[bestPm] += (vm.getMips() * vm.getCurrentCpuUtilization() / 100.0);
            active[bestPm] = true;
        }

        return new PlacementSolution(placement);
    }

    private static CloudSimResult runSingleCloudSimEpoch(
            int epoch, long timestamp, String algorithm, List<VM> epochVMs, Map<Long, Long> placementMap) {

        final CloudSimPlus simulation = new CloudSimPlus();
        final List<Host> hostList = createDatacenterHosts();

        VmAllocationPolicyFromFile allocationPolicy = new VmAllocationPolicyFromFile(
                placementMap, algorithm + "_Ep" + epoch);
        allocationPolicy.validateWithDatacenterHosts(hostList);

        final Datacenter datacenter = new DatacenterSimple(simulation, hostList, allocationPolicy);
        datacenter.setSchedulingInterval(10);

        final DatacenterBroker broker = new DatacenterBrokerSimple(simulation);

        final List<Vm> vmList = new ArrayList<>();
        for (int i = 0; i < epochVMs.size(); i++) {
            VM spec = epochVMs.get(i);
            Vm vm = new VmSimple(i, spec.getMips(), spec.getPes());
            vm.setRam((long) spec.getRamMb()).setBw(1000).setSize((long) spec.getStorageMb());
            vm.setCloudletScheduler(new CloudletSchedulerTimeShared());
            vm.enableUtilizationStats();
            vmList.add(vm);
        }

        final List<Cloudlet> cloudletList = new ArrayList<>();
        final UtilizationModel ramModel = new UtilizationModelDynamic(0.3);
        final UtilizationModel bwModel = new UtilizationModelFull();
        final Map<Long, Double> requestedUtilMap = new LinkedHashMap<>();

        for (int i = 0; i < vmList.size(); i++) {
            Vm vm = vmList.get(i);
            VM spec = epochVMs.get(i);

            // FIX: Deterministic trace path using actual VM identity (VM_001 -> 1.csv, VM_003 -> 3.csv)
            String tracePath = TraceResolver.resolveTracePath(spec.getId());
            BitbrainsUtilizationModel cpuModel = new BitbrainsUtilizationModel(tracePath, timestamp);

            Cloudlet cloudlet = new CloudletSimple(50_000_000, 1);
            cloudlet.setVm(vm);
            cloudlet.setUtilizationModelCpu(cpuModel);
            cloudlet.setUtilizationModelRam(ramModel);
            cloudlet.setUtilizationModelBw(bwModel);

            cloudletList.add(cloudlet);
            requestedUtilMap.put(vm.getId(), cpuModel.getMeanRequestedUtilization(OBSERVATION_WINDOW_SECONDS));
        }

        broker.submitVmList(vmList);
        broker.submitCloudletList(cloudletList);

        simulation.terminateAt(OBSERVATION_WINDOW_SECONDS);
        simulation.start();

        // Calculate real simulation metrics
        int slaViolations = 0;
        for (Vm vm : vmList) {
            double delivered = vm.getCpuUtilizationStats().getMean();
            double requested = requestedUtilMap.getOrDefault(vm.getId(), 0.0);
            if ((requested - delivered) > SLA_TOLERANCE) {
                slaViolations++;
            }
        }

        int activePmCount = 0;
        double totalEnergyWh = 0.0;
        double sumActiveCpu = 0.0;
        final double windowHours = OBSERVATION_WINDOW_SECONDS / 3600.0;

        for (Host host : hostList) {
            double meanUtil = host.getCpuUtilizationStats().getMean();
            if (Double.isNaN(meanUtil)) {
                meanUtil = 0.0;
            }
            double power = host.getPowerModel().getPower(meanUtil);
            totalEnergyWh += power * windowHours;

            if (!host.getVmCreatedList().isEmpty()) {
                activePmCount++;
                sumActiveCpu += meanUtil;
            }
        }

        double avgCpu = activePmCount > 0 ? sumActiveCpu / activePmCount : 0.0;

        return new CloudSimResult(totalEnergyWh, slaViolations, activePmCount, avgCpu);
    }

    private static List<PM> createPhysicalMachines() {
        List<PM> pms = new ArrayList<>();
        for (int i = 1; i <= 10; i++) {
            pms.add(new PM(String.format("PM_%03d", i), PM1_PES, PM1_MIPS, PM1_RAM_MB, PM1_STORAGE_MB, PM1_MAX_POWER, PM1_STATIC_POWER));
        }
        for (int i = 11; i <= 16; i++) {
            pms.add(new PM(String.format("PM_%03d", i), PM2_PES, PM2_MIPS, PM2_RAM_MB, PM2_STORAGE_MB, PM2_MAX_POWER, PM2_STATIC_POWER));
        }
        for (int i = 17; i <= 20; i++) {
            pms.add(new PM(String.format("PM_%03d", i), PM3_PES, PM3_MIPS, PM3_RAM_MB, PM3_STORAGE_MB, PM3_MAX_POWER, PM3_STATIC_POWER));
        }
        return pms;
    }

    private static List<Host> createDatacenterHosts() {
        List<Host> hostList = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            hostList.add(createHost(PM1_PES, PM1_MIPS, PM1_RAM_MB, PM1_STORAGE_MB, PM1_MAX_POWER, PM1_STATIC_POWER));
        }
        for (int i = 0; i < 6; i++) {
            hostList.add(createHost(PM2_PES, PM2_MIPS, PM2_RAM_MB, PM2_STORAGE_MB, PM2_MAX_POWER, PM2_STATIC_POWER));
        }
        for (int i = 0; i < 4; i++) {
            hostList.add(createHost(PM3_PES, PM3_MIPS, PM3_RAM_MB, PM3_STORAGE_MB, PM3_MAX_POWER, PM3_STATIC_POWER));
        }
        return hostList;
    }

    private static Host createHost(int numPes, long mipsPerPe, long ramMB, long storageMB,
                                   double maxPowerWatts, double staticPowerWatts) {
        List<Pe> peList = new ArrayList<>();
        for (int i = 0; i < numPes; i++) {
            peList.add(new PeSimple(mipsPerPe));
        }
        Host host = new HostSimple(ramMB, HOST_BW, storageMB, peList);
        host.setVmScheduler(new VmSchedulerTimeShared());
        PowerModelHost powerModel = new PowerModelHostSimple(maxPowerWatts, staticPowerWatts);
        host.setPowerModel(powerModel);
        host.enableUtilizationStats();
        return host;
    }

    private static void exportToCsv(File primaryFile, List<EpochResult> results) {
        File[] candidateFiles = {
            primaryFile,
            new File("results/rolling_experiment_results.csv"),
            new File("../results/rolling_experiment_results.csv")
        };

        for (File f : candidateFiles) {
            try {
                if (f.getParentFile() != null) {
                    f.getParentFile().mkdirs();
                }
                try (PrintWriter pw = new PrintWriter(new FileWriter(f))) {
                    pw.println("epoch,timestamp,method,energy_wh,sla_violations,active_pms,placement_changes,risk,mean_predicted_cpu_utilization,exploration_probability,adaptive_mode,best_iteration");
                    for (EpochResult r : results) {
                        pw.printf(Locale.US, "%d,%d,%s,%.4f,%d,%d,%d,%.4f,%.4f,%s,%s,%s%n",
                                r.epoch, r.timestamp, r.method, r.energyWh, r.slaViolations,
                                r.activePms, r.placementChanges, r.risk, r.meanPredictedCpuUtilization,
                                r.explorationProbability, r.adaptiveMode, r.bestIteration);
                    }
                }
            } catch (IOException ignored) {}
        }
    }

    private static void printSummaryTable(List<EpochResult> results, int epochCount) {
        System.out.println("\n==========================================================================");
        System.out.printf("   AGGREGATE COMPARISON ACROSS %d ROLLING EPOCHS%n", epochCount);
        System.out.println("==========================================================================");
        System.out.printf("%-15s | %-14s | %-12s | %-12s | %-12s%n",
                "Method", "Total Energy", "Mean Energy", "Total SLA", "Total Changes");
        System.out.println("--------------------------------------------------------------------------");

        String[] methods = { "PABFD", "Standard_HO", "WA-AHO" };
        for (String m : methods) {
            double totalEnergy = 0.0;
            int totalSla = 0;
            int totalChanges = 0;
            int count = 0;

            for (EpochResult r : results) {
                if (r.method.equals(m)) {
                    totalEnergy += r.energyWh;
                    totalSla += r.slaViolations;
                    totalChanges += r.placementChanges;
                    count++;
                }
            }

            double meanEnergy = count > 0 ? totalEnergy / count : 0.0;
            System.out.printf(Locale.US, "%-15s | %10.2f Wh | %9.2f Wh | %12d | %12d%n",
                    m, totalEnergy, meanEnergy, totalSla, totalChanges);
        }
        System.out.println("==========================================================================\n");
    }

    private static void printDetailedCheckpoints(List<EpochResult> results) {
        System.out.println("==========================================================================");
        System.out.println("   DETAILED CHECKPOINTS: EPOCH 1 & EPOCH 8 COMPARISON");
        System.out.println("==========================================================================");
        for (int targetEpoch : new int[]{1, 8}) {
            System.out.printf("--- EPOCH %d ---%n", targetEpoch);
            for (EpochResult r : results) {
                if (r.epoch == targetEpoch) {
                    System.out.printf("  %-11s | Placement: %-22s | Active PMs: %d | Energy: %.2f Wh | BestIter: %s | Fitness: %.6f%n",
                            r.method, r.placementStr, r.activePms, r.energyWh,
                            r.bestIteration.isEmpty() ? "N/A" : r.bestIteration, r.bestFitness);
                }
            }
            System.out.println();
        }
        System.out.println("==========================================================================\n");
    }

    private static double clamp(double value, double min, double max) {
        return Math.max(min, Math.min(max, value));
    }

    private static File resolveFile(String... paths) {
        for (String p : paths) {
            File f = new File(p);
            if (f.exists() && f.isFile()) {
                return f;
            }
        }
        return new File(paths[0]);
    }

    private static File resolveOutputFile(String path1, String path2) {
        File d1 = new File(path1).getParentFile();
        if (d1 != null && (d1.exists() || new File(".").getAbsolutePath().contains("cloudsim"))) {
            return new File(path1);
        }
        return new File(path2);
    }
}
