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
import org.cloudsimplus.utilizationmodels.UtilizationModel;
import org.cloudsimplus.utilizationmodels.UtilizationModelDynamic;
import org.cloudsimplus.utilizationmodels.UtilizationModelFull;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmSimple;

import java.util.ArrayList;
import java.util.List;

/**
 * Scaled simulation for the VM Placement project with PABFD policy
 * and real multi-trace Bitbrains workloads.
 */
public class Simulation {

    private static final int VM_COUNT = 24;

    // PM specs from the project's PM/VM configuration table
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

    public static void main(String[] args) {
        final CloudSimPlus simulation = new CloudSimPlus();
        final Datacenter datacenter = createDatacenter(simulation);
        final DatacenterBroker broker = new DatacenterBrokerSimple(simulation);

        final List<Vm> vmList = createVms(VM_COUNT);
        final List<Cloudlet> cloudletList = createCloudlets(VM_COUNT);

        // Sort VMs by descending expected utilization (Decreasing part of PABFD)
        vmList.sort((v1, v2) -> Double.compare(
            v2.getExpectedHostCpuUtilization(0),
            v1.getExpectedHostCpuUtilization(0)
        ));

        broker.submitVmList(vmList);
        broker.submitCloudletList(cloudletList);

        simulation.terminateAt(3600); // 1 simulated hour observation window
        simulation.start();

        System.out.println("\n========== UTILIZATION VERIFICATION ==========");
        for (int i = 0; i < Math.min(4, vmList.size()); i++) {
            Vm vm = vmList.get(i);
            System.out.printf("VM %d | CPU Utilization Mean: %.4f | Max: %.4f | Min: %.4f%n",
                vm.getId(),
                vm.getCpuUtilizationStats().getMean(),
                vm.getCpuUtilizationStats().getMax(),
                vm.getCpuUtilizationStats().getMin());
        }

        final List<Cloudlet> finishedCloudlets = broker.getCloudletFinishedList();
        System.out.println("\n========== SIMULATION RESULTS ==========");
        finishedCloudlets.forEach(c ->
            System.out.printf(
                "Cloudlet %2d | VM %2d | Status %s | Start %6.2f | Finish %6.2f%n",
                c.getId(), c.getVm().getId(), c.getStatus(),
                c.getStartTime(), c.getFinishTime()
            )
        );

        final long activeHostsCount = datacenter.getHostList().stream()
            .filter(host -> !host.getVmCreatedList().isEmpty())
            .count();

        System.out.printf("\nActive (utilized) hosts during simulation: %d / %d%n",
            activeHostsCount, datacenter.getHostList().size());

        System.out.println("\n========== POWER & ENERGY CONSUMPTION ==========");
        double totalEnergyWh = 0;
        double totalSimTimeHours = simulation.clock() / 3600.0;

        for (Host host : datacenter.getHostList()) {
            double meanUtilization = host.getCpuUtilizationStats().getMean();
            double meanPowerWatts = host.getPowerModel().getPower(meanUtilization);
            double hostEnergyWh = meanPowerWatts * totalSimTimeHours;
            int vmsPlaced = host.getVmCreatedList().size();

            System.out.printf("Host %2d | VMs: %2d | Mean Utilization: %5.2f%% | Mean Power: %6.2f W | Energy: %7.4f Wh%n",
                host.getId(), vmsPlaced, meanUtilization * 100, meanPowerWatts, hostEnergyWh);

            totalEnergyWh += hostEnergyWh;
        }
        System.out.printf("\nTotal energy consumed: %.4f Wh%n", totalEnergyWh);
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

        final Datacenter datacenter = new DatacenterSimple(simulation, hostList, new VmAllocationPolicyPabfd());
        datacenter.setSchedulingInterval(10); // sample every 10 simulated seconds
        return datacenter;
    }

    private static Host createHost(int numPes, long mipsPerPe, long ramMB, long storageMB,
                                    double maxPowerWatts, double staticPowerWatts) {
        final List<Pe> peList = new ArrayList<>();
        for (int i = 0; i < numPes; i++) {
            peList.add(new PeSimple(mipsPerPe));
        }
        final Host host = new HostSimple(ramMB, HOST_BW, storageMB, peList);

        final PowerModelHost powerModel = new PowerModelHostSimple(maxPowerWatts, staticPowerWatts);
        host.setPowerModel(powerModel);
        host.enableUtilizationStats();

        return host;
    }

    private static List<Vm> createVms(int count) {
        final List<Vm> vmList = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            int type = (i % 4) + 1;
            Vm vm = switch (type) {
                case 1 -> createVm(1, 500, 512, 40 * 1024);
                case 2 -> createVm(2, 1000, 1024, 60 * 1024);
                case 3 -> createVm(3, 1500, 2048, 80 * 1024);
                default -> createVm(4, 2000, 3072, 100 * 1024);
            };
            vmList.add(vm);
        }
        return vmList;
    }

    private static Vm createVm(int pes, long mips, long ramMB, long storageMB) {
        final Vm vm = new VmSimple(mips, pes);
        vm.setRam(ramMB).setBw(1000).setSize(storageMB);
        vm.setCloudletScheduler(new CloudletSchedulerTimeShared());
        vm.enableUtilizationStats();
        return vm;
    }

    private static List<Cloudlet> createCloudlets(int vmCount) {
        final List<Cloudlet> cloudletList = new ArrayList<>();
        final UtilizationModel ramUtilizationModel = new UtilizationModelDynamic(0.3);
        final UtilizationModel bwUtilizationModel = new UtilizationModelFull();

        for (int i = 0; i < vmCount; i++) {
            final String tracePath = "data/vm" + ((i % 4) + 1) + ".csv";
            final UtilizationModel cpuUtilizationModel = new BitbrainsUtilizationModel(tracePath);
            final long length = 50_000_000; // deliberately huge so cloudlets run throughout observation window
            final int pes = 1;

            final Cloudlet cloudlet = new CloudletSimple(length, pes);
            cloudlet.setUtilizationModelCpu(cpuUtilizationModel);
            cloudlet.setUtilizationModelRam(ramUtilizationModel);
            cloudlet.setUtilizationModelBw(bwUtilizationModel);

            cloudletList.add(cloudlet);
        }

        return cloudletList;
    }
}
