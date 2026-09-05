package org.vmplacement;

import org.cloudsimplus.allocationpolicies.VmAllocationPolicyAbstract;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.vms.Vm;

import java.util.Comparator;
import java.util.Optional;

/**
 * Power-Aware Best Fit Decreasing (PABFD) VM allocation policy.
 *
 * For a VM being placed, among all hosts with enough free capacity to fit it,
 * this picks the one that causes the SMALLEST increase in power consumption -
 * tending to consolidate VMs onto fewer, more efficiently-loaded hosts,
 * rather than spreading them across many lightly-loaded ones.
 *
 * Note: sorting VMs by decreasing utilization (the "Decreasing" part of
 * PABFD) happens once, outside this class, wherever you build/submit your
 * VM list to the broker - since findHostForVm is called once per VM in
 * whatever order they're submitted.
 */
public class VmAllocationPolicyPabfd extends VmAllocationPolicyAbstract {

    @Override
    protected Optional<Host> defaultFindHostForVm(Vm vm) {
        return getHostList().stream()
            .filter(host -> host.isSuitableForVm(vm))
            .min(Comparator.comparingDouble(host -> powerIncreaseIfPlaced(host, vm)));
    }

    /**
     * Estimates how much a host's power draw would increase if this VM
     * were placed on it: power at (current utilization + VM's utilization)
     * minus power at current utilization.
     */
    private double powerIncreaseIfPlaced(Host host, Vm vm) {
        final double currentUtilization = host.getCpuPercentUtilization();
        final double vmUtilizationShare = vm.getExpectedHostCpuUtilization(currentUtilization);

        final double projectedUtilization = Math.min(1.0, currentUtilization + vmUtilizationShare);

        final double currentPower = host.getPowerModel().getPower(currentUtilization);
        final double projectedPower = host.getPowerModel().getPower(projectedUtilization);

        return projectedPower - currentPower;
    }
}
