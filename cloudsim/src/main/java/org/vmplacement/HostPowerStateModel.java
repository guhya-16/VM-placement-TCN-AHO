package org.vmplacement;

/**
 * Defines the power accounting model for physical hosts in the datacenter.
 */
public enum HostPowerStateModel {
    /**
     * Facility-level always-on model:
     * All physical servers in the facility remain powered on.
     * Active hosts draw P(u) = P_static + (P_max - P_static) * utilization.
     * Inactive hosts with 0 allocated VMs draw idle static power P(0) = P_static.
     */
    FACILITY_ALWAYS_ON,

    /**
     * Power-state-aware server consolidation model:
     * Active hosts (with >= 1 allocated VM) draw P(u) = P_static + (P_max - P_static) * utilization.
     * Inactive hosts (with 0 allocated VMs) are powered off / placed in deep sleep, drawing standby power (default: 0.0 W).
     */
    ACTIVE_SERVERS_ONLY
}
