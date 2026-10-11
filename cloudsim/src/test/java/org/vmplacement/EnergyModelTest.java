package org.vmplacement;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class EnergyModelTest {

    private DatacenterState datacenterState;

    @BeforeEach
    void setUp() {
        datacenterState = new DatacenterState();
    }

    @Test
    @DisplayName("Verify Wh units: 1 hour observation window converts Watts directly to Watt-hours")
    void testEnergyUnitsAreWh() {
        List<VmPlacementSpec> vms = new ArrayList<>();
        // Place 1 VM on PM 0 (PM1: static 93.7W, max 135.0W)
        vms.add(new VmPlacementSpec("VM_001", 0, 0.0, 0.0, null, 1, 500, 512, 40 * 1024));

        PlacementRequest request = new PlacementRequest(
                1, 1, "CAND_UNITS", 1378606200L, "TEST", 0.05,
                vms, null, HostPowerStateModel.ACTIVE_SERVERS_ONLY, 0.0
        );

        PlacementEvaluationResult result = PlacementEvaluator.evaluate(request, datacenterState);
        assertTrue(result.isFeasible());

        // For PM1 at 0 utilization over 1 hour (3600s), energy should be exactly 93.7 Wh (+/- 0.5)
        assertEquals(1, result.getActivePms());
        assertTrue(result.getEnergyWh() >= 93.6 && result.getEnergyWh() <= 94.0,
                "Expected PM1 idle energy ~93.7 Wh, got: " + result.getEnergyWh());
    }

    @Test
    @DisplayName("Verify Facility-Always-On model: all 20 hosts consume static idle power totalling ~1424.4 Wh")
    void testFacilityAlwaysOnStaticBaseline() {
        List<VmPlacementSpec> vms = new ArrayList<>();
        // Place 1 VM on PM 0
        vms.add(new VmPlacementSpec("VM_001", 0, 0.0, 0.0, null, 1, 500, 512, 40 * 1024));

        PlacementRequest request = new PlacementRequest(
                1, 1, "CAND_FACILITY", 1378606200L, "TEST", 0.05,
                vms, null, HostPowerStateModel.FACILITY_ALWAYS_ON, 0.0
        );

        PlacementEvaluationResult result = PlacementEvaluator.evaluate(request, datacenterState);
        assertTrue(result.isFeasible());

        // 10*93.7 + 6*42.3 + 4*58.4 = 937.0 + 253.8 + 233.6 = 1424.4 Wh
        assertTrue(result.getEnergyWh() >= 1424.0 && result.getEnergyWh() <= 1425.5,
                "Facility always-on must include all 20 hosts baseline (~1424.4 Wh), got: " + result.getEnergyWh());
    }

    @Test
    @DisplayName("Verify Power-State-Aware model: powered-off hosts consume 0.0 W, active hosts consume P(u)")
    void testActiveServersOnlyModel() {
        List<VmPlacementSpec> vms = new ArrayList<>();
        // Place 2 VMs on 2 distinct PM3 hosts (Hosts 16 and 17, PM3 static 58.4W each)
        vms.add(new VmPlacementSpec("VM_001", 16, 0.0, 0.0, null, 2, 1000, 1024, 40 * 1024));
        vms.add(new VmPlacementSpec("VM_002", 17, 0.0, 0.0, null, 2, 1000, 1024, 40 * 1024));

        PlacementRequest request = new PlacementRequest(
                1, 1, "CAND_ACTIVE", 1378606200L, "TEST", 0.05,
                vms, null, HostPowerStateModel.ACTIVE_SERVERS_ONLY, 0.0
        );

        PlacementEvaluationResult result = PlacementEvaluator.evaluate(request, datacenterState);
        assertTrue(result.isFeasible());
        assertEquals(2, result.getActivePms());

        // 2 * 58.4 W = 116.8 Wh (the other 18 hosts consume 0.0 W)
        assertTrue(result.getEnergyWh() >= 116.0 && result.getEnergyWh() <= 118.0,
                "Expected ~116.8 Wh for 2 PM3 hosts with remaining 18 powered off, got: " + result.getEnergyWh());
    }

    @Test
    @DisplayName("Verify dynamic power scaling: higher CPU utilization increases energy consumed")
    void testDynamicPowerIncreasesWithUtilization() {
        List<VmPlacementSpec> vmsLow = new ArrayList<>();
        vmsLow.add(new VmPlacementSpec("VM_001", 0, 10.0, 10.0, null, 1, 500, 512, 40 * 1024));

        List<VmPlacementSpec> vmsHigh = new ArrayList<>();
        vmsHigh.add(new VmPlacementSpec("VM_001", 0, 90.0, 90.0, null, 1, 500, 512, 40 * 1024));

        PlacementRequest reqLow = new PlacementRequest(
                1, 1, "LOW_UTIL", 1378606200L, "TEST", 0.05,
                vmsLow, null, HostPowerStateModel.ACTIVE_SERVERS_ONLY, 0.0
        );
        PlacementRequest reqHigh = new PlacementRequest(
                1, 2, "HIGH_UTIL", 1378606200L, "TEST", 0.05,
                vmsHigh, null, HostPowerStateModel.ACTIVE_SERVERS_ONLY, 0.0
        );

        PlacementEvaluationResult resLow = PlacementEvaluator.evaluate(reqLow, datacenterState);
        PlacementEvaluationResult resHigh = PlacementEvaluator.evaluate(reqHigh, datacenterState);

        assertTrue(resHigh.getEnergyWh() >= resLow.getEnergyWh(),
                "Higher utilization must consume more or equal energy than lower utilization");
    }

    @Test
    @DisplayName("Verify Consolidation Benefit: Consolidating 4 VMs to 2 PM3 hosts saves energy under ACTIVE_SERVERS_ONLY")
    void testConsolidationSavesEnergyUnderPowerStateAwareModel() {
        // Dispersed placement: 4 VMs spread over 4 separate PM2 hosts (Hosts 10, 11, 12, 13)
        // 4 * 42.3W = 169.2 Wh static
        List<VmPlacementSpec> dispersed = new ArrayList<>();
        dispersed.add(new VmPlacementSpec("VM_001", 10, 20.0, 20.0, null, 1, 500, 512, 40 * 1024));
        dispersed.add(new VmPlacementSpec("VM_002", 11, 20.0, 20.0, null, 1, 500, 512, 40 * 1024));
        dispersed.add(new VmPlacementSpec("VM_003", 12, 20.0, 20.0, null, 1, 500, 512, 40 * 1024));
        dispersed.add(new VmPlacementSpec("VM_004", 13, 20.0, 20.0, null, 1, 500, 512, 40 * 1024));

        // Consolidated placement: 4 VMs packed onto 2 PM3 hosts (Hosts 16, 17)
        // 2 * 58.4W = 116.8 Wh static
        List<VmPlacementSpec> consolidated = new ArrayList<>();
        consolidated.add(new VmPlacementSpec("VM_001", 16, 20.0, 20.0, null, 1, 500, 512, 40 * 1024));
        consolidated.add(new VmPlacementSpec("VM_002", 16, 20.0, 20.0, null, 1, 500, 512, 40 * 1024));
        consolidated.add(new VmPlacementSpec("VM_003", 17, 20.0, 20.0, null, 1, 500, 512, 40 * 1024));
        consolidated.add(new VmPlacementSpec("VM_004", 17, 20.0, 20.0, null, 1, 500, 512, 40 * 1024));

        PlacementRequest reqDispersed = new PlacementRequest(
                1, 1, "DISPERSED", 1378606200L, "TEST", 0.05,
                dispersed, null, HostPowerStateModel.ACTIVE_SERVERS_ONLY, 0.0
        );
        PlacementRequest reqConsolidated = new PlacementRequest(
                1, 2, "CONSOLIDATED", 1378606200L, "TEST", 0.05,
                consolidated, null, HostPowerStateModel.ACTIVE_SERVERS_ONLY, 0.0
        );

        PlacementEvaluationResult resDispersed = PlacementEvaluator.evaluate(reqDispersed, datacenterState);
        PlacementEvaluationResult resConsolidated = PlacementEvaluator.evaluate(reqConsolidated, datacenterState);

        assertEquals(4, resDispersed.getActivePms());
        assertEquals(2, resConsolidated.getActivePms());

        double energySavings = resDispersed.getEnergyWh() - resConsolidated.getEnergyWh();
        assertTrue(energySavings > 40.0,
                "Consolidation from 4 PM2 hosts to 2 PM3 hosts must save >40 Wh, saved: " + energySavings + " Wh");
    }
}
