package org.vmplacement;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

public class PlacementEvaluatorTest {

    private DatacenterState datacenterState;

    @BeforeEach
    void setUp() {
        datacenterState = new DatacenterState();
    }

    @Test
    @DisplayName("Test 1: Single Valid Candidate Evaluation returns real physical CloudSim metrics")
    void testSingleValidCandidateEvaluation() {
        // Build 5 VM placement specs (heterogeneous VM specs mapped to suitable PMs)
        // PM 0-9: PM1 (2 PEs, 4GB RAM)
        // PM 10-15: PM2 (4 PEs, 8GB RAM)
        // PM 16-19: PM3 (12 PEs, 16GB RAM)
        List<VmPlacementSpec> vms = new ArrayList<>();
        vms.add(new VmPlacementSpec("VM_001", 0, 20.0, 25.0, null, 1, 500, 512, 40 * 1024));
        vms.add(new VmPlacementSpec("VM_003", 1, 40.0, 45.0, null, 2, 1000, 1024, 60 * 1024));
        vms.add(new VmPlacementSpec("VM_011", 10, 30.0, 35.0, null, 3, 1500, 2048, 80 * 1024));
        vms.add(new VmPlacementSpec("VM_012", 11, 50.0, 55.0, null, 4, 2000, 3072, 100 * 1024));
        vms.add(new VmPlacementSpec("VM_025", 16, 25.0, 30.0, null, 2, 1000, 1024, 60 * 1024));

        long timestamp = 1378606200L; // Example Bitbrains epoch timestamp
        PlacementRequest request = new PlacementRequest(
                1, 1, "CANDIDATE_TEST_001", timestamp,
                "WA-AHO", 0.05, vms, null
        );

        PlacementEvaluationResult result = PlacementEvaluator.evaluate(request, datacenterState);

        assertNotNull(result, "Evaluation result should not be null");
        assertTrue(result.isFeasible(), "Valid placement candidate must be feasible");
        assertEquals(1, result.getEpoch());
        assertEquals(1, result.getIteration());
        assertEquals("CANDIDATE_TEST_001", result.getCandidateId());

        // Physical CloudSim metric validations
        assertTrue(result.getEnergyWh() > 0.0, "Total energy must be strictly positive: " + result.getEnergyWh());
        assertEquals(5, result.getActivePms(), "5 distinct hosts should be active");
        assertTrue(result.getMeanCpuUtilization() >= 0.0 && result.getMeanCpuUtilization() <= 1.0,
                "Mean CPU utilization should be within [0, 1]: " + result.getMeanCpuUtilization());
        assertEquals(0, result.getPlacementChangesFromPreviousState(), "Initial epoch should have 0 placement changes");
        assertEquals(0.0, result.getMigrationCost(), "Initial epoch migration cost should be 0.0");
        assertFalse(result.getHostCpuUtilizationMap().isEmpty(), "Host CPU utilization map should not be empty");

        System.out.println("Valid Candidate Test Result: " + result);
    }

    @Test
    @DisplayName("Test 2: Infeasible Candidate Evaluation returns feasible=false and rejects without modification")
    void testInfeasibleCandidateEvaluation() {
        // Attempt to pack 5 large VMs (4 PEs each = 20 PEs total) onto PM 0 (PM1 has only 2 PEs)
        List<VmPlacementSpec> vms = new ArrayList<>();
        vms.add(new VmPlacementSpec("VM_001", 0, 80.0, 80.0, null, 4, 2000, 4096, 50 * 1024));
        vms.add(new VmPlacementSpec("VM_003", 0, 80.0, 80.0, null, 4, 2000, 4096, 50 * 1024));
        vms.add(new VmPlacementSpec("VM_011", 0, 80.0, 80.0, null, 4, 2000, 4096, 50 * 1024));

        PlacementRequest request = new PlacementRequest(
                1, 1, "INFEASIBLE_CANDIDATE", 1378606200L,
                "STANDARD_HO", 0.05, vms, null
        );

        PlacementEvaluationResult result = PlacementEvaluator.evaluate(request, datacenterState);

        assertNotNull(result);
        assertFalse(result.isFeasible(), "Over-allocated candidate must be marked infeasible");
        assertEquals(Double.POSITIVE_INFINITY, result.getEnergyWh(), "Infeasible energy should be POSITIVE_INFINITY");
        assertTrue(result.getMessage().contains("capacity exceeded"), "Error message should report capacity violation: " + result.getMessage());

        System.out.println("Infeasible Candidate Test Result: " + result);
    }

    @Test
    @DisplayName("Test 3: State Management preserves previous final state across candidate evaluations until committed")
    void testStateManagementAndPlacementChanges() {
        // Epoch 1 Winning Placement: VM_001 (1 PE)->PM 0, VM_003 (2 PEs)->PM 1, VM_011 (3 PEs)->PM 10
        Map<String, Integer> epoch1Winning = new LinkedHashMap<>();
        epoch1Winning.put("VM_001", 0);
        epoch1Winning.put("VM_003", 1);
        epoch1Winning.put("VM_011", 10);

        // Evaluate candidate for Epoch 1
        PlacementRequest epoch1Candidate = new PlacementRequest(1, 1, "EP1_CANDIDATE_1", 1378606200L, "WA-AHO", epoch1Winning);
        PlacementEvaluationResult ep1Result = PlacementEvaluator.evaluate(epoch1Candidate, datacenterState);

        assertTrue(ep1Result.isFeasible(), "Epoch 1 candidate must be feasible: " + ep1Result.getMessage());
        assertEquals(0, ep1Result.getPlacementChangesFromPreviousState(), "Epoch 1 has no previous state -> 0 changes");
        assertTrue(datacenterState.isEmpty(), "Evaluating candidate MUST NOT modify DatacenterState baseline");

        // Commit winning placement for Epoch 1
        datacenterState.commitFinalPlacement(1, 1378606200L, epoch1Winning);
        assertFalse(datacenterState.isEmpty());
        assertEquals(3, datacenterState.size());
        assertEquals(1, datacenterState.getCurrentEpoch());

        // Epoch 2 Candidate A: Move VM_003 to PM 2 (1 migration), VM_011 to PM 11 (1 migration) -> 2 total changes
        Map<String, Integer> epoch2CandidateA = new LinkedHashMap<>();
        epoch2CandidateA.put("VM_001", 0);  // unchanged
        epoch2CandidateA.put("VM_003", 2);  // changed: 1 -> 2 (2 PEs on PM 2 which has 2 PEs)
        epoch2CandidateA.put("VM_011", 11); // changed: 10 -> 11 (3 PEs on PM 11 which has 4 PEs)

        PlacementRequest ep2ReqA = new PlacementRequest(2, 1, "EP2_CANDIDATE_A", 1378606500L, "WA-AHO", epoch2CandidateA);
        PlacementEvaluationResult ep2ResultA = PlacementEvaluator.evaluate(ep2ReqA, datacenterState);

        assertTrue(ep2ResultA.isFeasible(), "Epoch 2 candidate A must be feasible: " + ep2ResultA.getMessage());
        assertEquals(2, ep2ResultA.getPlacementChangesFromPreviousState(), "Should detect exactly 2 placement changes");
        assertEquals(2.0, ep2ResultA.getMigrationCost(), "Migration cost should equal 2.0");

        // Verify that evaluating candidate A did NOT overwrite the baseline in datacenterState
        assertEquals(1, datacenterState.getHostForVm("VM_003"), "VM_003 baseline in state must still be PM 1");
        assertEquals(10, datacenterState.getHostForVm("VM_011"), "VM_011 baseline in state must still be PM 10");

        // Epoch 2 Candidate B: Keep all unchanged (0 migrations)
        PlacementRequest ep2ReqB = new PlacementRequest(2, 2, "EP2_CANDIDATE_B", 1378606500L, "WA-AHO", epoch1Winning);
        PlacementEvaluationResult ep2ResultB = PlacementEvaluator.evaluate(ep2ReqB, datacenterState);

        assertTrue(ep2ResultB.isFeasible(), "Epoch 2 candidate B must be feasible");
        assertEquals(0, ep2ResultB.getPlacementChangesFromPreviousState(), "Identical placement has 0 placement changes");

        // Commit Candidate A as the winning placement for Epoch 2
        datacenterState.commitFinalPlacement(2, 1378606500L, epoch2CandidateA);
        assertEquals(2, datacenterState.getCurrentEpoch());
        assertEquals(2, datacenterState.getHostForVm("VM_003"), "VM_003 committed state is now PM 2");
        assertEquals(11, datacenterState.getHostForVm("VM_011"), "VM_011 committed state is now PM 11");

        System.out.println("State Management Test successfully verified isolation and commit mechanics.");
    }
}
