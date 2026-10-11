package org.vmplacement;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class CpuInputValidationTest {

    private static final double EPSILON = 1e-9;

    @Test
    @DisplayName("Valid CPU percentages convert correctly to demand fractions, preserving zero")
    void testValidCpuPercentages() {
        assertEquals(0.0, PlacementEvaluator.toDemandFraction(0.0), EPSILON, "0.0% must yield exactly 0.0 fraction");
        assertEquals(0.0075, PlacementEvaluator.toDemandFraction(0.75), EPSILON, "0.75% must yield 0.0075 fraction");
        assertEquals(0.50, PlacementEvaluator.toDemandFraction(50.0), EPSILON, "50.0% must yield 0.50 fraction");
        assertEquals(1.0, PlacementEvaluator.toDemandFraction(100.0), EPSILON, "100.0% must yield 1.0 fraction");
    }

    @ParameterizedTest
    @ValueSource(doubles = {-0.01, -1.0, -10.0, -100.0})
    @DisplayName("Negative CPU percentages throw IllegalArgumentException")
    void testNegativeCpuThrows(double negativeValue) {
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
                () -> PlacementEvaluator.toDemandFraction(negativeValue),
                "Negative CPU " + negativeValue + " must throw IllegalArgumentException");
        assertTrue(ex.getMessage().contains("Invalid CPU percentage"), "Exception message should indicate invalid CPU percentage");
    }

    @ParameterizedTest
    @ValueSource(doubles = {100.01, 105.0, 150.0, 200.0, 1000.0})
    @DisplayName("CPU percentages above 100 throw IllegalArgumentException")
    void testAboveHundredCpuThrows(double aboveHundred) {
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
                () -> PlacementEvaluator.toDemandFraction(aboveHundred),
                "CPU above 100 (" + aboveHundred + ") must throw IllegalArgumentException");
        assertTrue(ex.getMessage().contains("Invalid CPU percentage"), "Exception message should indicate invalid CPU percentage");
    }

    @Test
    @DisplayName("NaN CPU percentage throws IllegalArgumentException")
    void testNanCpuThrows() {
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
                () -> PlacementEvaluator.toDemandFraction(Double.NaN),
                "Double.NaN must throw IllegalArgumentException");
        assertTrue(ex.getMessage().contains("Invalid CPU percentage"));
    }

    @Test
    @DisplayName("Positive infinity CPU percentage throws IllegalArgumentException")
    void testPositiveInfinityCpuThrows() {
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
                () -> PlacementEvaluator.toDemandFraction(Double.POSITIVE_INFINITY),
                "Double.POSITIVE_INFINITY must throw IllegalArgumentException");
        assertTrue(ex.getMessage().contains("Invalid CPU percentage"));
    }

    @Test
    @DisplayName("Negative infinity CPU percentage throws IllegalArgumentException")
    void testNegativeInfinityCpuThrows() {
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
                () -> PlacementEvaluator.toDemandFraction(Double.NEGATIVE_INFINITY),
                "Double.NEGATIVE_INFINITY must throw IllegalArgumentException");
        assertTrue(ex.getMessage().contains("Invalid CPU percentage"));
    }

    @ParameterizedTest
    @ValueSource(doubles = {-5.0, 120.0, Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY})
    @DisplayName("PlacementEvaluator.evaluate marks candidate infeasible for invalid CPU utilization")
    void testPlacementEvaluatorInfeasibleOnInvalidCpu(double invalidCpu) {
        List<VmPlacementSpec> vms = new ArrayList<>();
        vms.add(new VmPlacementSpec("VM_001", 0, invalidCpu, invalidCpu, null, 1, 500, 512, 40 * 1024));

        PlacementRequest request = new PlacementRequest(
                1, 1, "CAND_INVALID_CPU", 1378606200L,
                "TEST", 0.05, vms, null
        );

        PlacementEvaluationResult result = PlacementEvaluator.evaluate(request, new DatacenterState());
        assertNotNull(result);
        assertFalse(result.isFeasible(), "Candidate with CPU " + invalidCpu + " must be marked infeasible");
        assertEquals(Double.POSITIVE_INFINITY, result.getEnergyWh(), "Infeasible candidate energy must be POSITIVE_INFINITY");
        assertTrue(result.getMessage().contains("Invalid CPU utilization"),
                "Message must report invalid CPU utilization: " + result.getMessage());
    }

    @Test
    @DisplayName("PlacementEvaluator.evaluate permits legitimate 0.0% idle demand")
    void testPlacementEvaluatorPermitsZeroCpuIdleDemand() {
        List<VmPlacementSpec> vms = new ArrayList<>();
        vms.add(new VmPlacementSpec("VM_001", 0, 0.0, 0.0, null, 1, 500, 512, 40 * 1024));

        PlacementRequest request = new PlacementRequest(
                1, 1, "CAND_ZERO_CPU", 1378606200L,
                "TEST", 0.05, vms, null
        );

        PlacementEvaluationResult result = PlacementEvaluator.evaluate(request, new DatacenterState());
        assertNotNull(result);
        assertTrue(result.isFeasible(), "Candidate with 0.0% CPU must be feasible");
        assertTrue(result.getEnergyWh() > 0.0, "Active host draws idle static power");
        assertEquals(0, result.getSlaViolations(), "Idle VM has zero SLA violations");
    }

    @Test
    @DisplayName("BitbrainsUtilizationModel.clampFraction preserves 0.0 and enforces [0.0, 1.0] bounds")
    void testBitbrainsClampFractionPreservesZero() {
        assertEquals(0.0, BitbrainsUtilizationModel.clampFraction(0.0), EPSILON, "0.0 must be preserved");
        assertEquals(0.5, BitbrainsUtilizationModel.clampFraction(0.5), EPSILON, "0.5 must be preserved");
        assertEquals(1.0, BitbrainsUtilizationModel.clampFraction(1.0), EPSILON, "1.0 must be preserved");

        // Out of bounds
        assertEquals(0.0, BitbrainsUtilizationModel.clampFraction(-0.5), EPSILON, "Negative must clamp to 0.0");
        assertEquals(1.0, BitbrainsUtilizationModel.clampFraction(1.5), EPSILON, "Values > 1.0 must clamp to 1.0");

        // Non-finite
        assertEquals(0.0, BitbrainsUtilizationModel.clampFraction(Double.NaN), EPSILON, "NaN must fall back to 0.0");
        assertEquals(0.0, BitbrainsUtilizationModel.clampFraction(Double.POSITIVE_INFINITY), EPSILON, "+Inf must fall back to 0.0");
        assertEquals(0.0, BitbrainsUtilizationModel.clampFraction(Double.NEGATIVE_INFINITY), EPSILON, "-Inf must fall back to 0.0");
    }

    @Test
    @DisplayName("BitbrainsUtilizationModel trace replay preserves legitimate 0.0 idle utilization")
    void testBitbrainsTraceReplayPreservesZero() throws IOException {
        File tempTrace = File.createTempFile("bitbrains_zero_test", ".csv");
        tempTrace.deleteOnExit();

        try (FileWriter fw = new FileWriter(tempTrace)) {
            fw.write("Timestamp [ms];CPU cores;CPU capacity provisioned [MHZ];CPU usage [MHZ];CPU usage [%];Memory capacity provisioned [KB]\n");
            fw.write("1378600000;2;2660;0.0;0.0;4194304\n");
            fw.write("1378600300;2;2660;0.0;0.0;4194304\n");
            fw.write("1378600600;2;2660;1330.0;50.0;4194304\n");
        }

        BitbrainsUtilizationModel model = new BitbrainsUtilizationModel(tempTrace.getAbsolutePath(), 1378600000L);
        // At simulation time 0.0s (offset 0), trace has 0.0% CPU usage
        double utilAtZero = model.getUtilization(0.0);
        assertEquals(0.0, utilAtZero, EPSILON, "Trace with 0.0% CPU must return 0.0 utilization (not 0.0001)");

        // At simulation time 600.0s (offset 600), trace has 50.0% CPU usage
        double utilAt600 = model.getUtilization(600.0);
        assertEquals(0.50, utilAt600, EPSILON, "Trace with 50.0% CPU must return 0.50 utilization");
    }
}
