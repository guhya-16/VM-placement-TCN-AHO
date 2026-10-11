package com.vmplacement.optimization;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static org.junit.jupiter.api.Assertions.*;

public class CpuScalingTest {

    private static final double EPSILON = 1e-9;

    @Test
    @DisplayName("Case 1: 0.75 input in percent mode represents 0.75%, converting to fraction 0.0075")
    void testPointSevenFivePercent() {
        double cpuPercent = VM.cpuPercent(0.75, 1.0);
        assertEquals(0.75, cpuPercent, EPSILON, "0.75 with scale 1.0 must remain 0.75%");

        double fraction = VM.toUtilizationFraction(cpuPercent);
        assertEquals(0.0075, fraction, EPSILON, "0.75% must convert to utilization fraction 0.0075");
    }

    @Test
    @DisplayName("Case 2: 1.0 input in percent mode represents 1.0% (not 100%), converting to fraction 0.010")
    void testOnePointZeroPercent() {
        double cpuPercent = VM.cpuPercent(1.0, 1.0);
        assertEquals(1.0, cpuPercent, EPSILON, "1.0 with scale 1.0 must remain 1.0%");

        double fraction = VM.toUtilizationFraction(cpuPercent);
        assertEquals(0.010, fraction, EPSILON, "1.0% must convert to utilization fraction 0.010 (not 1.0)");
    }

    @Test
    @DisplayName("Case 3: 75.0 input in percent mode represents 75.0%, converting to fraction 0.75")
    void testSeventyFivePercent() {
        double cpuPercent = VM.cpuPercent(75.0, 1.0);
        assertEquals(75.0, cpuPercent, EPSILON, "75.0 with scale 1.0 must remain 75.0%");

        double fraction = VM.toUtilizationFraction(cpuPercent);
        assertEquals(0.75, fraction, EPSILON, "75.0% must convert to utilization fraction 0.75");
    }

    @Test
    @DisplayName("Case 3b: 5.0 input in percent mode represents 5.0%, converting to fraction 0.05")
    void testFivePercent() {
        double cpuPercent = VM.cpuPercent(5.0, 1.0);
        assertEquals(5.0, cpuPercent, EPSILON, "5.0 with scale 1.0 must remain 5.0%");

        double fraction = VM.toUtilizationFraction(cpuPercent);
        assertEquals(0.05, fraction, EPSILON, "5.0% must convert to utilization fraction 0.05");
    }

    @Test
    @DisplayName("Case 3c: 50.0 input in percent mode represents 50.0%, converting to fraction 0.50")
    void testFiftyPercent() {
        double cpuPercent = VM.cpuPercent(50.0, 1.0);
        assertEquals(50.0, cpuPercent, EPSILON, "50.0 with scale 1.0 must remain 50.0%");

        double fraction = VM.toUtilizationFraction(cpuPercent);
        assertEquals(0.50, fraction, EPSILON, "50.0% must convert to utilization fraction 0.50");
    }

    @Test
    @DisplayName("Case 4: 0.0 input in percent mode represents 0.0% (idle), converting to fraction 0.0")
    void testZeroPercentIdle() {
        double cpuPercent = VM.cpuPercent(0.0, 1.0);
        assertEquals(0.0, cpuPercent, EPSILON, "0.0 with scale 1.0 must remain 0.0%");

        double fraction = VM.toUtilizationFraction(cpuPercent);
        assertEquals(0.0, fraction, EPSILON, "0.0% must convert to utilization fraction 0.0 (not 1.0 fallback)");
    }

    @Test
    @DisplayName("Case 5: 100.0 input in percent mode represents 100.0%, converting to fraction 1.0")
    void testOneHundredPercentFull() {
        double cpuPercent = VM.cpuPercent(100.0, 1.0);
        assertEquals(100.0, cpuPercent, EPSILON, "100.0 with scale 1.0 must remain 100.0%");

        double fraction = VM.toUtilizationFraction(cpuPercent);
        assertEquals(1.0, fraction, EPSILON, "100.0% must convert to utilization fraction 1.0");
    }

    @ParameterizedTest
    @ValueSource(doubles = {-0.01, -10.0, 100.01, 150.0, Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY})
    @DisplayName("Case 6: Out of range and invalid values throw IllegalArgumentException")
    void testInvalidAndOutOfRangeCpuValues(double invalidValue) {
        assertThrows(IllegalArgumentException.class, () -> VM.cpuPercent(invalidValue, 1.0),
                "Value " + invalidValue + " must throw IllegalArgumentException");
    }

    @Test
    @DisplayName("Auto scale mode (scale < 0) safely defaults to percent scale 1.0 without threshold heuristics")
    void testAutoScaleDefaultsToPercent() {
        // In auto mode, 0.75 must be treated as 0.75% (not multiplied by 100)
        double cpuPercent = VM.cpuPercent(0.75, -1.0);
        assertEquals(0.75, cpuPercent, EPSILON);

        // In auto mode, 1.0 must be treated as 1.0% (not 100%)
        double cpuPercentOne = VM.cpuPercent(1.0, -1.0);
        assertEquals(1.0, cpuPercentOne, EPSILON);
    }
}
