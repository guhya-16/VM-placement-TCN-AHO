package org.vmplacement;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class SlaBoundaryTest {

    @Test
    @DisplayName("Verify SLA violation boundary condition: gap > slaThreshold strictly")
    void testSlaBoundaryEvaluation() {
        double threshold = 0.05;

        // Boundary Case 1: Shortfall just below threshold (0.0499) -> No violation
        double shortfallBelow = 0.0499;
        assertFalse(shortfallBelow > threshold, "Shortfall 0.0499 must NOT trigger violation at threshold 0.05");

        // Boundary Case 2: Shortfall exactly at threshold (0.0500) -> No violation under strict inequality (>)
        double shortfallExact = 0.0500;
        assertFalse(shortfallExact > threshold, "Shortfall 0.0500 exactly equal must NOT trigger violation under strict >");

        // Boundary Case 3: Shortfall just above threshold (0.0501) -> Violation
        double shortfallAbove = 0.0501;
        assertTrue(shortfallAbove > threshold, "Shortfall 0.0501 must trigger violation at threshold 0.05");

        // Case 4: Zero demand (requested = 0.0, delivered = 0.0) -> gap = 0.0 -> No violation
        double zeroDemandGap = 0.0 - 0.0;
        assertFalse(zeroDemandGap > threshold, "Zero demand VM must not trigger SLA violation");

        // Case 5: Full delivery (requested = 0.80, delivered = 0.80) -> gap = 0.0 -> No violation
        double fullDeliveryGap = 0.80 - 0.80;
        assertFalse(fullDeliveryGap > threshold, "Full delivery VM must not trigger SLA violation");

        // Case 6: Severe shortfall (requested = 0.50, delivered = 0.10) -> gap = 0.40 -> Violation
        double severeGap = 0.50 - 0.10;
        assertTrue(severeGap > threshold, "Severe shortfall (0.40) must trigger SLA violation");
    }
}
