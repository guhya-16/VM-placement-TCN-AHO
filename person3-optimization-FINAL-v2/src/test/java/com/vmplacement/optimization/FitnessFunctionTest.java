package com.vmplacement.optimization;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class FitnessFunctionTest {

    private static final double EPSILON = 1e-6;

    @Test
    @DisplayName("Verify lower fitness means better candidate and calculation matches formula")
    void testFitnessCalculationAndRanking() {
        FitnessFunction fitness = new FitnessFunction(
                0.40, 0.30, 0.15, 0.00, 0.15,
                200.0, 10.0, 4.0, 100.0, 4.0
        );

        // Candidate 1: Consolidated (2 PMs, 117 Wh, 0 SLA, 0 migrations)
        // Expected: 0.40 * (117.0 / 200.0) + 0.30 * (0 / 10.0) + 0.15 * (2 / 4.0) + 0.15 * (0 / 4.0)
        //         = 0.40 * 0.585 + 0 + 0.15 * 0.5 + 0 = 0.234 + 0.075 = 0.309
        CandidateEvaluationOutput out1 = new CandidateEvaluationOutput(
                1, 1, "C1", true, 117.0, 0L, 2, 0.001, 0, 0.0
        );

        // Candidate 2: Dispersed (4 PMs, 170 Wh, 0 SLA, 0 migrations)
        // Expected: 0.40 * (170.0 / 200.0) + 0.30 * 0 + 0.15 * (4 / 4.0) + 0
        //         = 0.40 * 0.85 + 0.15 * 1.0 = 0.340 + 0.150 = 0.490
        CandidateEvaluationOutput out2 = new CandidateEvaluationOutput(
                1, 1, "C2", true, 170.0, 0L, 4, 0.001, 0, 0.0
        );

        List<CloudSimEvaluationResult> results = List.of(
                new CloudSimEvaluationResult(out1),
                new CloudSimEvaluationResult(out2)
        );

        List<Double> scores = fitness.score(results);
        assertEquals(2, scores.size());
        assertEquals(0.309, scores.get(0), EPSILON, "Candidate 1 score should match 0.309");
        assertEquals(0.490, scores.get(1), EPSILON, "Candidate 2 score should match 0.490");
        assertTrue(scores.get(0) < scores.get(1), "Candidate 1 (consolidated) must have lower (better) fitness");
    }

    @Test
    @DisplayName("Verify infeasible candidates are rejected with POSITIVE_INFINITY")
    void testInfeasibleCandidateGetsInfinity() {
        FitnessFunction fitness = new FitnessFunction(
                0.40, 0.30, 0.15, 0.00, 0.15,
                200.0, 10.0, 4.0, 100.0, 4.0
        );

        CandidateEvaluationOutput infeasibleOut = new CandidateEvaluationOutput(
                1, 1, "C_INFEASIBLE", false, Double.POSITIVE_INFINITY, 0L, 0, 0.0, 0, 0.0
        );

        List<Double> scores = fitness.score(List.of(new CloudSimEvaluationResult(infeasibleOut)));
        assertEquals(1, scores.size());
        assertEquals(Double.POSITIVE_INFINITY, scores.get(0), "Infeasible candidate must have POSITIVE_INFINITY fitness");
    }

    @Test
    @DisplayName("Verify migration cost penalty adds to fitness score")
    void testMigrationCostImpact() {
        FitnessFunction fitness = new FitnessFunction(
                0.40, 0.30, 0.15, 0.00, 0.15,
                200.0, 10.0, 4.0, 100.0, 4.0
        );

        // Candidate with 2 migrations: 0.15 * (2.0 / 4.0) = 0.075 added to score
        CandidateEvaluationOutput outWithMigrations = new CandidateEvaluationOutput(
                2, 1, "C_MIG", true, 117.0, 0L, 2, 0.001, 2, 2.0
        );

        List<Double> scores = fitness.score(List.of(new CloudSimEvaluationResult(outWithMigrations)));
        // 0.309 (base) + 0.075 (migrations) = 0.384
        assertEquals(0.384, scores.get(0), EPSILON);
    }
}
