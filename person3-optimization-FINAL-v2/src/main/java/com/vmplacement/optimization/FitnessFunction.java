package com.vmplacement.optimization;

import java.util.ArrayList;
import java.util.List;

/**
 * Weighted fitness over metrics returned by CloudSim (or the clearly marked development evaluator).
 *
 * Metrics are normalized against configured physical/contract scales rather than min/max of the
 * current candidate batch. This keeps fitness values comparable across AHO iterations and avoids
 * changing the meaning of a fitness value merely because the current population changed.
 */
public final class FitnessFunction {
    private final double energyWeight, slaWeight, activePmWeight, meanUtilizationWeight, migrationWeight;
    private final double energyScaleWh, slaScale, activePmScale, meanUtilizationScale, migrationScale;

    public FitnessFunction(double energyWeight, double slaWeight, double activePmWeight,
                           double meanUtilizationWeight, double migrationWeight,
                           double energyScaleWh, double slaScale, double activePmScale,
                           double meanUtilizationScale, double migrationScale) {
        double sum = energyWeight + slaWeight + activePmWeight + meanUtilizationWeight + migrationWeight;
        if (sum <= 0) throw new IllegalArgumentException("At least one fitness weight must be positive");
        if (energyScaleWh <= 0 || slaScale <= 0 || activePmScale <= 0
                || meanUtilizationScale <= 0 || migrationScale <= 0) {
            throw new IllegalArgumentException("Fitness normalization scales must be positive");
        }
        this.energyWeight = energyWeight / sum;
        this.slaWeight = slaWeight / sum;
        this.activePmWeight = activePmWeight / sum;
        this.meanUtilizationWeight = meanUtilizationWeight / sum;
        this.migrationWeight = migrationWeight / sum;
        this.energyScaleWh = energyScaleWh;
        this.slaScale = slaScale;
        this.activePmScale = activePmScale;
        this.meanUtilizationScale = meanUtilizationScale;
        this.migrationScale = migrationScale;
    }

    public List<Double> score(List<CloudSimEvaluationResult> results) {
        List<Double> out = new ArrayList<>(results.size());
        for (CloudSimEvaluationResult r : results) {
            if (!r.isFeasible()) {
                out.add(Double.POSITIVE_INFINITY);
                continue;
            }
            if (energyWeight > 0 && !Double.isFinite(r.getEnergyWh())) {
                throw new IllegalStateException(
                        "CloudSim returned no finite energy while energy weight is positive");
            }

            double value = 0.0;
            if (energyWeight > 0) value += energyWeight * normalize(r.getEnergyWh(), energyScaleWh);
            if (slaWeight > 0) value += slaWeight * normalize(r.getSlaViolations(), slaScale);
            if (activePmWeight > 0) value += activePmWeight * normalize(r.getActivePms(), activePmScale);
            if (meanUtilizationWeight > 0) {
                value += meanUtilizationWeight * normalize(r.getMeanUtilization(), meanUtilizationScale);
            }
            if (migrationWeight > 0) {
                value += migrationWeight * normalize(r.getMigrationCost(), migrationScale);
            }
            out.add(value);
        }
        return out;
    }

    private static double normalize(double value, double scale) {
        if (!Double.isFinite(value)) return Double.POSITIVE_INFINITY;
        return Math.max(0.0, value) / scale;
    }
}
