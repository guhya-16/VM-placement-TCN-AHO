package com.vmplacement.optimization;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Non-adaptive HO-inspired baseline for ablation experiments.
 *
 * It deliberately does not calculate or consume the WA-AHO adaptive signal,
 * prediction risk, or workload volatility. Search controls are fixed for the
 * entire run and can be configured independently in application.properties.
 */
public final class StandardHippopotamusOptimization {
    private final EpochState epoch;
    private final int populationSize;
    private final int maxIterations;
    private final Random random;
    private final boolean verbose;
    private final CandidateEvaluationBridge bridge;
    private final FitnessFunction fitnessFunction;
    private final double explorationProbability;
    private final double mutationProbability;
    private final double attractionProbability;
    private final double defenseFraction;
    private final List<Hippopotamus> population = new ArrayList<>();
    private Hippopotamus best;
    private int bestIteration;
    private int currentEpochIndex;

    public StandardHippopotamusOptimization(
            EpochState epoch, int populationSize, int maxIterations, long seed, boolean verbose,
            CandidateEvaluationBridge bridge, FitnessFunction fitnessFunction,
            double explorationProbability, double mutationProbability,
            double attractionProbability, double defenseFraction) {
        if (epoch.getActiveVms().isEmpty()) {
            throw new IllegalArgumentException("Epoch has no active VMs: " + epoch.getTimestamp());
        }
        if (populationSize <= 0 || maxIterations <= 0) {
            throw new IllegalArgumentException("Population/iterations must be positive");
        }
        requireProbability(explorationProbability, "standard_ho.exploration_probability");
        requireProbability(mutationProbability, "standard_ho.mutation_probability");
        requireProbability(attractionProbability, "standard_ho.attraction_probability");
        if (!Double.isFinite(defenseFraction) || defenseFraction < 0.0 || defenseFraction > 1.0) {
            throw new IllegalArgumentException("standard_ho.defense_fraction must be in [0,1]");
        }
        this.epoch = epoch;
        this.populationSize = populationSize;
        this.maxIterations = maxIterations;
        this.random = new Random(seed);
        this.verbose = verbose;
        this.bridge = bridge;
        this.fitnessFunction = fitnessFunction;
        this.explorationProbability = explorationProbability;
        this.mutationProbability = mutationProbability;
        this.attractionProbability = attractionProbability;
        this.defenseFraction = defenseFraction;
    }

    private static void requireProbability(double value, String name) {
        if (!Double.isFinite(value) || value < 0.0 || value > 1.0) {
            throw new IllegalArgumentException(name + " must be in [0,1]");
        }
    }

    public PlacementSolution optimize() {
        initialize();
        evaluatePopulation(population, 1);
        updateBest(1);
        for (int iteration = 2; iteration <= maxIterations; iteration++) {
            List<Hippopotamus> candidates = new ArrayList<>(population.size());
            for (Hippopotamus hippo : population) {
                candidates.add(new Hippopotamus(generateCandidate(hippo.getSolution())));
            }
            evaluatePopulation(candidates, iteration);
            for (int i = 0; i < population.size(); i++) {
                if (candidates.get(i).getFitness() < population.get(i).getFitness()) {
                    population.set(i, candidates.get(i));
                }
            }
            updateBest(iteration);
            if (verbose) {
                System.out.printf("timestamp=%d iteration=%d/%d bestFitness=%.8f mode=STANDARD_HO exploration=%.4f%n",
                        epoch.getTimestamp(), iteration, maxIterations, best.getFitness(), explorationProbability);
            }
        }
        return best.getSolution().copy();
    }

    private void initialize() {
        population.clear();
        int attempts = 0;
        int maxAttempts = Math.max(populationSize * 50, populationSize);
        while (population.size() < populationSize && attempts++ < maxAttempts) {
            PlacementSolution solution = PlacementRepair.randomFeasible(
                    epoch.getActiveVms(), epoch.getPms(), random);
            if (solution != null) population.add(new Hippopotamus(solution));
        }
        if (population.size() < populationSize) {
            throw new IllegalStateException("Could not build the requested feasible Standard HO population: requested="
                    + populationSize + ", built=" + population.size() + ", timestamp=" + epoch.getTimestamp());
        }
    }

    private void evaluatePopulation(List<Hippopotamus> candidates, int iteration) {
        if (verbose) {
            System.out.printf("Standard HO | epoch=%d | iteration=%d%n", currentEpochIndex, iteration);
        }
        List<CloudSimEvaluationResult> results = new ArrayList<>(candidates.size());
        for (int i = 0; i < candidates.size(); i++) {
            results.add(bridge.evaluate(currentEpochIndex, iteration, i + 1, epoch,
                    candidates.get(i).getSolution()));
        }
        List<Double> scores = fitnessFunction.score(results);
        for (int i = 0; i < candidates.size(); i++) {
            candidates.get(i).setEvaluation(results.get(i));
            candidates.get(i).setFitness(scores.get(i));
        }
    }

    private void updateBest(int iteration) {
        for (Hippopotamus hippo : population) {
            if (best == null || hippo.getFitness() < best.getFitness()) {
                best = hippo;
                bestIteration = iteration;
            }
        }
    }

    private PlacementSolution generateCandidate(PlacementSolution current) {
        int[] mapping = current.getVmToPm();
        if (random.nextDouble() < explorationProbability) {
            int changes = Math.max(1, (int) Math.ceil(mapping.length * 0.30));
            for (int c = 0; c < changes; c++) {
                mapping[random.nextInt(mapping.length)] = random.nextInt(epoch.getPms().size());
            }
        } else if (best != null) {
            int[] bestMapping = best.getSolution().getVmToPm();
            for (int i = 0; i < mapping.length; i++) {
                if (random.nextDouble() < attractionProbability) mapping[i] = bestMapping[i];
            }
        }

        int defenseChanges = (int) Math.ceil(mapping.length * defenseFraction);
        for (int c = 0; c < defenseChanges; c++) {
            if (random.nextBoolean()) mapping[random.nextInt(mapping.length)] = random.nextInt(epoch.getPms().size());
        }
        if (random.nextDouble() < mutationProbability) {
            mapping[random.nextInt(mapping.length)] = random.nextInt(epoch.getPms().size());
        }

        PlacementSolution repaired = PlacementRepair.repair(
                epoch.getActiveVms(), epoch.getPms(), new PlacementSolution(mapping), random);
        return repaired == null ? current.copy() : repaired;
    }

    public void setEpochIndex(int epochIndex) {
        if (epochIndex <= 0) throw new IllegalArgumentException("epochIndex must be positive");
        this.currentEpochIndex = epochIndex;
    }

    public double getBestFitness() { return best == null ? Double.POSITIVE_INFINITY : best.getFitness(); }
    public PlacementSolution getBestSolution() { return best == null ? null : best.getSolution().copy(); }
    public CloudSimEvaluationResult getBestEvaluation() { return best == null ? null : best.getEvaluation(); }
    public int getBestIteration() { return bestIteration; }
}
