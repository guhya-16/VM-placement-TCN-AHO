package com.vmplacement.optimization;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class AdaptiveHippopotamusOptimization {

    private static final double VOLATILITY_P95 = 4.675352;

    private static final double MIN_EXPLORATION = 0.20;
    private static final double MAX_EXPLORATION = 0.80;

    private final List<VM> vms;
    private final List<PM> pms;

    private final int populationSize;
    private final int maxIterations;

    private final Random random;

    private final List<Hippopotamus> population;

    private Hippopotamus bestHippo;
    private int currentIteration = 0;
    private int bestIteration = 0;

    private final boolean verbose;

    private final List<RiskState> riskStates;

    /*
     * Common initial population for fair comparison.
     */
    private final List<PlacementSolution>
            suppliedInitialPopulation;

    private double workloadVariation;
    private double predictionRisk;
    private double adaptiveSignal;
    private double explorationProbability;

    private String adaptiveMode;

    // ============================================================
    // Constructors
    // ============================================================

    public AdaptiveHippopotamusOptimization(
            List<VM> vms,
            List<PM> pms,
            int populationSize,
            int maxIterations) {

        this(
                vms,
                pms,
                populationSize,
                maxIterations,
                System.currentTimeMillis(),
                null,
                null,
                true
        );
    }

    public AdaptiveHippopotamusOptimization(
            List<VM> vms,
            List<PM> pms,
            int populationSize,
            int maxIterations,
            boolean verbose) {

        this(
                vms,
                pms,
                populationSize,
                maxIterations,
                System.currentTimeMillis(),
                null,
                null,
                verbose
        );
    }

    public AdaptiveHippopotamusOptimization(
            List<VM> vms,
            List<PM> pms,
            int populationSize,
            int maxIterations,
            List<RiskState> riskStates) {

        this(
                vms,
                pms,
                populationSize,
                maxIterations,
                System.currentTimeMillis(),
                riskStates,
                null,
                true
        );
    }

    public AdaptiveHippopotamusOptimization(
            List<VM> vms,
            List<PM> pms,
            int populationSize,
            int maxIterations,
            List<RiskState> riskStates,
            boolean verbose) {

        this(
                vms,
                pms,
                populationSize,
                maxIterations,
                System.currentTimeMillis(),
                riskStates,
                null,
                verbose
        );
    }

    /*
     * Main constructor for fair experiment.
     */
    public AdaptiveHippopotamusOptimization(
            List<VM> vms,
            List<PM> pms,
            int populationSize,
            int maxIterations,
            long seed,
            List<RiskState> riskStates,
            List<PlacementSolution> initialPopulation,
            boolean verbose) {

        if (vms == null || vms.isEmpty()) {
            throw new IllegalArgumentException(
                    "VM list cannot be empty."
            );
        }

        if (pms == null || pms.isEmpty()) {
            throw new IllegalArgumentException(
                    "PM list cannot be empty."
            );
        }

        if (populationSize <= 0) {
            throw new IllegalArgumentException(
                    "Population size must be greater than zero."
            );
        }

        if (maxIterations <= 0) {
            throw new IllegalArgumentException(
                    "Maximum iterations must be greater than zero."
            );
        }

        this.vms = vms;
        this.pms = pms;

        this.populationSize =
                populationSize;

        this.maxIterations =
                maxIterations;

        this.random =
                new Random(seed);

        this.population =
                new ArrayList<>();

        this.bestHippo =
                null;

        this.verbose =
                verbose;

        this.riskStates =
                riskStates == null
                        ? new ArrayList<>()
                        : riskStates;

        this.suppliedInitialPopulation =
                deepCopyPopulation(
                        initialPopulation
                );

        this.workloadVariation = 0.0;
        this.predictionRisk = 0.0;
        this.adaptiveSignal = 0.0;

        this.explorationProbability =
                MIN_EXPLORATION;

        this.adaptiveMode =
                "EXPLOITATION";

        calculateAdaptiveParameters();
    }

    public AdaptiveHippopotamusOptimization(
            List<VM> vms,
            List<PM> pms,
            int populationSize,
            int maxIterations,
            long seed) {

        this(
                vms,
                pms,
                populationSize,
                maxIterations,
                seed,
                null,
                null,
                true
        );
    }

    public AdaptiveHippopotamusOptimization(
            List<VM> vms,
            List<PM> pms,
            int populationSize,
            int maxIterations,
            long seed,
            List<RiskState> riskStates) {

        this(
                vms,
                pms,
                populationSize,
                maxIterations,
                seed,
                riskStates,
                null,
                true
        );
    }

    // ============================================================
    // Adaptive controller
    // ============================================================

    private void calculateAdaptiveParameters() {

        if (!riskStates.isEmpty()) {

            double volatilitySum = 0.0;
            double riskSum = 0.0;

            for (RiskState state : riskStates) {

                volatilitySum +=
                        state.getVolatilityScore();

                riskSum +=
                        state.getRiskScore();
            }

            workloadVariation =
                    clamp(
                            (
                                    volatilitySum
                                            / riskStates.size()
                            ) / VOLATILITY_P95,
                            0.0,
                            1.0
                    );

            predictionRisk =
                    clamp(
                            riskSum
                                    / riskStates.size(),
                            0.0,
                            1.0
                    );

        } else {

            workloadVariation =
                    clamp(
                            WorkloadVariation
                                    .calculateAverageNormalized(
                                            vms
                                    ),
                            0.0,
                            1.0
                    );

            predictionRisk =
                    clamp(
                            PredictionRisk
                                    .calculateAverage(
                                            vms
                                    ),
                            0.0,
                            1.0
                    );
        }

        /*
         * Workload variation gets 60% importance.
         * Prediction risk gets 40%.
         */
        adaptiveSignal =
                0.60 * workloadVariation
                        + 0.40 * predictionRisk;

        if (adaptiveSignal < 0.10) {

            adaptiveMode =
                    "EXPLOITATION";

        } else if (adaptiveSignal < 0.25) {

            adaptiveMode =
                    "BALANCED";

        } else {

            adaptiveMode =
                    "EXPLORATION";
        }

        explorationProbability =
                clamp(
                        MIN_EXPLORATION
                                + adaptiveSignal
                                * (
                                MAX_EXPLORATION
                                        - MIN_EXPLORATION
                        ),
                        MIN_EXPLORATION,
                        MAX_EXPLORATION
                );
    }

    // ============================================================
    // Optimization
    // ============================================================

    public PlacementSolution optimize() {

        population.clear();
        bestHippo = null;
        currentIteration = 0;
        bestIteration = 0;

        initializePopulation();

        if (population.isEmpty()) {

            throw new IllegalStateException(
                    "Unable to create a feasible Adaptive HO population."
            );
        }

        if (verbose) {

            System.out.println(
                    "Created "
                            + population.size()
                            + " out of "
                            + populationSize
                            + " hippos."
            );

            System.out.printf(
                    "Workload Variation      : %.4f%n",
                    workloadVariation
            );

            System.out.printf(
                    "Prediction Risk         : %.4f%n",
                    predictionRisk
            );

            System.out.printf(
                    "Adaptive Signal         : %.4f%n",
                    adaptiveSignal
            );

            System.out.printf(
                    "Adaptive Mode           : %s%n",
                    adaptiveMode
            );

            System.out.printf(
                    "Exploration Probability : %.4f%n",
                    explorationProbability
            );
        }

        for (int iteration = 1;
             iteration <= maxIterations;
             iteration++) {

            currentIteration = iteration;

            for (Hippopotamus hippo
                    : population) {

                PlacementSolution candidate;

                /*
                 * Adaptive exploration/exploitation.
                 */
                if (random.nextDouble()
                        < explorationProbability) {

                    candidate =
                            adaptiveExploration(
                                    hippo.getSolution()
                            );

                } else {

                    candidate =
                            adaptiveExploitation(
                                    hippo.getSolution()
                            );
                }

                evaluateAndAccept(
                        hippo,
                        candidate
                );

                /*
                 * Defense.
                 */
                candidate =
                        defense(
                                hippo.getSolution()
                        );

                evaluateAndAccept(
                        hippo,
                        candidate
                );

                /*
                 * Escape.
                 */
                candidate =
                        escape(
                                hippo.getSolution()
                        );

                evaluateAndAccept(
                        hippo,
                        candidate
                );

                /*
                 * Adaptive mutation.
                 */
                double mutationProbability =
                        0.10
                                + 0.20
                                * adaptiveSignal;

                if (random.nextDouble()
                        < mutationProbability) {

                    candidate =
                            mutate(
                                    hippo.getSolution()
                            );

                    evaluateAndAccept(
                            hippo,
                            candidate
                    );
                }
            }

            if (verbose) {

                System.out.printf(
                        "Iteration %d/%d | Best Fitness = %.8f%n",
                        iteration,
                        maxIterations,
                        bestHippo.getFitness()
                );
            }
        }

        return copySolution(
                bestHippo.getSolution()
        );
    }

    // ============================================================
    // Initialization
    // ============================================================

    private void initializePopulation() {

        /*
         * FAIR EXPERIMENT:
         *
         * Use exactly the same population generated by
         * MultiSeedExperiment.
         */
        if (suppliedInitialPopulation != null
                && !suppliedInitialPopulation.isEmpty()) {

            for (PlacementSolution solution
                    : suppliedInitialPopulation) {

                if (population.size()
                        >= populationSize) {

                    break;
                }

                if (!isFeasible(solution)) {
                    continue;
                }

                PlacementSolution copy =
                        copySolution(solution);

                double fitness =
                        FitnessFunction.calculate(
                                vms,
                                pms,
                                copy
                        );

                population.add(
                        new Hippopotamus(
                                copy,
                                fitness
                        )
                );
            }

            updateGlobalBest();

            return;
        }

        /*
         * Normal initialization.
         */
        int attempts = 0;

        int maxAttempts =
                populationSize * 100;

        while (population.size()
                < populationSize
                && attempts < maxAttempts) {

            attempts++;

            PlacementSolution solution =
                    PlacementRepair
                            .createRandomFeasibleSolution(
                                    vms,
                                    pms,
                                    random,
                                    0.65
                            );

            if (solution == null) {
                continue;
            }

            if (!isFeasible(solution)) {
                continue;
            }

            if (containsSolution(solution)) {
                continue;
            }

            double fitness =
                    FitnessFunction.calculate(
                            vms,
                            pms,
                            solution
                    );

            population.add(
                    new Hippopotamus(
                            copySolution(solution),
                            fitness
                    )
            );
        }

        updateGlobalBest();
    }

    // ============================================================
    // Adaptive exploration
    // ============================================================

    private PlacementSolution adaptiveExploration(
            PlacementSolution current) {

        int[] mapping =
                current.getVmToPm().clone();

        int changes =
                Math.max(
                        1,
                        (int) Math.ceil(
                                vms.size()
                                        * (
                                        0.10
                                                + 0.50
                                                * adaptiveSignal
                                )
                        )
                );

        for (int i = 0;
             i < changes;
             i++) {

            int vmIndex =
                    random.nextInt(
                            vms.size()
                    );

            int pmIndex =
                    random.nextInt(
                            pms.size()
                    );

            mapping[vmIndex] =
                    pmIndex;
        }

        PlacementSolution candidate =
                new PlacementSolution(
                        mapping
                );

        return repairOrCurrent(
                candidate,
                current
        );
    }

    // ============================================================
    // Adaptive exploitation
    // ============================================================

    private PlacementSolution adaptiveExploitation(
            PlacementSolution current) {

        int[] mapping =
                current.getVmToPm().clone();

        if (bestHippo == null) {
            return copySolution(current);
        }

        int[] bestMapping =
                bestHippo
                        .getSolution()
                        .getVmToPm();

        double strength =
                0.70
                        + 0.25
                        * (
                        1.0
                                - adaptiveSignal
                );

        for (int i = 0;
             i < vms.size();
             i++) {

            if (random.nextDouble()
                    < strength) {

                mapping[i] =
                        bestMapping[i];
            }
        }

        PlacementSolution candidate =
                new PlacementSolution(
                        mapping
                );

        return repairOrCurrent(
                candidate,
                current
        );
    }

    // ============================================================
    // Defense
    // ============================================================

    private PlacementSolution defense(
            PlacementSolution current) {

        int[] mapping =
                current.getVmToPm().clone();

        int changes =
                Math.max(
                        1,
                        (int) Math.ceil(
                                vms.size()
                                        * (
                                        0.05
                                                + 0.20
                                                * adaptiveSignal
                                )
                        )
                );

        for (int i = 0;
             i < changes;
             i++) {

            int vmIndex =
                    random.nextInt(
                            vms.size()
                    );

            if (bestHippo != null
                    && random.nextDouble()
                    > adaptiveSignal) {

                mapping[vmIndex] =
                        bestHippo
                                .getSolution()
                                .getVmToPm()
                                [vmIndex];

            } else {

                mapping[vmIndex] =
                        random.nextInt(
                                pms.size()
                        );
            }
        }

        PlacementSolution candidate =
                new PlacementSolution(
                        mapping
                );

        return repairOrCurrent(
                candidate,
                current
        );
    }

    // ============================================================
    // Escape
    // ============================================================

    private PlacementSolution escape(
            PlacementSolution current) {

        int[] mapping =
                current.getVmToPm().clone();

        if (bestHippo != null) {

            int[] bestMapping =
                    bestHippo
                            .getSolution()
                            .getVmToPm();

            double probability =
                    0.30
                            + 0.40
                            * adaptiveSignal;

            for (int i = 0;
                 i < vms.size();
                 i++) {

                if (random.nextDouble()
                        < probability) {

                    mapping[i] =
                            bestMapping[i];
                }
            }
        }

        if (random.nextDouble()
                < adaptiveSignal) {

            int vmIndex =
                    random.nextInt(
                            vms.size()
                    );

            mapping[vmIndex] =
                    random.nextInt(
                            pms.size()
                    );
        }

        PlacementSolution candidate =
                new PlacementSolution(
                        mapping
                );

        return repairOrCurrent(
                candidate,
                current
        );
    }

    // ============================================================
    // Mutation
    // ============================================================

    private PlacementSolution mutate(
            PlacementSolution current) {

        int[] mapping =
                current.getVmToPm().clone();

        int changes =
                Math.max(
                        1,
                        (int) Math.ceil(
                                vms.size()
                                        * (
                                        0.05
                                                + 0.15
                                                * adaptiveSignal
                                )
                        )
                );

        for (int i = 0;
             i < changes;
             i++) {

            int vmIndex =
                    random.nextInt(
                            vms.size()
                    );

            mapping[vmIndex] =
                    random.nextInt(
                            pms.size()
                    );
        }

        PlacementSolution candidate =
                new PlacementSolution(
                        mapping
                );

        return repairOrCurrent(
                candidate,
                current
        );
    }

    // ============================================================
    // Feasibility
    // ============================================================

    private boolean isFeasible(
            PlacementSolution solution) {

        if (solution == null
                || solution.getVmToPm() == null) {

            return false;
        }

        /*
         * Use the validator that actually exists
         * in your project.
         */
        return PlacementValidator.isFeasible(
                vms,
                pms,
                solution
        );
    }

    private PlacementSolution repairOrCurrent(
            PlacementSolution candidate,
            PlacementSolution current) {

        if (isFeasible(candidate)) {
            return candidate;
        }

        /*
         * Your actual project signature is:
         *
         * repair(vms, pms, solution, random)
         */
        PlacementSolution repaired =
                PlacementRepair.repair(
                        vms,
                        pms,
                        candidate,
                        random
                );

        if (repaired != null
                && isFeasible(repaired)) {

            return repaired;
        }

        return copySolution(current);
    }

    // ============================================================
    // Evaluation
    // ============================================================

    private void evaluateAndAccept(
            Hippopotamus hippo,
            PlacementSolution candidate) {

        if (candidate == null
                || !isFeasible(candidate)) {

            return;
        }

        double candidateFitness =
                FitnessFunction.calculate(
                        vms,
                        pms,
                        candidate
                );

        if (candidateFitness
                < hippo.getFitness()) {

            hippo.setSolution(
                    copySolution(candidate)
            );

            hippo.setFitness(
                    candidateFitness
            );

            if (bestHippo == null
                    || candidateFitness
                    < bestHippo.getFitness()) {

                bestHippo =
                        new Hippopotamus(
                                copySolution(
                                        candidate
                                ),
                                candidateFitness
                        );
                bestIteration = currentIteration;
            }
        }
    }

    public int getBestIteration() {
        return bestIteration;
    }

    public PlacementSolution getBestPlacement() {
        return getBestSolution();
    }

    private void updateGlobalBest() {

        for (Hippopotamus hippo
                : population) {

            if (bestHippo == null
                    || hippo.getFitness()
                    < bestHippo.getFitness()) {

                bestHippo =
                        new Hippopotamus(
                                copySolution(
                                        hippo.getSolution()
                                ),
                                hippo.getFitness()
                        );
                bestIteration = 0;
            }
        }
    }

    // ============================================================
    // Utilities
    // ============================================================

    private boolean containsSolution(
            PlacementSolution solution) {

        for (Hippopotamus hippo
                : population) {

            if (samePlacement(
                    hippo.getSolution(),
                    solution
            )) {

                return true;
            }
        }

        return false;
    }

    private boolean samePlacement(
            PlacementSolution a,
            PlacementSolution b) {

        if (a == null || b == null) {
            return false;
        }

        int[] x =
                a.getVmToPm();

        int[] y =
                b.getVmToPm();

        if (x == null
                || y == null
                || x.length != y.length) {

            return false;
        }

        for (int i = 0;
             i < x.length;
             i++) {

            if (x[i] != y[i]) {
                return false;
            }
        }

        return true;
    }

    private PlacementSolution copySolution(
            PlacementSolution solution) {

        if (solution == null) {
            return null;
        }

        return new PlacementSolution(
                solution
                        .getVmToPm()
                        .clone()
        );
    }

    private List<PlacementSolution>
    deepCopyPopulation(
            List<PlacementSolution> source) {

        if (source == null) {
            return null;
        }

        List<PlacementSolution> copy =
                new ArrayList<>();

        for (PlacementSolution solution
                : source) {

            if (solution != null) {

                copy.add(
                        copySolution(solution)
                );
            }
        }

        return copy;
    }

    private double clamp(
            double value,
            double min,
            double max) {

        return Math.max(
                min,
                Math.min(max, value)
        );
    }

    // ============================================================
    // Getters
    // ============================================================

    public double getBestFitness() {

        if (bestHippo == null) {
            return Double.MAX_VALUE;
        }

        return bestHippo.getFitness();
    }

    public PlacementSolution getBestSolution() {

        if (bestHippo == null) {
            return null;
        }

        return copySolution(
                bestHippo.getSolution()
        );
    }

    public double getWorkloadVariation() {
        return workloadVariation;
    }

    public double getPredictionRisk() {
        return predictionRisk;
    }

    public double getAdaptiveSignal() {
        return adaptiveSignal;
    }

    public double getExplorationProbability() {
        return explorationProbability;
    }

    public String getAdaptiveMode() {
        return adaptiveMode;
    }

    public String getCurrentAdaptiveMode() {
        return adaptiveMode;
    }

    public double getCurrentExplorationProbability() {
        return explorationProbability;
    }

    // ============================================================
    // Inner class
    // ============================================================

    private static class Hippopotamus {

        private PlacementSolution solution;
        private double fitness;

        public Hippopotamus(
                PlacementSolution solution,
                double fitness) {

            this.solution =
                    solution;

            this.fitness =
                    fitness;
        }

        public PlacementSolution getSolution() {
            return solution;
        }

        public void setSolution(
                PlacementSolution solution) {

            this.solution =
                    solution;
        }

        public double getFitness() {
            return fitness;
        }

        public void setFitness(
                double fitness) {

            this.fitness =
                    fitness;
        }
    }
}