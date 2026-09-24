package com.vmplacement.optimization;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class HippopotamusOptimization {

    private final List<VM> vms;
    private final List<PM> pms;
    private final int populationSize;
    private final int maxIterations;
    private final Random random;
    private final boolean verbose;

    private final List<Hippopotamus> population;

    private Hippopotamus bestHippo;
    private int currentIteration = 0;
    private int bestIteration = 0;

    /*
     * Used for fair comparison with Adaptive HO.
     * If provided, both algorithms start from the same population.
     */
    private final List<PlacementSolution> suppliedInitialPopulation;

    // ============================================================
    // Constructors
    // ============================================================

    public HippopotamusOptimization(
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
                true
        );
    }

    public HippopotamusOptimization(
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
                verbose
        );
    }

    public HippopotamusOptimization(
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
                true
        );
    }

    /*
     * Constructor for fair Standard HO vs Adaptive HO comparison.
     */
    public HippopotamusOptimization(
            List<VM> vms,
            List<PM> pms,
            int populationSize,
            int maxIterations,
            long seed,
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
        this.populationSize = populationSize;
        this.maxIterations = maxIterations;
        this.random = new Random(seed);
        this.verbose = verbose;

        this.population = new ArrayList<>();

        this.suppliedInitialPopulation =
                deepCopyPopulation(initialPopulation);

        this.bestHippo = null;
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
                    "Unable to create a feasible HO population."
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
        }

        for (int iteration = 1;
             iteration <= maxIterations;
             iteration++) {

            currentIteration = iteration;

            for (Hippopotamus hippo : population) {

                /*
                 * Phase 1:
                 * Movement toward best solution.
                 */
                PlacementSolution candidate =
                        phaseOneMovement(
                                hippo.getSolution()
                        );

                evaluateAndAccept(
                        hippo,
                        candidate
                );

                /*
                 * Phase 2:
                 * Defense.
                 */
                candidate =
                        phaseTwoDefense(
                                hippo.getSolution()
                        );

                evaluateAndAccept(
                        hippo,
                        candidate
                );

                /*
                 * Phase 3:
                 * Escape.
                 */
                candidate =
                        phaseThreeEscape(
                                hippo.getSolution()
                        );

                evaluateAndAccept(
                        hippo,
                        candidate
                );

                /*
                 * Mutation.
                 */
                if (random.nextDouble() < 0.20) {

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
    // Population initialization
    // ============================================================

    private void initializePopulation() {

        /*
         * FAIR EXPERIMENT:
         *
         * If MultiSeedExperiment supplies a population,
         * use exactly that population.
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
         * Normal random initialization.
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
    // Phase 1
    // ============================================================

    private PlacementSolution phaseOneMovement(
            PlacementSolution current) {

        int[] currentMapping =
                current.getVmToPm().clone();

        if (bestHippo == null) {
            return copySolution(current);
        }

        int[] bestMapping =
                bestHippo
                        .getSolution()
                        .getVmToPm();

        for (int i = 0;
             i < vms.size();
             i++) {

            if (random.nextDouble() < 0.50) {

                currentMapping[i] =
                        bestMapping[i];
            }
        }

        PlacementSolution candidate =
                new PlacementSolution(
                        currentMapping
                );

        return repairOrCurrent(
                candidate,
                current
        );
    }

    // ============================================================
    // Phase 2
    // ============================================================

    private PlacementSolution phaseTwoDefense(
            PlacementSolution current) {

        int[] mapping =
                current.getVmToPm().clone();

        int changes =
                Math.max(
                        1,
                        (int) Math.ceil(
                                vms.size() * 0.15
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
    // Phase 3
    // ============================================================

    private PlacementSolution phaseThreeEscape(
            PlacementSolution current) {

        int[] mapping =
                current.getVmToPm().clone();

        if (bestHippo != null) {

            int[] bestMapping =
                    bestHippo
                            .getSolution()
                            .getVmToPm();

            for (int i = 0;
                 i < vms.size();
                 i++) {

                if (random.nextDouble()
                        < 0.35) {

                    mapping[i] =
                            bestMapping[i];
                }
            }
        }

        if (random.nextDouble() < 0.30) {

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

        int vmIndex =
                random.nextInt(
                        vms.size()
                );

        mapping[vmIndex] =
                random.nextInt(
                        pms.size()
                );

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
         * Use the existing validator in your project.
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
         * IMPORTANT:
         *
         * Your existing PlacementRepair.repair()
         * expects:
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
                                copySolution(candidate),
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