package com.vmplacement.optimization;

import java.util.Arrays;

public class Hippopotamus {

    private PlacementSolution solution;
    private double fitness;

    public Hippopotamus(PlacementSolution solution) {
        this.solution = solution;
        this.fitness = Double.MAX_VALUE;
    }

    public PlacementSolution getSolution() {
        return solution;
    }

    public void setSolution(PlacementSolution solution) {
        this.solution = solution;
    }

    public double getFitness() {
        return fitness;
    }

    public void setFitness(double fitness) {
        this.fitness = fitness;
    }

    /*
     * Create an independent copy of this hippopotamus.
     */
    public Hippopotamus copy() {

        int[] placement =
                solution.getVmToPm().clone();

        Hippopotamus copy =
                new Hippopotamus(
                        new PlacementSolution(placement)
                );

        copy.setFitness(fitness);

        return copy;
    }

    @Override
    public String toString() {
        return "Solution: " + solution +
               ", Fitness: " + fitness;
    }
}