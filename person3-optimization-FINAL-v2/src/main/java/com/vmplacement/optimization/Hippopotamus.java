package com.vmplacement.optimization;

public final class Hippopotamus {
    private PlacementSolution solution;
    private double fitness=Double.POSITIVE_INFINITY;
    private CloudSimEvaluationResult evaluation;

    public Hippopotamus(PlacementSolution solution){this.solution=solution.copy();}
    public PlacementSolution getSolution(){return solution;}
    public void setSolution(PlacementSolution s){solution=s.copy();}
    public double getFitness(){return fitness;}
    public void setFitness(double f){fitness=f;}
    public CloudSimEvaluationResult getEvaluation(){return evaluation;}
    public void setEvaluation(CloudSimEvaluationResult e){evaluation=e;}
}
