package com.vmplacement.optimization;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/** Risk-adaptive HO. Exactly populationSize candidate evaluations are performed per configured iteration. */
public final class AdaptiveHippopotamusOptimization {
    private final EpochState epoch; private final int populationSize,maxIterations; private final Random random;
    private final boolean verbose; private final FitnessFunction fitnessFunction; private final AdaptiveSignalCalculator signalCalculator;
    private final double mutationBase,mutationRange; private final CandidateEvaluationBridge bridge;
    private final List<Hippopotamus> population=new ArrayList<>(); private Hippopotamus best; private int bestIteration=0;
    private AdaptiveSignalCalculator.Result signal;

    public AdaptiveHippopotamusOptimization(EpochState epoch,int populationSize,int maxIterations,long seed,boolean verbose,
            CandidateEvaluationBridge bridge,FitnessFunction fitnessFunction,AdaptiveSignalCalculator signalCalculator,
            double mutationBase,double mutationRange){
        if(epoch.getActiveVms().isEmpty())throw new IllegalArgumentException("Epoch has no active VMs: "+epoch.getTimestamp());
        if(populationSize<=0||maxIterations<=0)throw new IllegalArgumentException("Population/iterations must be positive");
        this.epoch=epoch;this.populationSize=populationSize;this.maxIterations=maxIterations;this.random=new Random(seed);
        this.verbose=verbose;this.bridge=bridge;this.fitnessFunction=fitnessFunction;this.signalCalculator=signalCalculator;
        this.mutationBase=mutationBase;this.mutationRange=mutationRange;this.signal=signalCalculator.calculate(epoch.getActiveVms());
    }

    public PlacementSolution optimize(){
        initialize();
        // Iteration 1 evaluates the initial population. Iterations 2..N evaluate the newly generated population.
        evaluatePopulation(population,1);
        updateBest(1);
        for(int it=2;it<=maxIterations;it++){
            List<Hippopotamus> candidates=new ArrayList<>(population.size());
            for(int i=0;i<population.size();i++) candidates.add(new Hippopotamus(generateCandidate(population.get(i).getSolution())));
            evaluatePopulation(candidates,it);
            for(int i=0;i<population.size();i++) if(candidates.get(i).getFitness()<population.get(i).getFitness()) population.set(i,candidates.get(i));
            updateBest(it);
            if(verbose)System.out.printf("timestamp=%d iteration=%d/%d bestFitness=%.8f mode=%s exploration=%.4f%n",
                    epoch.getTimestamp(),it,maxIterations,best.getFitness(),signal.mode(),signal.explorationProbability());
        }
        return best.getSolution().copy();
    }

    private void initialize(){
        population.clear(); int attempts=0,maxAttempts=Math.max(populationSize*50,populationSize);
        while(population.size()<populationSize && attempts++<maxAttempts){
            PlacementSolution s=PlacementRepair.randomFeasible(epoch.getActiveVms(),epoch.getPms(),random);
            if(s!=null)population.add(new Hippopotamus(s));
        }
        if(population.size()<populationSize)throw new IllegalStateException("Could not build the requested feasible population: requested="+populationSize+", built="+population.size()+", timestamp="+epoch.getTimestamp());
    }
    private boolean contains(PlacementSolution s){for(Hippopotamus h:population)if(h.getSolution().equals(s))return true;return false;}
    private void evaluatePopulation(List<Hippopotamus> hs,int iteration){
        List<CloudSimEvaluationResult> results=new ArrayList<>(hs.size());
        for(int i=0;i<hs.size();i++) results.add(bridge.evaluate(currentEpochIndex,iteration,i+1,epoch,hs.get(i).getSolution()));
        List<Double> scores=fitnessFunction.score(results);
        for(int i=0;i<hs.size();i++){hs.get(i).setEvaluation(results.get(i));hs.get(i).setFitness(scores.get(i));}
    }
    private void updateBest(int iteration){for(Hippopotamus h:population)if(best==null||h.getFitness()<best.getFitness()){best=h;bestIteration=iteration;}}
    private PlacementSolution generateCandidate(PlacementSolution current){
        int[] m=current.getVmToPm(); double exploration=signal.explorationProbability();
        if(random.nextDouble()<exploration){int changes=Math.max(1,(int)Math.ceil(m.length*(0.10+0.50*signal.adaptiveSignal())));for(int c=0;c<changes;c++)m[random.nextInt(m.length)]=random.nextInt(epoch.getPms().size());}
        else if(best!=null){double strength=0.70+0.25*(1-signal.adaptiveSignal());int[] bm=best.getSolution().getVmToPm();for(int i=0;i<m.length;i++)if(random.nextDouble()<strength)m[i]=bm[i];}
        int defenseChanges=Math.max(1,(int)Math.ceil(m.length*(0.05+0.20*signal.adaptiveSignal())));
        for(int c=0;c<defenseChanges;c++)if(random.nextDouble()<0.5)m[random.nextInt(m.length)]=random.nextInt(epoch.getPms().size());
        double mutationProbability=mutationBase+mutationRange*signal.adaptiveSignal();
        if(random.nextDouble()<mutationProbability)m[random.nextInt(m.length)]=random.nextInt(epoch.getPms().size());
        PlacementSolution repaired=PlacementRepair.repair(epoch.getActiveVms(),epoch.getPms(),new PlacementSolution(m),random);
        return repaired==null?current.copy():repaired;
    }
    // Set by Main before optimization so candidate handoff rows carry a dynamic epoch number.
    private int currentEpochIndex;
    public void setEpochIndex(int epochIndex){if(epochIndex<=0)throw new IllegalArgumentException("epochIndex must be positive");this.currentEpochIndex=epochIndex;}
    public double getBestFitness(){return best==null?Double.POSITIVE_INFINITY:best.getFitness();}
    public PlacementSolution getBestSolution(){return best==null?null:best.getSolution().copy();}
    public AdaptiveSignalCalculator.Result getSignal(){return signal;}
    public CloudSimEvaluationResult getBestEvaluation(){return best==null?null:best.getEvaluation();}
    public int getBestIteration(){return bestIteration;}
}
