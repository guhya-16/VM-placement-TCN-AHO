package com.vmplacement.optimization;

import java.util.List;

public final class AdaptiveSignalCalculator {
    private final double volatilityWeight;
    private final double riskWeight;
    private final double minExploration;
    private final double maxExploration;

    public AdaptiveSignalCalculator(double volatilityWeight, double riskWeight,
                                    double minExploration, double maxExploration) {
        if (volatilityWeight < 0 || riskWeight < 0 || volatilityWeight + riskWeight <= 0)
            throw new IllegalArgumentException("Adaptive weights must be non-negative and not both zero");
        this.volatilityWeight = volatilityWeight;
        this.riskWeight = riskWeight;
        this.minExploration = minExploration;
        this.maxExploration = maxExploration;
    }

    public Result calculate(List<VM> vms) {
        double minV=Double.POSITIVE_INFINITY,maxV=Double.NEGATIVE_INFINITY;
        double minR=Double.POSITIVE_INFINITY,maxR=Double.NEGATIVE_INFINITY;
        for (VM vm:vms) { minV=Math.min(minV,vm.getVolatilityScore()); maxV=Math.max(maxV,vm.getVolatilityScore());
            minR=Math.min(minR,vm.getRiskScore()); maxR=Math.max(maxR,vm.getRiskScore()); }
        double avgV=0,avgR=0;
        for (VM vm:vms) { avgV+=normalize(vm.getVolatilityScore(),minV,maxV); avgR+=normalize(vm.getRiskScore(),minR,maxR); }
        avgV/=vms.size(); avgR/=vms.size();
        double total=volatilityWeight+riskWeight;
        double signal=(volatilityWeight*avgV+riskWeight*avgR)/total;
        double exploration=clamp(minExploration+signal*(maxExploration-minExploration),minExploration,maxExploration);
        String mode=signal<0.10?"EXPLOITATION":signal<0.25?"BALANCED":"EXPLORATION";
        return new Result(avgV,avgR,signal,exploration,mode);
    }

    private static double normalize(double x,double min,double max) { return max-min<1e-12?0.0:clamp((x-min)/(max-min),0,1); }
    private static double clamp(double x,double a,double b){return Math.max(a,Math.min(b,x));}

    public record Result(double workloadVariation,double predictionRisk,double adaptiveSignal,
                         double explorationProbability,String mode) {}
}
