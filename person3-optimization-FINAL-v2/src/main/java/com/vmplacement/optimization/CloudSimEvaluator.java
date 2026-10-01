package com.vmplacement.optimization;

/**
 * Person 3 calls this synchronously for every candidate. Person 2 implements it
 * using isolated CloudSim evaluation and returns the exact candidate-output metrics.
 */
public interface CloudSimEvaluator {
    CandidateEvaluationOutput evaluate(CandidateEvaluationInput input);
}
