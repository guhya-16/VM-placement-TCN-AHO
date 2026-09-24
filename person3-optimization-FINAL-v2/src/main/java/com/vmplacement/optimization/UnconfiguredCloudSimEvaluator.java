package com.vmplacement.optimization;

/** Fails closed until Person 2 supplies the real CloudSim implementation. */
public final class UnconfiguredCloudSimEvaluator implements CloudSimEvaluator {
    @Override public CandidateEvaluationOutput evaluate(CandidateEvaluationInput input) {
        throw new IllegalStateException(
                "No real Person 2 CloudSim evaluator is configured. Implement the adapter before running final optimization.");
    }
}
