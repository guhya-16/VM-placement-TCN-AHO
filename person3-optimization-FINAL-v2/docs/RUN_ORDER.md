# Run order

1. Place Person 1's actual `risk_state.csv` at `input/risk_state.csv`.
2. `CSVWorkloadReader` validates the 15-column handoff and groups rows by actual timestamp.
3. For each timestamp, `InfrastructureProvider.snapshot(epoch)` supplies authoritative VM/PM configuration from the CloudSim side.
4. `DecisionEpochConverter` combines Person 1 workload/risk values with those resource specifications.
5. AHO creates a current-resource-feasible population.
6. For each iteration, Person 3 builds candidate handoff rows and calls `CloudSimEvaluator` synchronously.
7. Returned CloudSim metrics are recorded and scored by `FitnessFunction`.
8. After the final iteration, only the best candidate is written to `person3_to_person2_final_placement.csv`.
9. That selected mapping becomes the previous committed state for the next timestamp.
10. No dataset-size constant controls any loop.

The default entry point intentionally fails closed until both the real `InfrastructureProvider` and `CloudSimEvaluator` are connected.
