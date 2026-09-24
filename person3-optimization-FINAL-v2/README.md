# Person 3 — Dynamic Person 1 Handoff + Adaptive Hippopotamus Optimization

This is the clean Person 3 optimization module. It is designed around the agreed three handoffs and does not contain Person 2's CloudSim implementation.

## The three handoffs

### 1. Person 1 → Person 3
Person 1 supplies `input/risk_state.csv` with the frozen 15-column schema:

`vm_id,decision_timestamp,current_cpu,predicted_cpu_t5m,predicted_cpu_t10m,predicted_cpu_t15m,predicted_cpu_t30m,predicted_cpu_t45m,predicted_cpu_t60m,predicted_mean_cpu,predicted_peak_cpu,predicted_std_cpu,volatility_score,risk_score,risk_state`

The program groups rows by the actual `decision_timestamp`. There is no hardcoded number of timestamps, rows, or VMs.

### 2. Person 3 → CloudSim (every candidate)
For every candidate evaluated during the configured AHO iterations, Person 3 writes rows to:

`output/person3_to_cloudsim_candidate_input.csv`

Exact header:

`epoch,iteration,candidate_id,timestamp,vm_id,pm_id,current_cpu_utilization,predicted_cpu_utilization,previous_pm_id,vm_mips,vm_pe_count,vm_ram,vm_storage,pm_mips_per_pe,pm_pe_count,pm_ram,pm_storage,pm_max_power,pm_static_power,sla_threshold`

`predicted_cpu_utilization` is the Person 1 `predicted_mean_cpu` value used as the representative future utilization in this contract. Person 2 may additionally use the other forecast/risk fields from the epoch state if the team extends the contract; no field is silently invented.

CloudSim returns one row per candidate through the Java `CloudSimEvaluator` interface, and Person 3 records it in:

`output/cloudsim_to_person3_candidate_output.csv`

Exact header:

`epoch,iteration,candidate_id,feasible,energy_wh,sla_violations,active_pms,mean_cpu_utilization,placement_changes_from_previous_state,migration_cost`

The candidate output is used immediately by AHO to calculate fitness. A hypothetical candidate-to-candidate transition is never committed as a real migration.

### 3. Person 3 → Person 2 (after AHO)
After the configured iteration budget, only the selected best placement is written to:

`output/person3_to_person2_final_placement.csv`

Exact header:

`epoch,timestamp,vm_id,pm_id,current_cpu_utilization,predicted_cpu_utilization,prediction_risk,risk_state,best_iteration`

This is the committed placement for the epoch. The next epoch uses this committed mapping as its previous state.

## Important resource ownership

There are deliberately **no `vm_profiles.csv` or `pm_profiles.csv` files** in this version.

Person 3 does not invent or duplicate infrastructure configuration. Person 2 / the CloudSim side supplies the authoritative VM resource specifications and PM specifications through an `InfrastructureProvider` implementation. This provider produces an `InfrastructureSnapshot` for each decision epoch.

That snapshot supplies:
- VM MIPS, PEs, RAM, storage
- PM MIPS/PE, PEs, RAM, storage, maximum power, static power

Those values are then copied into the candidate-evaluation input contract.

## Loop semantics

For each actual decision timestamp:

1. Read all active VMs for that timestamp.
2. Obtain the authoritative infrastructure snapshot.
3. Build one `EpochState`.
4. Build a feasible AHO population of the configured size.
5. **Iteration 1:** evaluate the initial population — exactly `population_size` candidates.
6. **Iterations 2..N:** generate one candidate per population member and evaluate exactly `population_size` candidates.
7. Compare each candidate with its parent and retain the better one.
8. Track the best candidate across all iterations.
9. After iteration N, write only that best placement to the final-placement handoff.
10. Commit that placement as the previous state for the next timestamp.

With the default population 20 and 50 iterations, the loop performs exactly 20 × 50 = 1,000 candidate evaluations per epoch (assuming every CloudSim call succeeds). It does not evaluate a candidate population twice for normalization.

## Candidate isolation rule

Every candidate evaluation receives the same previous **committed** placement for the epoch. Person 2's CloudSim adapter must create an isolated temporary evaluation. Candidate A must not alter the starting state used for Candidate B.

Only the selected best candidate becomes the real next state.

## Migration semantics

A migration is a VM present in both consecutive committed states whose PM changes. A new VM is not a migration, and a departed VM is not a migration. Hypothetical AHO candidates are never counted as real migrations.

## CloudSim adapter

Implement `CloudSimEvaluator` on the Person 2 side:

```java
public final class Person2CloudSimEvaluator implements CloudSimEvaluator {
    @Override
    public CandidateEvaluationOutput evaluate(CandidateEvaluationInput input) {
        // isolated CloudSim evaluation
    }
}
```

Also implement `InfrastructureProvider` so Person 3 can obtain the same authoritative VM/PM specifications used by CloudSim.

The current `UnconfiguredCloudSimEvaluator` and `UnconfiguredInfrastructureProvider` intentionally fail closed. This prevents fabricated energy or infrastructure values from becoming final evidence.

## No hardcoded dataset counts

There is no hardcoded 997, 6915, 1250, or any other dataset-specific row/epoch/VM count in the Java loop. The only fixed numbers in `application.properties` are algorithm/configuration choices such as population size and iteration budget.

## CloudSim integration status

The Person 3 optimization module has been structurally validated before connecting the real CloudSim evaluator. The validation focused on verifying the optimization workflow, dynamic decision-epoch handling, and handoff contracts.

The validation covered:

* dynamic decision-epoch processing
* dynamic active-VM handling
* feasible population generation
* population × iteration evaluation loops
* candidate evaluation handoff contracts
* previous-placement propagation
* migration bookkeeping
* best-candidate selection
* final-placement handoff

The current execution path is designed to fail closed until the real CloudSim-backed implementations are connected. This prevents fabricated infrastructure, energy, or SLA values from being treated as project results.

The current interfaces are:

* `InfrastructureProvider` — supplies authoritative VM and PM resource information.
* `CloudSimEvaluator` — evaluates candidate placements through the CloudSim integration.
* `UnconfiguredInfrastructureProvider` — fail-closed implementation used until the real infrastructure provider is connected.
* `UnconfiguredCloudSimEvaluator` — fail-closed implementation used until the real CloudSim evaluator is connected.

Once Person 2 connects the real CloudSim implementations, the core AHO workflow can use the same interfaces without changing the optimization architecture.


### Fitness normalization

Fitness is normalized against configured scales instead of the current candidate batch's min/max values. This keeps fitness values comparable across AHO iterations. The scale values in `application.properties` are test values and must be calibrated/replaced for the final CloudSim experiment.
