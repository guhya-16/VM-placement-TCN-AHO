# Person 3 ↔ Person 2 / CloudSim contract

## Candidate input: Person 3 → CloudSim

Person 3 creates one candidate input object for every candidate evaluated. It also appends the exact rows to `output/person3_to_cloudsim_candidate_input.csv`.

```csv
epoch,iteration,candidate_id,timestamp,vm_id,pm_id,current_cpu_utilization,predicted_cpu_utilization,previous_pm_id,vm_mips,vm_pe_count,vm_ram,vm_storage,pm_mips_per_pe,pm_pe_count,pm_ram,pm_storage,pm_max_power,pm_static_power,sla_threshold
```

One candidate can have multiple VM rows. The identity fields (`epoch`, `iteration`, `candidate_id`, `timestamp`) are repeated for each row so a separate CloudSim process can reconstruct the candidate.

## Candidate output: CloudSim → Person 3

CloudSim returns exactly one result for the candidate:

```csv
epoch,iteration,candidate_id,feasible,energy_wh,sla_violations,active_pms,mean_cpu_utilization,placement_changes_from_previous_state,migration_cost
```

The Java adapter returns this same record to AHO synchronously, and Person 3 records it in `output/cloudsim_to_person3_candidate_output.csv`.

Definitions must be agreed by the team and implemented in CloudSim. Person 3 must not replace missing values with estimated energy or fabricated metrics.

## Final placement: Person 3 → Person 2

After the AHO iteration budget for an epoch, Person 3 writes only the selected best placement:

```csv
epoch,timestamp,vm_id,pm_id,current_cpu_utilization,predicted_cpu_utilization,prediction_risk,risk_state,best_iteration
```

The output contains one row per active VM in the selected placement.

## Infrastructure

The candidate input contains VM/PM resource fields, but Person 3 does not own a duplicate CSV manifest for them. Person 2 supplies an `InfrastructureProvider` implementation backed by the authoritative CloudSim configuration. The provider returns an `InfrastructureSnapshot` for each epoch.

## Isolation and state

- Every candidate in an epoch starts from the same previous committed state.
- Candidate A must not mutate the state used by Candidate B.
- Candidate-to-candidate PM changes are hypothetical and do not count as real migrations.
- Only the best candidate is committed.
- The committed mapping becomes `previous_pm_id` for the next epoch.
