# Person 3 Optimization - Final Clean Version

## Final pipeline
Bitbrains RiskState -> Decision Epoch -> TCN prediction information -> workload/risk signal
-> Standard HO / Adaptive HO -> VM-to-PM placement.

## Important final fixes
1. `DecisionEpochConverter.java`
   - Risk-state CPU values are stored as fractions (0.0-1.0).
   - The optimizer uses percentage utilization (0-100).
   - Conversion is now performed at the module boundary.
   - Prediction mean/peak/std are converted consistently.

2. `FitnessFunction.java`
   - Current resource feasibility remains a hard constraint.
   - Predicted workload is a soft optimization signal.
   - Predicted mean utilization above 80% receives an SLA penalty.
   - Predicted peak utilization above 90% receives an additional risk penalty.
   - Predicted load imbalance, energy, and active PM count remain part of the objective.
   - The lightweight power model is for optimization only; final energy evaluation belongs to CloudSim Plus.

3. `Main.java`
   - Standard HO and Adaptive HO use the same initial population for each decision epoch.
   - Separate deterministic seeds are used for reproducibility.
   - Feasibility means current resource feasibility; predicted overload is not treated as a hard rejection.
   - PM definitions use representative CloudSim host classes from Person 2's environment.

4. Standard HO vs Adaptive HO
   - The algorithms are compared fairly from the same initial population.
   - A lower fitness value is better.
   - Adaptive HO is not artificially forced to outperform Standard HO.

## Verification
The complete Java source was compiled successfully with `javac` and the main experiment was executed successfully over all 997 decision epochs in the supplied `risk_state.csv`.

The generated `experiment_results.csv` is included as the latest local experiment output.

## Run with the user's JDK 25 + Maven wrapper
From this folder:

```powershell
$env:JAVA_HOME = "C:\Program Files\Java\jdk-25"
.\mvnw.cmd clean compile
java -cp target\classes com.vmplacement.optimization.Main
```

For the synthetic multi-seed comparison:

```powershell
java -cp target\classes com.vmplacement.optimization.MultiSeedExperiment
```
