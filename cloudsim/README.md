# VM Placement — CloudSim Plus Starter

Uses real PM1/PM2/PM3 host specs and real VM types, with a PLACEHOLDER
workload (UtilizationModelDynamic) and the default VmAllocationPolicySimple.

Next steps:
1. Scale createDatacenter() from 6 hosts up to 20-30.
2. Replace UtilizationModelDynamic in createCloudlets() with a custom
   UtilizationModel reading real Bitbrains CPU trace data.
3. Implement PABFD as a custom VmAllocationPolicy, with a linear power
   model per host using each PM's Pmax/Pmin values.
4. Collect energy, active PM count, SLA violations, migrations as results.
