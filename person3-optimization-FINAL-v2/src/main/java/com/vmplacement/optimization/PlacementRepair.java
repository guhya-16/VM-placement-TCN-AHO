package com.vmplacement.optimization;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public final class PlacementRepair {
    private PlacementRepair() {}

    public static PlacementSolution randomFeasible(List<VM> vms, List<PM> pms, Random random) {
        List<Integer> order = new ArrayList<>();
        for (int i=0;i<vms.size();i++) order.add(i);
        Collections.shuffle(order, random);
        order.sort((a,b) -> Double.compare(requiredCpu(vms.get(b)), requiredCpu(vms.get(a))));
        int[] mapping = new int[vms.size()];
        java.util.Arrays.fill(mapping, -1);
        double[] cpu = new double[pms.size()];
        double[] ram = new double[pms.size()];
        double[] storage = new double[pms.size()];
        int[] pes = new int[pms.size()];
        for (int vi : order) {
            List<Integer> feasible = new ArrayList<>();
            for (int pi=0;pi<pms.size();pi++) {
                PM pm=pms.get(pi); VM vm=vms.get(vi);
                if (pes[pi]+vm.getPes() <= pm.getPes()
                        && cpu[pi]+requiredCpu(vm) <= pm.getTotalMips()+1e-9
                        && ram[pi]+vm.getRamMb() <= pm.getRamMb()+1e-9
                        && storage[pi]+vm.getStorageMb() <= pm.getStorageMb()+1e-9) feasible.add(pi);
            }
            if (feasible.isEmpty()) return null;
            int chosen = feasible.get(random.nextInt(feasible.size()));
            mapping[vi]=chosen;
            VM vm=vms.get(vi);
            pes[chosen]+=vm.getPes(); cpu[chosen]+=requiredCpu(vm);
            ram[chosen]+=vm.getRamMb(); storage[chosen]+=vm.getStorageMb();
        }
        PlacementSolution solution = new PlacementSolution(mapping);
        return PlacementValidator.isFeasible(vms,pms,solution) ? solution : null;
    }

    public static PlacementSolution repair(List<VM> vms, List<PM> pms, PlacementSolution candidate,
                                           Random random) {
        if (PlacementValidator.isFeasible(vms,pms,candidate)) return candidate.copy();
        return randomFeasible(vms,pms,random);
    }

    private static double requiredCpu(VM vm) {
        return vm.getMips()*vm.getCurrentCpuUtilization()/100.0;
    }
}
