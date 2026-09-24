package com.vmplacement.optimization;

import java.util.List;

public final class PlacementValidator {
    private PlacementValidator() {}

    public static boolean isFeasible(List<VM> vms, List<PM> pms, PlacementSolution solution) {
        if (solution == null || solution.getVmToPm().length != vms.size()) return false;
        double[] cpu = new double[pms.size()];
        double[] ram = new double[pms.size()];
        double[] storage = new double[pms.size()];
        int[] pes = new int[pms.size()];
        int[] mapping = solution.getVmToPm();
        for (int i=0;i<vms.size();i++) {
            int pm = mapping[i];
            if (pm < 0 || pm >= pms.size()) return false;
            VM vm = vms.get(i); PM host = pms.get(pm);
            cpu[pm] += vm.getMips() * vm.getCurrentCpuUtilization() / 100.0;
            ram[pm] += vm.getRamMb(); storage[pm] += vm.getStorageMb(); pes[pm] += vm.getPes();
            if (cpu[pm] > host.getTotalMips()+1e-9 || ram[pm] > host.getRamMb()+1e-9
                    || storage[pm] > host.getStorageMb()+1e-9 || pes[pm] > host.getPes()) return false;
        }
        return true;
    }
}
