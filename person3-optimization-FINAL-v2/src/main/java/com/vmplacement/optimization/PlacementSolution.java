package com.vmplacement.optimization;

import java.util.Arrays;

public final class PlacementSolution {
    private final int[] vmToPm;

    public PlacementSolution(int[] vmToPm) {
        if (vmToPm == null) throw new IllegalArgumentException("Placement cannot be null");
        this.vmToPm = vmToPm.clone();
    }

    public int[] getVmToPm() { return vmToPm.clone(); }
    public int getPmIndex(int vmIndex) { return vmToPm[vmIndex]; }
    public PlacementSolution copy() { return new PlacementSolution(vmToPm); }

    @Override public String toString() { return Arrays.toString(vmToPm); }
    @Override public boolean equals(Object o) {
        return o instanceof PlacementSolution other && Arrays.equals(vmToPm, other.vmToPm);
    }
    @Override public int hashCode() { return Arrays.hashCode(vmToPm); }
}
