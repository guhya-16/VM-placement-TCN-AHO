package com.vmplacement.optimization;

import java.util.Arrays;

public class PlacementSolution {

    private int[] vmToPm;

    public PlacementSolution(int[] vmToPm) {
        this.vmToPm = vmToPm;
    }

    public int[] getVmToPm() {
        return vmToPm;
    }

    public void setVmToPm(int[] vmToPm) {
        this.vmToPm = vmToPm;
    }

    public PlacementSolution copy() {
        return new PlacementSolution(
                vmToPm.clone()
        );
    }

    @Override
    public String toString() {
        return Arrays.toString(vmToPm);
    }
}