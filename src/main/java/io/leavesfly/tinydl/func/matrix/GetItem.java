package io.leavesfly.tinydl.func.matrix;

import io.leavesfly.tinydl.func.Function;
import io.leavesfly.tinydl.ndarr.NdArray;

import java.util.Collections;
import java.util.List;

public class GetItem extends Function {
    private int[] rowSlices;
    private int[] colSlices;

    public GetItem(int[] _rowSlices, int[] _colSlices) {
        this.rowSlices = _rowSlices;
        this.colSlices = _colSlices;
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        return inputs[0].getItem(rowSlices, colSlices);
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        NdArray xGrad = NdArray.zeros(inputs[0].getValue().getShape()).addAt(rowSlices, colSlices, yGrad);
        return Collections.singletonList(xGrad);
    }

    @Override
    public int requireInputNum() {
        return 1;
    }
}
