package io.leavesfly.tinydl.func.math;

import io.leavesfly.tinydl.func.Function;
import io.leavesfly.tinydl.ndarr.NdArray;

import java.util.Collections;
import java.util.List;

public class Sigmoid extends Function {
    @Override
    public NdArray forward(NdArray... inputs) {
        return inputs[0].sigmoid();
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        NdArray y = getOutput().getValue();
        return Collections.singletonList(yGrad.mul(y).mul(NdArray.ones(y.getShape()).sub(y)));
    }

    @Override
    public int requireInputNum() {
        return 1;
    }
}
