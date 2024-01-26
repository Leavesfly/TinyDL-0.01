package io.leavesfly.tinydl.func.base;

import io.leavesfly.tinydl.func.Function;
import io.leavesfly.tinydl.ndarr.NdArray;

import java.util.Arrays;
import java.util.List;

public class Div extends Function {

    @Override
    public NdArray forward(NdArray... inputs) {
        return inputs[0].div(inputs[1]);
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {

        NdArray ndArray0 = inputs[0].getValue();
        NdArray ndArray1 = inputs[1].getValue();

        return Arrays.asList(yGrad.div(ndArray1), yGrad.mul(ndArray0.neg().div(ndArray1.square())));
    }

    @Override
    public int requireInputNum() {
        return 2;
    }
}