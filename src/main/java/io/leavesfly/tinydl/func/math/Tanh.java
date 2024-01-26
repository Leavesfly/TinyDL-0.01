package io.leavesfly.tinydl.func.math;

import io.leavesfly.tinydl.func.Function;
import io.leavesfly.tinydl.ndarr.NdArray;

import java.util.Collections;
import java.util.List;

/**
 * Tanh函数
 */
public class Tanh extends Function {
    @Override
    public NdArray forward(NdArray... inputs) {
        return inputs[0].tanh();
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        NdArray outputValue = output.getValue();
        return Collections.singletonList(
                yGrad.mul(NdArray.ones(outputValue.getShape()).sub(outputValue.square())));
    }

    @Override
    public int requireInputNum() {
        return 1;
    }
}
