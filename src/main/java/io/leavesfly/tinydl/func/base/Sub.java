package io.leavesfly.tinydl.func.base;

import io.leavesfly.tinydl.func.Function;
import io.leavesfly.tinydl.ndarr.NdArray;

import java.util.Arrays;
import java.util.List;

/**
 * 减法
 */
public class Sub extends Function {

    @Override
    public NdArray forward(NdArray... inputs) {
        return inputs[0].sub(inputs[1]);
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        return Arrays.asList(yGrad, yGrad.neg());
    }

    @Override
    public int requireInputNum() {
        return 2;
    }
}