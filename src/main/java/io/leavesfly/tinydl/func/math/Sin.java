package io.leavesfly.tinydl.func.math;

import io.leavesfly.tinydl.func.Function;
import io.leavesfly.tinydl.ndarr.NdArray;

import java.util.Collections;
import java.util.List;

/**
 * 以sin函数
 */
public class Sin extends Function {
    @Override
    public NdArray forward(NdArray... inputs) {
        return inputs[0].sin();
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        return Collections.singletonList(inputs[0].getValue().cos().mul(yGrad.transpose()));
    }

    @Override
    public int requireInputNum() {
        return 1;
    }
}
