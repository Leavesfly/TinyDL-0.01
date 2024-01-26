package io.leavesfly.tinydl.func.math;

import io.leavesfly.tinydl.func.Function;
import io.leavesfly.tinydl.ndarr.NdArray;

import java.util.Collections;
import java.util.List;

/**
 * 以e为底的对数
 */
public class Log extends Function {
    @Override
    public NdArray forward(NdArray... inputs) {
        return inputs[0].log();
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        return Collections.singletonList(yGrad.div(inputs[0].getValue()));
    }

    @Override
    public int requireInputNum() {
        return 1;
    }
}
