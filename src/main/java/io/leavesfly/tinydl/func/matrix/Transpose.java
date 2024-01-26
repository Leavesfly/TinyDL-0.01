package io.leavesfly.tinydl.func.matrix;

import io.leavesfly.tinydl.func.Function;
import io.leavesfly.tinydl.ndarr.NdArray;

import java.util.Collections;
import java.util.List;

/**
 * 转置
 */
public class Transpose extends Function {
    @Override
    public NdArray forward(NdArray... inputs) {
        return inputs[0].transpose();
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        return Collections.singletonList(yGrad.transpose());
    }

    @Override
    public int requireInputNum() {
        return 1;
    }
}
