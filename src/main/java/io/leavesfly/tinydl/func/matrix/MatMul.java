package io.leavesfly.tinydl.func.matrix;

import io.leavesfly.tinydl.func.Function;
import io.leavesfly.tinydl.ndarr.NdArray;

import java.util.Arrays;
import java.util.List;

/**
 * 矩阵内积
 */
public class MatMul extends Function {
    @Override
    public NdArray forward(NdArray... inputs) {
        NdArray x = inputs[0];
        NdArray w = inputs[1];

        return x.dot(w);
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        NdArray x = inputs[0].getValue();
        NdArray w = inputs[1].getValue();

        return Arrays.asList(yGrad.dot(w.transpose()), x.transpose().dot(yGrad));
    }

    @Override
    public int requireInputNum() {
        return 2;
    }
}
