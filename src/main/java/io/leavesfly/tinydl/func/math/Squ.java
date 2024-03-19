package io.leavesfly.tinydl.func.math;

import io.leavesfly.tinydl.func.Function;
import io.leavesfly.tinydl.ndarr.NdArray;

import java.util.Collections;
import java.util.List;

/**
 * 平方
 */
public class Squ extends Function {

    @Override
    public NdArray forward(NdArray... inputs) {
        return inputs[0].pow(2);
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        NdArray x = inputs[0].getValue();
        return Collections.singletonList(x.mulNum(2).mul(yGrad));
    }

    @Override
    public int requireInputNum() {
        return 1;
    }
}
