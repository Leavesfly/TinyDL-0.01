package io.leavesfly.tinydl.func.math;

import io.leavesfly.tinydl.func.Function;
import io.leavesfly.tinydl.ndarr.NdArray;

import java.util.Collections;
import java.util.List;

public class ReLu extends Function {
    @Override
    public NdArray forward(NdArray... inputs) {
        return inputs[0].maximum(0f);
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        return Collections.singletonList(inputs[0].getValue().mask(0).mul(yGrad));
    }

    @Override
    public int requireInputNum() {
        return 1;
    }
}
