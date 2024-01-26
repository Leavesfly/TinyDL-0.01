package io.leavesfly.tinydl.func.base;

import io.leavesfly.tinydl.func.Function;
import io.leavesfly.tinydl.ndarr.NdArray;

import java.util.Collections;
import java.util.List;

/**
 * 取反
 */
public class Neg extends Function {
    @Override
    public NdArray forward(NdArray... inputs) {
        return inputs[0].neg();
    }

    /**
     * @param yGrad
     * @return
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        return Collections.singletonList(yGrad.neg());
    }

    @Override
    public int requireInputNum() {
        return 1;
    }
}
