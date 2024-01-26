package io.leavesfly.tinydl.func.math;

import io.leavesfly.tinydl.func.Function;
import io.leavesfly.tinydl.ndarr.NdArray;

import java.util.Collections;
import java.util.List;

/**
 * 在最小最大中的值
 */
public class Clip extends Function {
    private float min;
    private float max;

    public Clip(float min, float max) {
        this.min = min;
        this.max = max;
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        return inputs[0].clip(min, max);
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        NdArray x = inputs[0].getValue();
        if (x.isLarger(NdArray.like(x.getShape(), min)) && !x.isLarger(NdArray.like(x.getShape(), max))) {
            return Collections.singletonList(yGrad);
        }
        return Collections.singletonList(yGrad.neg());
    }

    @Override
    public int requireInputNum() {
        return 1;
    }
}
