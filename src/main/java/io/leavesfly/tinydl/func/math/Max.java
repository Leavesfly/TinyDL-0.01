package io.leavesfly.tinydl.func.math;

import io.leavesfly.tinydl.func.Function;
import io.leavesfly.tinydl.ndarr.NdArray;

import java.util.Collections;
import java.util.List;

/**
 * 最大值
 */
public class Max extends Function {
    int axis;
    boolean keepdims;

    public Max(int _axis, boolean _keepdims) {
        this.axis = _axis;
        this.keepdims = _keepdims;
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        if (keepdims) {
            return inputs[0].max(axis).broadcastTo(inputs[0].getShape());
        }
        return inputs[0].max(axis);
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        NdArray x = inputs[0].getValue();
        NdArray y = output.getValue();

        if (!keepdims) {
            yGrad = yGrad.broadcastTo(x.getShape());
            y = y.broadcastTo(x.getShape());
        }
        return Collections.singletonList(x.eq(y).mul(yGrad));
    }

    @Override
    public int requireInputNum() {
        return 1;
    }
}
