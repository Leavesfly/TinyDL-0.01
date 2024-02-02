package io.leavesfly.tinydl.func.math;


import io.leavesfly.tinydl.ndarr.NdArray;

/**
 * 最小值
 */
public class Min extends Max {

    public Min(int _axis, boolean _keepdims) {
        super(_axis, _keepdims);
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        if (keepdims) {
            return inputs[0].min(axis).broadcastTo(inputs[0].getShape());
        }
        return inputs[0].min(axis);
    }

}
