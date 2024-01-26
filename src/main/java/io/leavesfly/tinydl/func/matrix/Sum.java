package io.leavesfly.tinydl.func.matrix;

import io.leavesfly.tinydl.func.Function;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;

import java.util.Collections;
import java.util.List;

/**
 * 取和
 */
public class Sum extends Function {

    private Shape inputShape;
    @Override
    public NdArray forward(NdArray... inputs) {
        inputShape = inputs[0].getShape();
        return inputs[0].sum();
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {

        return Collections.singletonList(yGrad.broadcastTo(inputShape));
    }

    @Override
    public int requireInputNum() {
        return 1;
    }
}
