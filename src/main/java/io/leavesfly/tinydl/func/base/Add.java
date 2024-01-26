package io.leavesfly.tinydl.func.base;

import io.leavesfly.tinydl.func.Function;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;

import java.util.Arrays;
import java.util.List;

/**
 * 加法
 */
public class Add extends Function {

    private Shape inputShape;

    @Override
    public NdArray forward(NdArray... inputs) {

        if (!inputs[1].getShape().equals(inputs[0].getShape())) {
            inputShape = inputs[1].getShape();
            return inputs[0].add(inputs[1].broadcastTo(inputs[0].getShape()));
        } else {
            return inputs[0].add(inputs[1]);
        }
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        return Arrays.asList(yGrad, inputShape == null ? yGrad : yGrad.sumTo(inputShape));
    }

    @Override
    public int requireInputNum() {
        return 2;
    }
}
