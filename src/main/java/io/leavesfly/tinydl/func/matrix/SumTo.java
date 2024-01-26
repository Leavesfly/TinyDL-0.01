package io.leavesfly.tinydl.func.matrix;

import io.leavesfly.tinydl.func.Function;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;

import java.util.Collections;
import java.util.List;

/**
 * 累加和到
 */
public class SumTo extends Function {

    private Shape shape;
    private Shape inputShape;

    public SumTo(Shape _shape) {
        this.shape = _shape;
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        inputShape = inputs[0].getShape();
        return inputs[0].broadcastTo(shape);

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
