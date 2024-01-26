package io.leavesfly.tinydl.func.matrix;

import io.leavesfly.tinydl.func.Function;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;

import java.util.Collections;
import java.util.List;

/**
 * 矩阵变形
 */
public class Reshape extends Function {

    private Shape shape;
    private Shape inputShape;

    public Reshape(Shape _shape) {
        this.shape = _shape;
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        inputShape = inputs[0].getShape();
        return inputs[0].reshape(shape);

    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        return Collections.singletonList(yGrad.reshape(inputShape));
    }

    @Override
    public int requireInputNum() {
        return 0;
    }
}
