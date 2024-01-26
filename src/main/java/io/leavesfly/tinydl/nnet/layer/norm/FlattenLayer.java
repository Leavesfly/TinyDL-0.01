package io.leavesfly.tinydl.nnet.layer.norm;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.Layer;

import java.util.Collections;
import java.util.List;

/**
 * 将矩阵打平
 */
public class FlattenLayer extends Layer {

    public FlattenLayer(String _name, Shape _xInputShape, Shape _yOutputShape) {
        super(_name, _xInputShape, _yOutputShape);
    }

    @Override
    public NdArray forward(NdArray... inputs) {

        return inputs[0].flatten();
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        return Collections.singletonList(yGrad.reshape(xInputShape));
    }

    @Override
    public int requireInputNum() {
        return 1;
    }

    @Override
    public void init() {

        int outputSize = 1;
        for (int i = 0; i < xInputShape.dimension.length; i++) {
            outputSize *= xInputShape.dimension[i];
        }
        yOutputShape = new Shape(1, outputSize);
    }

    @Override
    public Variable forward(Variable... inputs) {
        return this.call(inputs[0]);
    }
}
