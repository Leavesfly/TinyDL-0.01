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
        // 动态计算输出形状，保持batch维度
        NdArray input = inputs[0];
        int batchSize = input.shape.dimension[0];
        int flattenedSize = input.shape.size() / batchSize;
        Shape dynamicOutputShape = new Shape(batchSize, flattenedSize);
        return input.reshape(dynamicOutputShape);
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        return Collections.singletonList(yGrad.reshape(inputShape));
    }

    @Override
    public int requireInputNum() {
        return 1;
    }

    @Override
    public void init() {
        // 计算除了batch维度以外的所有维度的乘积
        int outputSize = 1;
        for (int i = 1; i < inputShape.dimension.length; i++) {
            outputSize *= inputShape.dimension[i];
        }
        // 输出形状：保持batch维度，其他维度展平
        outputShape = new Shape(inputShape.dimension[0], outputSize);
    }

    @Override
    public Variable layerForward(Variable... inputs) {
        return this.call(inputs[0]);
    }
    
    @Override
    public Shape getOutputShape() {
        if (outputShape == null && inputShape != null) {
            init();
        }
        return outputShape;
    }
}
