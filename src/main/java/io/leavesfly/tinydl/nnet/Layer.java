package io.leavesfly.tinydl.nnet;

import io.leavesfly.tinydl.ndarr.Shape;

import java.util.HashMap;

/**
 * 表示神经网络中具体的层，对应数学中的一个函数
 */
public abstract class Layer extends LayerAble {

    public Layer(String _name, Shape _inputShape) {
        name = _name;
        this.params = new HashMap<>();
        inputShape = _inputShape;
    }

    public Layer(String _name, Shape _inputShape, Shape _outputShape) {
        name = _name;
        this.params = new HashMap<>();
        inputShape = _inputShape;
        outputShape = _outputShape;
    }

    @Override
    public void clearGrads() {
        for (Parameter parameter : params.values()) {
            parameter.clearGrad();
        }
    }

}
