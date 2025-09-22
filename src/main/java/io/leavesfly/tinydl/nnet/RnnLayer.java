package io.leavesfly.tinydl.nnet;

import io.leavesfly.tinydl.ndarr.Shape;

/**
 * 递归神经网络层，区别于普通的前馈网络，有状态
 */
public abstract class RnnLayer extends Layer {
    public RnnLayer(String _name, Shape _inputShape, Shape _outputShape) {
        super(_name, _inputShape, _outputShape);
    }

    public abstract void resetState();

}
