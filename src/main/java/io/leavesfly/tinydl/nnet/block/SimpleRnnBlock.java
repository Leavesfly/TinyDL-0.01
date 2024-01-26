package io.leavesfly.tinydl.nnet.block;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.Block;
import io.leavesfly.tinydl.nnet.layer.dnn.LinearLayer;
import io.leavesfly.tinydl.nnet.layer.rnn.SimpleRnnlayer;

/**
 * 简单的递归神经网络的实现
 */
public class SimpleRnnBlock extends Block {
    private SimpleRnnlayer rnnLayer;
    private LinearLayer linearLayer;

    public SimpleRnnBlock(String name, int inputSize, int hiddenSize, int outputSize) {
        super(name, new Shape(-1, inputSize), new Shape(-1, outputSize));

        rnnLayer = new SimpleRnnlayer("rnn", new Shape(-1, inputSize), new Shape(-1, hiddenSize));

        getLayers().add(rnnLayer);

        linearLayer = new LinearLayer("line", hiddenSize, outputSize, true);
        getLayers().add(linearLayer);

    }

    @Override
    public void init() {

    }

    @Override
    public Variable forward(Variable... inputs) {

        Variable state = rnnLayer.forward(inputs);
        return linearLayer.forward(state);
    }
}
