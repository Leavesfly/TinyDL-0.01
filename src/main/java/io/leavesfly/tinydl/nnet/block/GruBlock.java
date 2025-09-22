package io.leavesfly.tinydl.nnet.block;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.Block;
import io.leavesfly.tinydl.nnet.layer.dnn.LinearLayer;
import io.leavesfly.tinydl.nnet.layer.rnn.GruLayer;

/**
 * GRU块，包含一个GRU层和一个线性输出层
 */
public class GruBlock extends Block {
    private GruLayer gruLayer;
    private LinearLayer linearLayer;

    public GruBlock(String name, int inputSize, int hiddenSize, int outputSize) {
        super(name, new Shape(-1, inputSize), new Shape(-1, outputSize));

        gruLayer = new GruLayer("gru", new Shape(-1, inputSize), new Shape(-1, hiddenSize));
        addLayer(gruLayer);

        linearLayer = new LinearLayer("line", hiddenSize, outputSize, true);
        addLayer(linearLayer);
    }

    @Override
    public void init() {

    }

    @Override
    public Variable layerForward(Variable... inputs) {
        Variable state = gruLayer.layerForward(inputs);
        return linearLayer.layerForward(state);
    }

    public void resetState() {
        gruLayer.resetState();
    }
}