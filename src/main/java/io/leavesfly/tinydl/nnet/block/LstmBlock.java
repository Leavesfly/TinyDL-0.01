package io.leavesfly.tinydl.nnet.block;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.Block;
import io.leavesfly.tinydl.nnet.layer.dnn.LinearLayer;
import io.leavesfly.tinydl.nnet.layer.rnn.LstmLayer;

public class LstmBlock extends Block {
    private LstmLayer lstmLayer;
    private LinearLayer linearLayer;

    public LstmBlock(String name, int inputSize, int hiddenSize, int outputSize) {
        super(name, new Shape(-1, inputSize), new Shape(-1, outputSize));

        lstmLayer = new LstmLayer("lstmLayer", new Shape(-1, inputSize), new Shape(inputSize, hiddenSize));
        getLayers().add(lstmLayer);

        linearLayer = new LinearLayer("linearLayer", hiddenSize, outputSize, true);
        getLayers().add(linearLayer);
    }

    public void resetState() {
        lstmLayer.resetState();
    }

    @Override
    public void init() {

    }

    @Override
    public Variable layerForward(Variable... inputs) {
        Variable state = lstmLayer.layerForward(inputs);
        return linearLayer.layerForward(state);
    }
}
