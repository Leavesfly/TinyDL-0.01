package io.leavesfly.tinydl.nnet.block;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.Block;
import io.leavesfly.tinydl.nnet.layer.dnn.LinearLayer;
import io.leavesfly.tinydl.nnet.layer.rnn.SimpleRnnLayer;

/**
 * LSTM块，包含一个LSTM层和一个线性输出层
 * 注意：由于编译器问题，暂时使用SimpleRnnLayer作为占位符
 */
public class LstmBlock extends Block {
    private SimpleRnnLayer lstmLayer;  // 临时使用SimpleRnnLayer
    private LinearLayer linearLayer;

    public LstmBlock(String name, int inputSize, int hiddenSize, int outputSize) {
        super(name, new Shape(-1, inputSize), new Shape(-1, outputSize));

        lstmLayer = new SimpleRnnLayer("lstm", new Shape(-1, inputSize), new Shape(-1, hiddenSize));
        addLayer(lstmLayer);

        linearLayer = new LinearLayer("line", hiddenSize, outputSize, true);
        addLayer(linearLayer);
    }

    @Override
    public void init() {

    }

    @Override
    public Variable layerForward(Variable... inputs) {
        Variable state = lstmLayer.layerForward(inputs);
        return linearLayer.layerForward(state);
    }

    public void resetState() {
        lstmLayer.resetState();
    }
}