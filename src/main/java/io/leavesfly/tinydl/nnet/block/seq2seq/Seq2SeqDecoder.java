package io.leavesfly.tinydl.nnet.block.seq2seq;

import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.layer.dnn.LinearLayer;
import io.leavesfly.tinydl.nnet.layer.embedd.Embedding;
import io.leavesfly.tinydl.nnet.layer.rnn.LstmLayer;

public class Seq2SeqDecoder extends Decoder {

    private Embedding embedding;
    private LstmLayer lstMlayer;
    private LinearLayer linearLayer;

    public Seq2SeqDecoder(String _name, Shape _xInputShape, Shape _yOutputShape) {
        super(_name, _xInputShape, _yOutputShape);
    }

    @Override
    public void init() {

    }

    @Override
    public Variable forward(Variable... inputs) {
        Variable input = inputs[0];
        Variable y = embedding.forward(input);
        y = lstMlayer.forward(y);
        y = linearLayer.forward(y);
        return y;
    }


    @Override
    public void initState(NdArray init) {

    }
}
