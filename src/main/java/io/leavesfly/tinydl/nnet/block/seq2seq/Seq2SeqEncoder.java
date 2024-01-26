package io.leavesfly.tinydl.nnet.block.seq2seq;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.layer.embedd.Embedding;
import io.leavesfly.tinydl.nnet.layer.norm.Dropout;
import io.leavesfly.tinydl.nnet.layer.rnn.LstmLayer;

public class Seq2SeqEncoder extends Encoder {

    private Embedding embedding;
    private LstmLayer lstMlayer;
    private Dropout dropout;

    public Seq2SeqEncoder(String _name, Shape _xInputShape, Shape _yOutputShape) {
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
        y = dropout.forward(y);
        return y;
    }
}
