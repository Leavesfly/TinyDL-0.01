package io.leavesfly.tinydl.nnet.block.seq2seq;

import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.Block;

/**
 * 编码器
 */
public abstract class Encoder extends Block {

    public Encoder(String _name, Shape _xInputShape, Shape _yOutputShape) {
        super(_name, _xInputShape, _yOutputShape);
    }
}
