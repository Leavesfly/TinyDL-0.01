package io.leavesfly.tinydl.nnet.block.seq2seq;

import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.Block;

/**
 * 解码器
 */
public abstract class Decoder extends Block {

    public Decoder(String _name, Shape _xInputShape, Shape _yOutputShape) {
        super(_name, _xInputShape, _yOutputShape);
    }

    public abstract void initState(NdArray init);


}
