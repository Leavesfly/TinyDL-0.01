package io.leavesfly.tinydl.nnet.block;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.Block;

public class SequentialBlock extends Block {

    public SequentialBlock(String _name, Shape _xInputShape, Shape _yOutputShape) {
        super(_name, _xInputShape, _yOutputShape);
    }

    @Override
    public void init() {
    }

    @Override
    public Variable forward(Variable... inputs) {
        Variable y = getLayers().get(0).forward(inputs);
        for (int i = 1; i < getLayers().size(); i++) {
            y = getLayers().get(i).forward(y);
        }
        return y;
    }
}
