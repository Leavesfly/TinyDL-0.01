package io.leavesfly.tinydl.nnet.layer.rnn;

import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.Parameter;
import io.leavesfly.tinydl.nnet.RnnLayer;

import java.util.List;
import java.util.Objects;

/**
 * 递归网络层
 */
public class SimpleRnnlayer extends RnnLayer {

    Parameter x2h;

    Parameter b;

    Parameter h2h;

    private Variable state;

    private int hiddeSize;

    public SimpleRnnlayer(String _name, Shape _xInputShape, Shape _yOutputShape) {
        super(_name, _xInputShape, _yOutputShape);
        hiddeSize = _yOutputShape.getColumn();
        init();
    }


    @Override
    public void resetState() {
        state = null;
    }

    @Override
    public void init() {

        int inputSize = xInputShape.getColumn();

        NdArray initWeight = NdArray.likeRandomN(new Shape(inputSize, hiddeSize))
                .mulNum(Math.sqrt((double) 1 / inputSize));
        x2h = new Parameter(initWeight);
        x2h.setName("x2h");
        addParam(x2h.getName(), x2h);

        b = new Parameter(NdArray.zeros(new Shape(1, hiddeSize)));
        b.setName("x2h-b");
        addParam(b.getName(), b);


        initWeight = NdArray.likeRandomN(new Shape(hiddeSize, hiddeSize))
                .mulNum(Math.sqrt((double) 1 / hiddeSize));
        h2h = new Parameter(initWeight);
        h2h.setName("h2h");
        addParam(h2h.getName(), h2h);

    }

    @Override
    public Variable forward(Variable... inputs) {
        Variable x = inputs[0];
        if (Objects.isNull(state)) {
            state = x.linear(x2h, b).tanh();
        } else {
            state = x.linear(x2h, b).add(state.linear(h2h, null)).tanh();
        }
        return state;
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        return null;
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        return null;
    }

    @Override
    public int requireInputNum() {
        return 0;
    }
}
