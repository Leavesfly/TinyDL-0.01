package io.leavesfly.tinydl.nnet.layer.dnn;

import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.Layer;
import io.leavesfly.tinydl.nnet.Parameter;

import java.util.List;

public class LinearLayer extends Layer {
    private Parameter w;
    private Parameter b;

    public LinearLayer(String _name, int hiddenRow, int hiddenCol, boolean needBias) {
        super(_name, new Shape(-1, hiddenRow), new Shape(-1, hiddenCol));
        NdArray initWeight = NdArray.likeRandomN(new Shape(hiddenRow, hiddenCol)).mulNum(Math.sqrt((double) 1 / hiddenRow));
        w = new Parameter(initWeight);
        w.setName("w");
        addParam(w.getName(), w);

        if (needBias) {
            b = new Parameter(NdArray.zeros(new Shape(1, hiddenCol)));
            b.setName("b");
            addParam(b.getName(), b);
        }
    }

    @Override
    public void init() {

    }

    @Override
    public Variable forward(Variable... inputs) {
        return inputs[0].linear(w, b);
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
