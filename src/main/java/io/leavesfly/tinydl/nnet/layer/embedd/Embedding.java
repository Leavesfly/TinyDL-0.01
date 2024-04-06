package io.leavesfly.tinydl.nnet.layer.embedd;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.Layer;
import io.leavesfly.tinydl.nnet.Parameter;
import io.leavesfly.tinydl.utils.Util;

import java.util.List;

public class Embedding extends Layer {
    private Parameter wIn;

    public Embedding(String _name, int vocabSize, int hidden_size) {
        super(_name, null, null);
        NdArray initWeight = NdArray.likeRandomN(new Shape(vocabSize, hidden_size)).mulNum(0.01f);
        wIn = new Parameter(initWeight);
        wIn.setName("wIn");
        addParam(wIn.getName(), wIn);
    }

    @Override
    public void init() {

    }

    @Override
    public Variable layerForward(Variable... inputs) {
        Variable input = inputs[0];
        int[] slices = Util.toInt(input.transpose().getValue().getMatrix()[0]);
        return wIn.getItem(slices, null);
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
