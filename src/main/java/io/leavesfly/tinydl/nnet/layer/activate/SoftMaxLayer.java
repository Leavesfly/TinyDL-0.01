package io.leavesfly.tinydl.nnet.layer.activate;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.func.matrix.SoftMax;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.nnet.Layer;

import java.util.List;

/**
 * SoftMax
 */
public class SoftMaxLayer extends Layer {

    public SoftMaxLayer(String _name) {
        super(_name, null, null);
    }

    @Override
    public void init() {

    }

    @Override
    public Variable forward(Variable... inputs) {
        return new SoftMax().call(inputs[0]);
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
        return 1;
    }


}
