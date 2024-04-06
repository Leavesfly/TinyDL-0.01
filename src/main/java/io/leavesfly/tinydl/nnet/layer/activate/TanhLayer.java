package io.leavesfly.tinydl.nnet.layer.activate;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.func.math.Tanh;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.nnet.Layer;

import java.util.List;

/**
 * Tanh活函数
 */
public class TanhLayer extends Layer {
    public TanhLayer(String _name) {
        super(_name, null, null);
    }

    @Override
    public void init() {

    }

    @Override
    public Variable layerForward(Variable... inputs) {
        return new Tanh().call(inputs[0]);
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
