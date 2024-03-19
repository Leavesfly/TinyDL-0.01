package io.leavesfly.tinydl.func.math;

import io.leavesfly.tinydl.func.Function;
import io.leavesfly.tinydl.ndarr.NdArray;

import java.util.Collections;
import java.util.List;

public class Pow extends Function {

    private float pow = 1f;

    public Pow(float pow) {
        this.pow = pow;
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        return inputs[0].pow(pow);
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        NdArray x0 = inputs[0].getValue();
        return Collections.singletonList(x0.mulNum(pow).pow(pow - 1f).mul(yGrad));
    }

    @Override
    public int requireInputNum() {
        return 1;
    }

}
