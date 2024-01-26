package io.leavesfly.tinydl.func.matrix;

import io.leavesfly.tinydl.func.Function;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;

import java.util.Collections;
import java.util.List;

public class SoftMax extends Function {
    @Override
    public NdArray forward(NdArray... inputs) {
        return inputs[0].softMax();
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {

        NdArray y = getOutput().getValue();
        NdArray gx = y.mul(yGrad);
        NdArray sumDx = gx.sumTo(new Shape(gx.getShape().getRow(), 1)).broadcastTo(gx.getShape());
        gx = gx.sub(y.mul(sumDx));
        return Collections.singletonList(gx);
    }

    @Override
    public int requireInputNum() {
        return 1;
    }
}
