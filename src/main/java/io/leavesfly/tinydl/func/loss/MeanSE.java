package io.leavesfly.tinydl.func.loss;

import io.leavesfly.tinydl.func.Function;
import io.leavesfly.tinydl.ndarr.NdArray;

import java.util.Arrays;
import java.util.List;

/**
 * MeanSE
 */
public class MeanSE extends Function {

    @Override
    public NdArray forward(NdArray... inputs) {
        NdArray predict = inputs[0];
        NdArray labelY = inputs[1];

        int size = predict.getShape().getRow();
        return predict.sub(labelY).square().sum().divNumber(size);
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {

        NdArray predict = inputs[0].getValue();
        NdArray labelY = inputs[1].getValue();

        NdArray diff = predict.sub(labelY);
        int len = diff.getShape().getRow();
        NdArray gx0 = yGrad.broadcastTo(diff.getShape()).mul(diff).mulNumber(2).divNumber(len);

        return Arrays.asList(gx0, gx0.neg());
    }

    @Override
    public int requireInputNum() {
        return 2;
    }
}
