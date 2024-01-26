package io.leavesfly.tinydl.func.loss;

import io.leavesfly.tinydl.func.Function;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.utils.Util;

import java.util.Arrays;
import java.util.List;

/**
 * SoftmaxCE
 */
public class SoftmaxCE extends Function {
    @Override
    public NdArray forward(NdArray... inputs) {

        NdArray predict = inputs[0];
        NdArray labelY = inputs[1];

        int row = predict.getShape().getRow();

        NdArray max = predict.max(1);
        NdArray max2PredictShape = max.broadcastTo(predict.getShape());
        max = max.add(predict.sub(max2PredictShape).exp().sumTo(new Shape(row, 1)).log());

        int[] colSlices = Util.toInt(labelY.transpose().getMatrix()[0]);
        float sum = predict.sub(max.broadcastTo(predict.getShape())).getItem(
                Util.getSeq(row), colSlices).sum().getNumber().floatValue();
        return new NdArray(-sum / (float) row);
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {

        NdArray predict = inputs[0].getValue();
        NdArray label = inputs[1].getValue();

        int row = predict.getShape().getRow();
        int column = predict.getShape().getColumn();

        NdArray gy = yGrad.mulNumber(1 / (float) row);
        NdArray y = predict.softMax();
        NdArray oneHot = NdArray.eye(new Shape(column, column)).getItem(
                Util.toInt(label.transpose().getMatrix()[0]), null);

        y = y.sub(oneHot).mulNumber(gy.getNumber());

        return Arrays.asList(y, label.like(1));
    }

    @Override
    public int requireInputNum() {
        return 2;
    }
}
