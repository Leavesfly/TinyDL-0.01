package io.leavesfly.tinydl.func.loss;

import io.leavesfly.tinydl.func.Function;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.utils.Util;

import java.util.Arrays;
import java.util.List;

/**
 * SigmoidCE
 * 支持二分类
 */
public class SigmoidCE extends Function {

    private NdArray sigmoid;

    @Override
    public NdArray forward(NdArray... inputs) {

        NdArray predict = inputs[0];
        NdArray labelY = inputs[1];
        sigmoid = predict.sigmoid();

        if (predict.getShape().getColumn() != 1) {
            throw new RuntimeException(" predict.getShape().getColumn() != 1 error!");
        }
        NdArray other = sigmoid.like(1f).sub(sigmoid);

        float loss = crossEntropyError(NdArray.merge(1, other, sigmoid), labelY);
        return new NdArray(loss);
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        //要求 预测值只有一个，支持二分类
        NdArray predict = inputs[0].getValue();
        NdArray label = inputs[1].getValue();
        int batchSize = predict.getShape().getRow();

        NdArray xGrad = sigmoid.sub(label).mul(yGrad.broadcastTo(label.getShape())).divNumber(batchSize);

        return Arrays.asList(xGrad, label.like(1));
    }

    @Override
    public int requireInputNum() {
        return 2;
    }

    public static float crossEntropyError(NdArray predict, NdArray labelY) {

        if (labelY.getShape().getColumn() != 1) {
            //说明labelY进行了 one-hot编码
            labelY = labelY.argMax(1);
        }

        int batchSize = predict.getShape().getRow();
        int[] colSlices = Util.toInt(labelY.transpose().getMatrix()[0]);
        NdArray subPredict = predict.getItem(Util.getSeq(batchSize), colSlices);

        float crossEntropyError = subPredict.add(subPredict.like(1e-7)).log().sum().getNumber().floatValue() / (float) batchSize;
        return -crossEntropyError;
    }
}
