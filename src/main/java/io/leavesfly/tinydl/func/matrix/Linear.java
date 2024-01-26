package io.leavesfly.tinydl.func.matrix;

import io.leavesfly.tinydl.func.Function;
import io.leavesfly.tinydl.ndarr.NdArray;

import java.util.Arrays;
import java.util.List;

public class Linear extends Function {
    @Override
    public NdArray forward(NdArray... inputs) {

        NdArray y = inputs[0].dot(inputs[1]);
        if (inputs.length == 2) {
            return y;
        }
        return y.add(inputs[2].broadcastTo(y.getShape()));
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {

        NdArray x = inputs[0].getValue();
        NdArray w = inputs[1].getValue();

        if (inputs.length == 2) {
            return Arrays.asList(yGrad.dot(w.transpose()), x.transpose().dot(yGrad));
        } else {
            NdArray b = inputs[2].getValue();
            return Arrays.asList(yGrad.dot(w.transpose()), x.transpose().dot(yGrad), yGrad.sumTo(b.getShape()));
        }
    }

    @Override
    public int requireInputNum() {
        return -1;
    }
}
