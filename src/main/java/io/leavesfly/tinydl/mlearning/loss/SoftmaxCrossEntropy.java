package io.leavesfly.tinydl.mlearning.loss;

import io.leavesfly.tinydl.func.Variable;

public class SoftmaxCrossEntropy extends Loss {
    @Override
    public Variable loss(Variable y, Variable predict) {

        return predict.softmaxCrossEntropy(y);
    }
}
