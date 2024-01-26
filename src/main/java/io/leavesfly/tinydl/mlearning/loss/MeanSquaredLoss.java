package io.leavesfly.tinydl.mlearning.loss;

import io.leavesfly.tinydl.func.Variable;

/**
 * MeanSquaredLoss
 */
public class MeanSquaredLoss extends Loss {

    @Override
    public Variable loss(Variable y, Variable predict) {
        return predict.meanSquaredError(y);
    }
}
