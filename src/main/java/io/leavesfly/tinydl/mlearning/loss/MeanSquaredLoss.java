package io.leavesfly.tinydl.mlearning.loss;

import io.leavesfly.tinydl.func.Variable;

/**
 * 均方误差损失函数
 * 
 * 用于回归任务的损失计算，计算预测值与真实值之间的均方误差。
 * 
 * @author TinyDL
 * @version 1.0
 */
public class MeanSquaredLoss extends Loss {

    @Override
    public Variable loss(Variable y, Variable predict) {
        return predict.meanSquaredError(y);
    }
}