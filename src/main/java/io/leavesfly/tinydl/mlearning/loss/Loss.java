package io.leavesfly.tinydl.mlearning.loss;

import io.leavesfly.tinydl.func.Variable;

/**
 * 损失函数
 */
public abstract class Loss {
    public abstract Variable loss(Variable y, Variable predict);
}
