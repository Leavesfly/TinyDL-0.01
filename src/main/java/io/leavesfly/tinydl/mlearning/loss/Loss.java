package io.leavesfly.tinydl.mlearning.loss;

import io.leavesfly.tinydl.func.Variable;

public abstract class Loss {
    public abstract Variable loss(Variable y, Variable predict);
}
