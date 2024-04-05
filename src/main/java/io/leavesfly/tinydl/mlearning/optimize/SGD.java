package io.leavesfly.tinydl.mlearning.optimize;

import io.leavesfly.tinydl.mlearning.Model;
import io.leavesfly.tinydl.nnet.Parameter;

/**
 * 随机梯度下降
 */
public class SGD extends Optimizer {

    private float lr;

    public SGD(Model target, float learnRate) {
        super(target);
        lr = learnRate;
    }

    @Override
    public void updateOne(Parameter parameter) {
        parameter.setValue(parameter.getValue().sub(parameter.getGrad().mulNum(lr)));
    }
}
