package io.leavesfly.tinydl.mlearning.optimize;

import io.leavesfly.tinydl.mlearning.Model;
import io.leavesfly.tinydl.nnet.Parameter;

import java.util.Map;

/**
 * 参数优化器
 */
public abstract class Optimizer {

    private Model target;

    public Optimizer(Model target) {
        this.target = target;
    }

    public void update() {
        Map<String, Parameter> parameterMap = target.getAllParams();
        for (Parameter parameter : parameterMap.values()) {
            updateOne(parameter);
        }
    }

    public abstract void updateOne(Parameter parameter);

}
