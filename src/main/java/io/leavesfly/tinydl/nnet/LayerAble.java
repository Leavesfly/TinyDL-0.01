package io.leavesfly.tinydl.nnet;

import io.leavesfly.tinydl.func.Function;
import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.Shape;

import java.util.Map;

/**
 * 神经网络层的抽象
 */
public abstract class LayerAble extends Function {

    abstract public String getName();

    abstract public Shape getInputShape();

    abstract public Shape getOutputShape();


    abstract public void init();

    public Variable forward(Variable... inputs) {
        return this.call(inputs);
    }

    public abstract Map<String, Parameter> getParams();

    public abstract void addParam(String paramName, Parameter value);

    public abstract Parameter getParamBy(String paramName);

    public abstract void clearGrads();

}
