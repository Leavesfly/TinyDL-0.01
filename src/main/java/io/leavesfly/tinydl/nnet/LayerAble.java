package io.leavesfly.tinydl.nnet;

import io.leavesfly.tinydl.func.Function;
import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;

import java.util.List;
import java.util.Map;

/**
 * 神经网络层的抽象
 */
public abstract class LayerAble extends Function {

    abstract public String getName();

    abstract public Shape getInputShape();

    abstract public Shape getOutputShape();


    abstract public void init();

    /**
     * inputs 不包含内部的参数部分,表示层的前向传播，是func的前向传播一种应用
     *
     * @param inputs
     * @return
     */
    public abstract Variable layerForward(Variable... inputs);

    public abstract Map<String, Parameter> getParams();

    public abstract void addParam(String paramName, Parameter value);

    public abstract Parameter getParamBy(String paramName);

    public abstract void clearGrads();

    @Override
    public int requireInputNum() {
        return -1;
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        return null;
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        return null;
    }

}
