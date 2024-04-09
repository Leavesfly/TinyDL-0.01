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


    protected String name;

    protected Map<String, Parameter> params;


    protected Shape inputShape;

    protected Shape outputShape;

    protected boolean alreadyInit = false;


    abstract public void init();

    /**
     * inputs 不包含内部的参数部分,表示层的前向传播，是func的前向传播一种应用
     *
     * @param inputs
     * @return
     */
    public abstract Variable layerForward(Variable... inputs);


    public abstract void clearGrads();


    public String getName() {
        return name;
    }

    public Map<String, Parameter> getParams() {
        return params;
    }

    public void addParam(String paramName, Parameter value) {
        getParams().put(name + "." + paramName, value);
    }

    public Parameter getParamBy(String paramName) {
        return getParams().get(name + "." + paramName);
    }

    public Shape getInputShape() {
        return inputShape;
    }

    public Shape getOutputShape() {
        return outputShape;
    }


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
