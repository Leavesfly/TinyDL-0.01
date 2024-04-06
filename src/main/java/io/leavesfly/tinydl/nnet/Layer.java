package io.leavesfly.tinydl.nnet;

import io.leavesfly.tinydl.func.Function;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 表示神经网络中具体的层，对应数学中的一个函数
 */
public abstract class Layer extends LayerAble {

    protected String name;

    protected Map<String, Parameter> params;

    protected Shape inputShape;

    protected Shape outputShape;

    protected boolean alreadyInit = false;

    public Layer(String _name, Shape _inputShape) {
        name = _name;
        this.params = new HashMap<>();
        inputShape = _inputShape;
    }

    public Layer(String _name, Shape _inputShape, Shape _outputShape) {
        name = _name;
        this.params = new HashMap<>();
        inputShape = _inputShape;
        outputShape = _outputShape;
    }

    public void clearGrads() {
        for (Parameter parameter : params.values()) {
            parameter.clearGrad();
        }
    }

    @Override
    public Shape getInputShape() {
        return inputShape;
    }

    @Override
    public Shape getOutputShape() {
        return outputShape;
    }

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

}
