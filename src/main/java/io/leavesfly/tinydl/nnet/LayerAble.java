package io.leavesfly.tinydl.nnet;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.Shape;

import java.util.Map;

/**
 * 神经网络层的抽象
 */
public interface LayerAble {

    String getName();

    Shape getXInputShape();

    Shape getYOutputShape();

    void init();

    /**
     * 层内的参数不会通过inputs传递进入
     *
     * @param inputs
     * @return
     */
    Variable forward(Variable... inputs);

    Map<String, Parameter> getParams();

    void addParam(String paramName, Parameter value);

    Parameter getParamBy(String paramName);

    void clearGrads();

}
