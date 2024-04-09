package io.leavesfly.tinydl.nnet;

import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.func.Variable;

/**
 * 神经网络中需要训练的参数，对应数学中的函数就是变量
 */
public class Parameter extends Variable {
    public Parameter(NdArray value) {
        super(value);
    }
}
