package io.leavesfly.tinydl.func.math;

import io.leavesfly.tinydl.func.Function;
import io.leavesfly.tinydl.ndarr.NdArray;

import java.util.Collections;
import java.util.List;

/**
 * 余弦函数
 * 
 * 计算输入值的余弦值。
 */
public class Cos extends Function {

    /**
     * 前向传播计算余弦
     * 
     * 计算输入值的余弦值：cos(x)
     * 
     * @param inputs 输入的NdArray数组，长度为1
     * @return 余弦值的NdArray
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        return inputs[0].cos();
    }

    /**
     * 反向传播计算梯度
     * 
     * 对于余弦函数，梯度计算公式为：
     * ∂cos(x)/∂x = -sin(x)
     * 
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        return Collections.singletonList(inputs[0].getValue().neg().sin().mul(yGrad));
    }

    /**
     * 获取所需输入参数个数
     * 
     * 余弦函数需要一个输入参数。
     * 
     * @return 输入参数个数，固定为1
     */
    @Override
    public int requireInputNum() {
        return 1;
    }
}
