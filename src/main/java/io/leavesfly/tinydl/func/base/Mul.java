package io.leavesfly.tinydl.func.base;

import io.leavesfly.tinydl.func.Function;
import io.leavesfly.tinydl.ndarr.NdArray;

import java.util.Arrays;
import java.util.List;


/**
 * 乘法函数
 * 
 * 实现两个变量的乘法运算。
 */
public class Mul extends Function {
    
    /**
     * 前向传播计算乘法
     * 
     * 执行两个NdArray的乘法运算：inputs[0] * inputs[1]
     * 
     * @param inputs 输入的NdArray数组，长度为2
     * @return 乘法运算结果的NdArray
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        return inputs[0].mul(inputs[1]);
    }

    /**
     * 反向传播计算梯度
     * 
     * 计算乘法运算的梯度。
     * 对于 z = x * y，有：
     * - ∂z/∂x = y
     * - ∂z/∂y = x
     * 
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        NdArray ndArray0 = inputs[0].getValue();
        NdArray ndArray1 = inputs[1].getValue();

        return Arrays.asList(yGrad.mul(ndArray1), yGrad.mul(ndArray0));
    }

    /**
     * 获取所需输入参数个数
     * 
     * 乘法运算需要两个输入参数。
     * 
     * @return 输入参数个数，固定为2
     */
    @Override
    public int requireInputNum() {
        return 2;
    }
}
