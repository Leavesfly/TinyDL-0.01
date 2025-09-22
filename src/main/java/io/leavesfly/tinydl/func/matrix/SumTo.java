package io.leavesfly.tinydl.func.matrix;

import io.leavesfly.tinydl.func.Function;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;

import java.util.Collections;
import java.util.List;

/**
 * 累加和到指定形状函数
 * 
 * 将输入数组累加和到指定形状。
 */
public class SumTo extends Function {

    private Shape shape;
    private Shape inputShape;

    /**
     * 构造函数
     * 
     * @param _shape 目标形状
     */
    public SumTo(Shape _shape) {
        this.shape = _shape;
    }

    /**
     * 前向传播计算累加和到指定形状
     * 
     * 将输入数组累加和到指定形状。
     * 
     * @param inputs 输入的NdArray数组，长度为1
     * @return 累加和到指定形状后的NdArray
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        inputShape = inputs[0].getShape();
        return inputs[0].broadcastTo(shape);

    }

    /**
     * 反向传播计算梯度
     * 
     * 对于累加和到指定形状操作，梯度计算通过广播操作将梯度值传播到原始形状。
     * 
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        return Collections.singletonList(yGrad.broadcastTo(inputShape));
    }

    /**
     * 获取所需输入参数个数
     * 
     * 累加和到指定形状函数需要一个输入参数。
     * 
     * @return 输入参数个数，固定为1
     */
    @Override
    public int requireInputNum() {
        return 1;
    }
}
