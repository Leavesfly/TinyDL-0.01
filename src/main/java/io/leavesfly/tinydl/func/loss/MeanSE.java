package io.leavesfly.tinydl.func.loss;

import io.leavesfly.tinydl.func.Function;
import io.leavesfly.tinydl.ndarr.NdArray;

import java.util.Arrays;
import java.util.List;

/**
 * 均方误差损失函数 (Mean Squared Error)
 * 
 * 计算预测值与真实标签之间的均方误差，常用于回归问题。
 */
public class MeanSE extends Function {

    /**
     * 前向传播计算均方误差
     * 
     * 计算公式：MSE = Σ(predict - label)² / n
     * 其中n为样本数量
     * 
     * @param inputs 输入的NdArray数组，包含预测值和真实标签
     * @return 均方误差值
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        NdArray predict = inputs[0];
        NdArray labelY = inputs[1];

        int size = predict.getShape().getRow();
        return predict.sub(labelY).square().sum().divNum(size);
    }

    /**
     * 反向传播计算梯度
     * 
     * 对于均方误差损失函数，梯度计算公式为：
     * - ∂MSE/∂predict = 2 * (predict - label) / n
     * - ∂MSE/∂label = -2 * (predict - label) / n
     * 
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {

        NdArray predict = inputs[0].getValue();
        NdArray labelY = inputs[1].getValue();

        NdArray diff = predict.sub(labelY);
        int len = diff.getShape().getRow();
        NdArray gx0 = yGrad.broadcastTo(diff.getShape()).mul(diff).mulNum(2).divNum(len);

        return Arrays.asList(gx0, gx0.neg());
    }

    /**
     * 获取所需输入参数个数
     * 
     * 均方误差损失函数需要两个输入参数：预测值和真实标签。
     * 
     * @return 输入参数个数，固定为2
     */
    @Override
    public int requireInputNum() {
        return 2;
    }
}
