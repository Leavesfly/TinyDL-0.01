package io.leavesfly.tinydl.func.loss;

import io.leavesfly.tinydl.func.Function;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.utils.Util;

import java.util.Arrays;
import java.util.List;

/**
 * Softmax交叉熵损失函数
 * 
 * 用于多分类问题的损失函数，结合了Softmax激活函数和交叉熵损失。
 */
public class SoftmaxCE extends Function {
    /**
     * 前向传播计算Softmax交叉熵损失
     * 
     * 计算公式：Loss = -Σ(yi*log(σ(xi)))
     * 其中σ(x)为Softmax函数，y为真实标签
     * 
     * @param inputs 输入的NdArray数组，包含预测值和真实标签
     * @return Softmax交叉熵损失值
     */
    @Override
    public NdArray forward(NdArray... inputs) {

        NdArray predict = inputs[0];
        NdArray labelY = inputs[1];

        int row = predict.getShape().getRow();

        NdArray max = predict.max(1);
        NdArray max2PredictShape = max.broadcastTo(predict.getShape());
        max = max.add(predict.sub(max2PredictShape).exp().sumTo(new Shape(row, 1)).log());

        int[] colSlices = Util.toInt(labelY.transpose().getMatrix()[0]);

        predict = predict.sub(max.broadcastTo(predict.getShape()));

        predict = predict.getItem(Util.getSeq(row), colSlices);

        float sum = predict.sum().getNumber().floatValue();
        return new NdArray(-sum / (float) row);
    }

    /**
     * 反向传播计算梯度
     * 
     * 对于Softmax交叉熵损失函数，梯度计算公式为：
     * ∂Loss/∂x = (σ(x) - y) / n
     * 其中σ(x)为Softmax函数，y为真实标签，n为批次大小
     * 
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {

        NdArray predict = inputs[0].getValue();
        NdArray label = inputs[1].getValue();

        int row = predict.getShape().getRow();
        int column = predict.getShape().getColumn();

        NdArray gy = yGrad.mulNum(1 / (float) row);
        NdArray y = predict.softMax();
        NdArray oneHot = NdArray.eye(new Shape(column, column)).getItem(
                Util.toInt(label.transpose().getMatrix()[0]), null);

        y = y.sub(oneHot).mulNum(gy.getNumber());

        return Arrays.asList(y, label.like(1));
    }

    /**
     * 获取所需输入参数个数
     * 
     * Softmax交叉熵损失函数需要两个输入参数：预测值和真实标签。
     * 
     * @return 输入参数个数，固定为2
     */
    @Override
    public int requireInputNum() {
        return 2;
    }
}
