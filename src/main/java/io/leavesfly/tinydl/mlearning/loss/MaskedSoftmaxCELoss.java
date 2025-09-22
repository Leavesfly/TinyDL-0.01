package io.leavesfly.tinydl.mlearning.loss;

import io.leavesfly.tinydl.func.Variable;

/**
 * 掩码Softmax交叉熵损失函数
 * 
 * 用于处理序列模型中的掩码交叉熵损失计算。
 * TODO: 当前实现尚未完成，需要进一步实现掩码处理逻辑。
 * 
 * @author TinyDL
 * @version 1.0
 */
public class MaskedSoftmaxCELoss extends SoftmaxCrossEntropy {
    @Override
    public Variable loss(Variable y, Variable predict) {
        //todo
        return null;
    }
}