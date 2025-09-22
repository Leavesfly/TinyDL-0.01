package io.leavesfly.tinydl.mlearning.loss;

import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.func.Variable;

/**
 * 分类评估工具类
 * 
 * 提供分类模型的评估方法，如准确率计算等。
 * 
 * @author TinyDL
 * @version 1.0
 */
public class Classify {
    /**
     * 计算准确率
     * @param label 真实标签
     * @param predict 预测值
     * @return 准确率
     */
    public float accuracyRate(Variable label, Variable predict) {
        int size = label.getValue().getShape().getRow();
        NdArray labelNdArray = label.getValue();
        NdArray predictNdArray = predict.getValue();

        NdArray argMax = predictNdArray.argMax(1);
        NdArray sames = argMax.eq(labelNdArray);
        return sames.sum().divNum((float) size).getNumber().floatValue();
    }
}