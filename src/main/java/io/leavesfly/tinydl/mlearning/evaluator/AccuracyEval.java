package io.leavesfly.tinydl.mlearning.evaluator;

import io.leavesfly.tinydl.utils.Config;
import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.mlearning.Model;
import io.leavesfly.tinydl.mlearning.dataset.Batch;
import io.leavesfly.tinydl.mlearning.dataset.DataSet;
import io.leavesfly.tinydl.mlearning.loss.Classify;

import java.util.List;

/**
 * 准确率评估器
 * 
 * 用于评估分类模型的准确率性能指标。
 * 通过计算模型在测试数据集上的预测准确率来评估模型性能。
 * 
 * @author TinyDL
 * @version 1.0
 */
public class AccuracyEval extends Evaluator {

    private Classify classify;

    /**
     * 构造函数
     * @param _classify 分类损失函数
     * @param _model 模型
     * @param _dataSet 数据集
     */
    public AccuracyEval(Classify _classify, Model _model, DataSet _dataSet) {
        this.model = _model;
        this.dataSet = _dataSet;
        this.classify = _classify;
    }

    @Override
    public void evaluate() {

        List<Batch> batches = dataSet.getTestDataSet().getBatches();

        Config.train = false;
        float accRation = 0f;
        for (Batch batch : batches) {
            Variable variableX = batch.toVariableX().setName("x").setRequireGrad(false);
            Variable variableY = batch.toVariableY().setName("y").setRequireGrad(false);

            Variable predictY = model.forward(variableX);
            accRation += classify.accuracyRate(variableY, predictY);
        }
        accRation = accRation / batches.size();
        System.out.println("avg-accuracy rate is :" + accRation);
    }

}