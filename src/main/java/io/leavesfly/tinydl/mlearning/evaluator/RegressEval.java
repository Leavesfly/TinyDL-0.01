package io.leavesfly.tinydl.mlearning.evaluator;

import io.leavesfly.tinydl.utils.Config;
import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.mlearning.Model;
import io.leavesfly.tinydl.mlearning.dataset.Batch;
import io.leavesfly.tinydl.mlearning.dataset.DataSet;
import io.leavesfly.tinydl.mlearning.loss.Loss;

import java.util.List;

public class RegressEval extends Evaluator {

    private Loss loss;

    public RegressEval(Loss _loss, Model _model, DataSet _dataSet) {
        this.loss = _loss;
        this.model = _model;
        this.dataSet = _dataSet;
    }

    @Override
    public void evaluate() {
        List<Batch> batches = dataSet.getTestDataSet().getBatches();
        float lossValue = 0f;
        for (Batch batch : batches) {
            Variable variableX = batch.toVariableX().setName("x").setRequireGrad(false);
            Variable variableY = batch.toVariableY().setName("y").setRequireGrad(false);

            Config.train = false;

            Variable predictY = model.forward(variableX);
            Variable lossVariable = loss.loss(variableY, predictY);
            lossValue += lossVariable.getValue().getNumber().floatValue();
        }

        System.out.println(" Test dataset model's avg loss is :" + lossValue / batches.size());
    }
}
