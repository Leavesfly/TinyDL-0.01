package io.leavesfly.tinydl.mlearning;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.mlearning.dataset.Batch;
import io.leavesfly.tinydl.mlearning.dataset.DataSet;
import io.leavesfly.tinydl.mlearning.evaluator.Evaluator;
import io.leavesfly.tinydl.mlearning.loss.Loss;
import io.leavesfly.tinydl.mlearning.optimize.Optimizer;


import java.util.List;

/**
 * 模型训练器的简单实现
 */
public class Trainer {

    private DataSet dataSet;

    private Model model;

    private Loss loss;

    private Optimizer optimizer;

    private Monitor monitor;

    private Evaluator evaluator;

    private int maxEpoch;

    public Trainer(int _maxEpoch, Monitor _monitor, Evaluator _evaluator) {
        this.maxEpoch = _maxEpoch;
        monitor = _monitor;
        evaluator = _evaluator;
    }

    public void init(DataSet _dataSet, Model _model, Loss _loss, Optimizer _optimizer) {
        dataSet = _dataSet;
        _dataSet.prepare();

        model = _model;
        loss = _loss;
        optimizer = _optimizer;
    }

    /**
     * 简单的单线程实现
     *
     * @param shuffleData
     */
    public void train(boolean shuffleData) {

        DataSet trainDataSet = dataSet.getTrainDataSet();
        if (shuffleData) {
            trainDataSet.shuffle();
        }

        for (int i = 0; i < maxEpoch; i++) {

            model.resetState();
            monitor.startNewEpoch(i);

            List<Batch> batches = trainDataSet.getBatches();
            float lossSum = 0f;

            for (Batch batch : batches) {
                Variable variableX = batch.toVariableX().setName("x").setRequireGrad(false);
                Variable variableY = batch.toVariableY().setName("y").setRequireGrad(false);

                Variable predictY = model.forward(variableX);
                Variable lossVariable = loss.loss(variableY, predictY);
                lossVariable.setName("loss");

                model.clearGrads();
                lossSum += lossVariable.getValue().getNumber().floatValue();
                lossVariable.backward();
                optimizer.update();
                lossVariable.unChainBackward();

                model.tmpPredict = predictY;
            }
            monitor.collectInfo(lossSum / batches.size());
            monitor.printTrainInfo();
        }
        monitor.plot();
    }

    /**
     * 并行训练
     *
     * @param shuffleData
     */
    public void parallelTrain(boolean shuffleData) {

        //todo 按batch 并行训练 按loss权重平均加和
    }


    public void evaluate() {
        evaluator.evaluate();
    }

}
