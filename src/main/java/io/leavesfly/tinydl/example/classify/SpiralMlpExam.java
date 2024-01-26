package io.leavesfly.tinydl.example.classify;


import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.mlearning.Monitor;
import io.leavesfly.tinydl.mlearning.Trainer;
import io.leavesfly.tinydl.mlearning.dataset.Batch;
import io.leavesfly.tinydl.mlearning.evaluator.AccuracyEval;
import io.leavesfly.tinydl.mlearning.evaluator.Evaluator;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.Block;
import io.leavesfly.tinydl.nnet.block.MlpBlock;
import io.leavesfly.tinydl.mlearning.Model;
import io.leavesfly.tinydl.mlearning.dataset.ArrayDataset;
import io.leavesfly.tinydl.mlearning.dataset.simple.SpiralDateSet;
import io.leavesfly.tinydl.mlearning.loss.Classify;
import io.leavesfly.tinydl.mlearning.loss.Loss;
import io.leavesfly.tinydl.mlearning.loss.SoftmaxCrossEntropy;
import io.leavesfly.tinydl.mlearning.optimize.Optimizer;
import io.leavesfly.tinydl.mlearning.optimize.SGD;
import io.leavesfly.tinydl.utils.Plot;

import java.util.List;

/**
 * 螺旋数据分类
 */
public class SpiralMlpExam {

    public static void main(String[] args) {
        test();
    }

    public static void test() {

        int maxEpoch = 300;
        int batchSize = 10;

        float learRate = 1.0f;

        int inputSize = 2;
        int hiddenSize = 30;
        int outputSize = 3;

        Block block = new MlpBlock("MlpBlock", batchSize, null, inputSize, hiddenSize, hiddenSize, outputSize);
        Model model = new Model("SpiralMlpExam", block);

        ArrayDataset dataSet = new SpiralDateSet(batchSize);

        Optimizer optimizer = new SGD(model, learRate);
        Evaluator evaluator = new AccuracyEval(new Classify(), model, dataSet);

        Trainer trainer = new Trainer(maxEpoch, new Monitor(), evaluator);
        Loss loss = new SoftmaxCrossEntropy();

        trainer.init(dataSet, model, loss, optimizer);

        trainer.train(true);

        trainer.evaluate();
    }


    public static void test1() {
        //==定义超参数
        int maxEpoch = 300;
        int batchSize = 10;

        int inputSize = 2;
        int hiddenSize = 30;
        int outputSize = 3;

        float learRate = 1.0f;

        ArrayDataset dataSet = new SpiralDateSet(batchSize);
        dataSet.prepare();
        dataSet.shuffle();
        List<Batch> batches = dataSet.getBatches();

        Block block = new MlpBlock("MlpBlock", batchSize, null, inputSize, hiddenSize, hiddenSize, outputSize);
        Model model = new Model("SpiralMlpExam", block);

        Optimizer optimizer = new SGD(model, learRate);
        Loss lossFunc = new SoftmaxCrossEntropy();
        Classify accuracy = new Classify();

        float[] lossArray = new float[maxEpoch];
        float[] accArray = new float[maxEpoch];
        for (int i = 0; i < maxEpoch; i++) {
            float sumLoss = 0f;
            float sumAcc = 0f;
            for (Batch batch : batches) {
                Variable variableX = batch.toVariableX().setName("x").setRequireGrad(false);
                Variable variableY = batch.toVariableY().setName("y").setRequireGrad(false);
                Variable predict = model.forward(variableX);
                Variable loss = lossFunc.loss(variableY, predict);
                float acc = accuracy.accuracyRate(variableY, predict);

                model.clearGrads();
                loss.backward();

                optimizer.update();
                sumLoss += loss.getValue().getNumber().floatValue() * batch.getSize();
                sumAcc += acc * batch.getSize();
            }

            sumLoss = sumLoss / dataSet.getSize();
            sumAcc = sumAcc / dataSet.getSize();
            lossArray[i] = sumLoss;
            accArray[i] = sumAcc;

            if (i % (maxEpoch / 10) == 0 || (i == maxEpoch - 1)) {
                System.out.println("i=" + i + ", loss:" + sumLoss + " acc: " + sumAcc);
            }
        }
        //预测与绘制
        Variable variableX = new Variable(NdArray.likeRandom(-1, 1, new Shape(2000, 2)));
        Variable y = model.forward(variableX);
        SpiralDateSet spiralDateSet = SpiralDateSet.toSpiralDateSet(variableX, y);

        Plot plot = new Plot();
//        plot.line(Util.toFloat(Util.getSeq(maxEpoch)), lossArray, "loss");
//        plot.line(Util.toFloat(Util.getSeq(maxEpoch)), accArray, "accuracy");
        int[] type = new int[]{0, 1, 2};
        plot.scatter(dataSet, type);
        plot.scatter(spiralDateSet, type);
        plot.show();
    }
}
