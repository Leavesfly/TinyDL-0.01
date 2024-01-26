package io.leavesfly.tinydl.example.regress;

import io.leavesfly.tinydl.nnet.Block;
import io.leavesfly.tinydl.utils.Config;
import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.nnet.block.MlpBlock;
import io.leavesfly.tinydl.mlearning.Model;
import io.leavesfly.tinydl.mlearning.dataset.Batch;
import io.leavesfly.tinydl.mlearning.dataset.DataSet;
import io.leavesfly.tinydl.mlearning.dataset.simple.SinDataSet;
import io.leavesfly.tinydl.mlearning.loss.Loss;
import io.leavesfly.tinydl.mlearning.loss.MeanSquaredLoss;
import io.leavesfly.tinydl.mlearning.optimize.Optimizer;
import io.leavesfly.tinydl.mlearning.optimize.SGD;
import io.leavesfly.tinydl.utils.Plot;

import java.util.List;

/**
 * 带有噪声的sin曲线数据的拟合
 */
public class MlpSinExam {

    public static void main(String[] args) {

        //====== 1,生成数据====
        int batchSize = 100;
        DataSet dataSet = new SinDataSet(batchSize);
        dataSet.prepare();
        List<Batch> batches = dataSet.getBatches();

        Variable variableX = batches.get(0).toVariableX().setName("x").setRequireGrad(false);
        Variable variableY = batches.get(0).toVariableY().setName("y").setRequireGrad(false);

        Block block = new MlpBlock("MlpBlock", batchSize, Config.ActiveFunc.Sigmoid, 1, 10, 1);

        Model model = new Model("MlpSinExam", block);
        Optimizer optimizer = new SGD(model, 0.2f);
        Loss lossFunc = new MeanSquaredLoss();

        //train
        int maxEpoch = 10000;

        for (int i = 0; i < maxEpoch; i++) {
            Variable predictY = model.forward(variableX);
            Variable loss = lossFunc.loss(variableY, predictY);

            model.clearGrads();
            loss.backward();

            optimizer.update();

            if (i % (maxEpoch / 10) == 0 || (i == maxEpoch - 1)) {
                System.out.println("i=" + i + " loss:" + loss.getValue().getNumber());
            }
        }

//        model.plot(variableX);
//
        Variable predictY = model.forward(variableX);
        float[] p_y = predictY.transpose().getValue().getMatrix()[0];
        float[] x = variableX.transpose().getValue().getMatrix()[0];
        float[] y = variableY.transpose().getValue().getMatrix()[0];
        Plot plot = new Plot();
        plot.scatter(x, y);
        plot.line(x, p_y, "line");
        plot.show();
    }
}
