package io.leavesfly.tinydl.example.regress;

import io.leavesfly.tinydl.mlearning.Model;
import io.leavesfly.tinydl.mlearning.dataset.Batch;
import io.leavesfly.tinydl.mlearning.dataset.DataSet;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.mlearning.dataset.simple.CosDataSet;
import io.leavesfly.tinydl.nnet.block.SimpleRnnBlock;

import io.leavesfly.tinydl.mlearning.loss.Loss;
import io.leavesfly.tinydl.mlearning.loss.MeanSquaredLoss;
import io.leavesfly.tinydl.mlearning.optimize.Optimizer;
import io.leavesfly.tinydl.mlearning.optimize.SGD;

import java.util.List;

/**
 * 用递归网络来拟合序列数据
 */
public class RnnCosExam {
    public static void main(String[] args) {

        test0();
    }

    public static void test0() {
        //==1，定义超参数
        int maxEpoch = 100;

        //比较特殊表示RNN的隐藏层的大小，
        // 和batchSize得一致，
        // 表示每次预测训练依赖前N个数据
        int bpttLength = 3;
        int inputSize = 1;
        int hiddenSize = 20;
        int outputSize = 1;
        float learnRate = 0.01f;

        //==2，数据集合
        CosDataSet cosCurveDataSet = new CosDataSet(bpttLength);
        cosCurveDataSet.prepare();
        DataSet trainDataSet = cosCurveDataSet.getTrainDataSet();

        List<Batch> batches = trainDataSet.getBatches();

        //==3，定义网络结构
        SimpleRnnBlock rnnBlock = new SimpleRnnBlock("rnn", inputSize, hiddenSize, outputSize);
        Model model = new Model("RnnCosExam", rnnBlock);
        Optimizer optimizer = new SGD(model, learnRate);
        Loss lossFunc = new MeanSquaredLoss();

        //==4，训练网络
        for (int i = 0; i < maxEpoch; i++) {
            //对于递归网络 有状态 每次重新训练的时候要清理中间状态
            model.resetState();

            float lossSum = 0f;
            int batchIndex = 0;
            for (Batch batch : batches) {

                NdArray[] xArray = batch.getX();
                NdArray[] yArray = batch.getX();
                Variable loss = new Variable(0f);
                loss.setName("loss");
                for (int j = 0; j < batch.getSize(); j++) {
                    Variable x = new Variable(xArray[j]).setName("x");
                    Variable y = new Variable(yArray[j]).setName("y");
                    Variable predict = model.forward(x);
                    loss = loss.add(lossFunc.loss(y, predict));
                    loss.setName("loss" + j);
                }

                model.clearGrads();
                loss.backward();
                optimizer.update();

                lossSum += loss.getValue().getNumber().floatValue() / batch.getSize();
                batchIndex++;
                if (i == maxEpoch - 1 && batchIndex == batches.size() - 1) {
//                    System.out.println(Uml.getDotGraph(loss));
                }
                //切断计算图 每批数据要清理重新构建计算图
                loss.unChainBackward();
            }
            if (i % (maxEpoch / 10) == 0 || (i == maxEpoch - 1)) {
                System.out.println("epoch: " + i + "  avg-loss:" + lossSum / batches.size());
            }
        }
    }
}
