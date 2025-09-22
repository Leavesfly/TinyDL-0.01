package io.leavesfly.tinydl.example.rnn;

import io.leavesfly.tinydl.mlearning.Model;
import io.leavesfly.tinydl.mlearning.dataset.Batch;
import io.leavesfly.tinydl.mlearning.dataset.DataSet;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.mlearning.dataset.simple.SinDataSet;
import io.leavesfly.tinydl.nnet.block.SimpleRnnBlock;
import io.leavesfly.tinydl.nnet.block.GruBlock;
import io.leavesfly.tinydl.nnet.block.LstmBlock;

import io.leavesfly.tinydl.mlearning.loss.Loss;
import io.leavesfly.tinydl.mlearning.loss.MeanSquaredLoss;
import io.leavesfly.tinydl.mlearning.optimize.Optimizer;
import io.leavesfly.tinydl.mlearning.optimize.Adam;

import java.util.List;

/**
 * 完整的RNN示例，比较不同RNN层的性能
 * 包含SimpleRNN、LSTM和GRU的实现和对比
 */
public class CompleteRnnExample {
    public static void main(String[] args) {
        // 比较不同类型的RNN
        System.out.println("=== Simple RNN ===");
        testSimpleRNN();
        
        System.out.println("\n=== LSTM ===");
        testLSTM();
        
        System.out.println("\n=== GRU ===");
        testGRU();
    }

    /**
     * 测试SimpleRNN
     */
    public static void testSimpleRNN() {
        // 定义超参数
        int maxEpoch = 100;
        int bpttLength = 10;
        int inputSize = 1;
        int hiddenSize = 20;
        int outputSize = 1;
        float learnRate = 0.001f;

        // 数据集合
        SinDataSet sinDataSet = new SinDataSet(bpttLength);
        sinDataSet.prepare();
        DataSet trainDataSet = sinDataSet.getTrainDataSet();
        List<Batch> batches = trainDataSet.getBatches();

        // 定义网络结构
        SimpleRnnBlock rnnBlock = new SimpleRnnBlock("simple-rnn", inputSize, hiddenSize, outputSize);
        Model model = new Model("SimpleRNN", rnnBlock);
        Adam optimizer = new Adam(model, learnRate, 0.9f, 0.999f, 1e-8f);
        Loss lossFunc = new MeanSquaredLoss();

        // 训练网络
        trainModel(model, optimizer, lossFunc, batches, maxEpoch);
    }

    /**
     * 测试LSTM
     */
    public static void testLSTM() {
        // 定义超参数
        int maxEpoch = 100;
        int bpttLength = 10;
        int inputSize = 1;
        int hiddenSize = 20;
        int outputSize = 1;
        float learnRate = 0.001f;

        // 数据集合
        SinDataSet sinDataSet = new SinDataSet(bpttLength);
        sinDataSet.prepare();
        DataSet trainDataSet = sinDataSet.getTrainDataSet();
        List<Batch> batches = trainDataSet.getBatches();

        // 定义网络结构
        LstmBlock lstmBlock = new LstmBlock("lstm", inputSize, hiddenSize, outputSize);
        Model model = new Model("LSTM", lstmBlock);
        Adam optimizer = new Adam(model, learnRate, 0.9f, 0.999f, 1e-8f);
        Loss lossFunc = new MeanSquaredLoss();

        // 训练网络
        trainModel(model, optimizer, lossFunc, batches, maxEpoch);
    }

    /**
     * 测试GRU
     */
    public static void testGRU() {
        // 定义超参数
        int maxEpoch = 100;
        int bpttLength = 10;
        int inputSize = 1;
        int hiddenSize = 20;
        int outputSize = 1;
        float learnRate = 0.001f;

        // 数据集合
        SinDataSet sinDataSet = new SinDataSet(bpttLength);
        sinDataSet.prepare();
        DataSet trainDataSet = sinDataSet.getTrainDataSet();
        List<Batch> batches = trainDataSet.getBatches();

        // 定义网络结构
        GruBlock gruBlock = new GruBlock("gru", inputSize, hiddenSize, outputSize);
        Model model = new Model("GRU", gruBlock);
        Adam optimizer = new Adam(model, learnRate, 0.9f, 0.999f, 1e-8f);
        Loss lossFunc = new MeanSquaredLoss();

        // 训练网络
        trainModel(model, optimizer, lossFunc, batches, maxEpoch);
    }

    /**
     * 通用训练函数
     */
    private static void trainModel(Model model, Adam optimizer, Loss lossFunc, 
                                  List<Batch> batches, int maxEpoch) {
        for (int i = 0; i < maxEpoch; i++) {
            // 对于递归网络有状态，每次重新训练的时候要清理中间状态
            model.resetState();

            float lossSum = 0f;
            int batchCount = 0;
            
            for (Batch batch : batches) {
                NdArray[] xArray = batch.getX();
                NdArray[] yArray = batch.getY();
                
                Variable loss = new Variable(0f);
                loss.setName("loss");
                
                for (int j = 0; j < batch.getSize(); j++) {
                    Variable x = new Variable(xArray[j]).setName("x");
                    Variable y = new Variable(yArray[j]).setName("y");
                    Variable predict = model.forward(x);
                    loss = loss.add(lossFunc.loss(y, predict));
                }

                model.clearGrads();
                loss.backward();
                optimizer.update();

                lossSum += loss.getValue().getNumber().floatValue() / batch.getSize();
                batchCount++;
                
                // 切断计算图，每批数据要清理重新构建计算图
                loss.unChainBackward();
            }
            
            if (i % (maxEpoch / 10) == 0 || (i == maxEpoch - 1)) {
                System.out.println("epoch: " + i + "  avg-loss:" + lossSum / batchCount);
            }
        }
    }
}