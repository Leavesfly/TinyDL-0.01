package io.leavesfly.tinydl.example.classify;

import io.leavesfly.tinydl.mlearning.Monitor;
import io.leavesfly.tinydl.mlearning.Trainer;
import io.leavesfly.tinydl.mlearning.dataset.DataSet;
import io.leavesfly.tinydl.mlearning.evaluator.AccuracyEval;
import io.leavesfly.tinydl.mlearning.evaluator.Evaluator;
import io.leavesfly.tinydl.nnet.Block;
import io.leavesfly.tinydl.utils.Config;
import io.leavesfly.tinydl.mlearning.dataset.simple.MnistDataSet;
import io.leavesfly.tinydl.nnet.block.MlpBlock;
import io.leavesfly.tinydl.mlearning.Model;

import io.leavesfly.tinydl.mlearning.loss.Classify;
import io.leavesfly.tinydl.mlearning.loss.Loss;
import io.leavesfly.tinydl.mlearning.loss.SoftmaxCrossEntropy;
import io.leavesfly.tinydl.mlearning.optimize.Optimizer;
import io.leavesfly.tinydl.mlearning.optimize.SGD;

/**
 * 手写数字识别示例
 * 
 * @author leavesfly
 * @version 0.01
 * 
 * 使用MLP神经网络对MNIST手写数字数据集进行分类训练的示例。
 * 展示了完整的深度学习训练流程：
 * 1. 定义超参数
 * 2. 定义模型结构
 * 3. 准备数据集
 * 4. 配置训练器
 * 5. 执行模型训练
 * 6. 评估模型性能
 */
public class MnistMlpExam {

    /**
     * 主函数，执行MNIST手写数字识别训练
     * 
     * @param args 命令行参数
     */
    public static void main(String[] args) {
        //===1,定义超参数===
        int maxEpoch = 50;
        int batchSize = 100;

        int inputSize = 28 * 28;
        int hiddenSize1 = 100;
        int hiddenSize2 = 100;
        int outputSize = 10;

        float learRate = 0.1f;

        //===2,定义模型===
        Block block = new MlpBlock("MlpBlock", batchSize, Config.ActiveFunc.Sigmoid, inputSize, hiddenSize1, hiddenSize2, outputSize);
        Model model = new Model("MnistMlpExam", block);

        DataSet mnistDataSet = new MnistDataSet(batchSize);
        Evaluator evaluator = new AccuracyEval(new Classify(), model, mnistDataSet);
        Optimizer optimizer = new SGD(model, learRate);

        Trainer trainer = new Trainer(maxEpoch, new Monitor(), evaluator);

        Loss loss = new SoftmaxCrossEntropy();
        trainer.init(mnistDataSet, model, loss, optimizer);

        //===3,模型训练==
        trainer.train(true);

        //===4,效果评估==
        trainer.evaluate();

//        model.plot();
    }
//    epoch = 0, loss:1.8379626
//    epoch = 1, loss:0.70686436
//    epoch = 2, loss:0.4548468
//    epoch = 3, loss:0.36916062
//    epoch = 4, loss:0.32379228
//    epoch = 5, loss:0.29304275
//    epoch = 6, loss:0.26911864
//    epoch = 7, loss:0.24910116
//    epoch = 8, loss:0.23163618
//    epoch = 9, loss:0.21598813
//    avg-accuracy rate is :0.9143001
}