package io.leavesfly.tinydl.example.parallel;

import io.leavesfly.tinydl.mlearning.*;
import io.leavesfly.tinydl.mlearning.dataset.simple.SpiralDateSet;
import io.leavesfly.tinydl.mlearning.evaluator.AccuracyEval;
import io.leavesfly.tinydl.mlearning.loss.SoftmaxCrossEntropy;
import io.leavesfly.tinydl.mlearning.loss.Classify;
import io.leavesfly.tinydl.mlearning.optimize.Adam;
import io.leavesfly.tinydl.nnet.block.MlpBlock;
import io.leavesfly.tinydl.utils.Config;

/**
 * 并行训练测试示例
 * 
 * 该示例演示如何使用新的并行训练功能
 */
public class ParallelTrainingTest {
    
    public static void main(String[] args) {
        System.out.println("=== TinyDL 并行训练测试 ===");
        
        // 测试参数
        int maxEpoch = 5;
        int batchSize = 16;
        int threadCount = 2;
        
        try {
            // 准备数据集（螺旋分类数据）
            SpiralDateSet dataSet = new SpiralDateSet(batchSize);
            
            // 创建模型
            Model model = createModel("并行训练模型");
            
            // 创建损失函数和优化器
            SoftmaxCrossEntropy loss = new SoftmaxCrossEntropy();
            Adam optimizer = new Adam(model, 0.01f, 0.9f, 0.999f, 1e-8f);
            
            // 测试并行训练
            System.out.println("开始并行训练测试...");
            testParallelTraining(dataSet, model, loss, optimizer, maxEpoch, threadCount);
            
        } catch (Exception e) {
            System.err.println("测试过程中发生错误: " + e.getMessage());
            e.printStackTrace();
        }
        
        System.out.println("=== 并行训练测试完成 ===");
    }
    
    /**
     * 创建MLP模型
     */
    private static Model createModel(String name) {
        MlpBlock mlpBlock = new MlpBlock(name, 32, Config.ActiveFunc.ReLU, 2, 16, 16, 3);
        
        return new Model(name, mlpBlock);
    }
    
    /**
     * 测试并行训练
     */
    private static void testParallelTraining(SpiralDateSet dataSet, Model model, 
                                           SoftmaxCrossEntropy loss, Adam optimizer, 
                                           int maxEpoch, int threadCount) {
        long startTime = System.currentTimeMillis();
        
        Monitor monitor = new Monitor();
        Classify classify = new Classify();
        AccuracyEval evaluator = new AccuracyEval(classify, model, dataSet);
        
        // 创建并行训练器
        Trainer trainer = new Trainer(maxEpoch, monitor, evaluator, true, threadCount);
        trainer.init(dataSet, model, loss, optimizer);
        
        System.out.println("并行训练配置: 线程数=" + trainer.getParallelThreadCount() + 
                          ", 是否启用=" + trainer.isParallelTrainingEnabled());
        
        // 执行简化版并行训练（适用于模型不支持序列化的情况）
        trainer.simplifiedParallelTrain(true);
        
        long endTime = System.currentTimeMillis();
        System.out.println("训练完成，耗时: " + (endTime - startTime) + " ms");
        
        // 评估结果 - 暂时跳过图表绘制
        // trainer.evaluate();
        
        // 清理资源
        trainer.shutdown();
    }
}