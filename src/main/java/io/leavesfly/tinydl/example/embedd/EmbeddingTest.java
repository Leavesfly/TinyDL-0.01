package io.leavesfly.tinydl.example.embedd;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.layer.embedd.Embedding;

/**
 * Embedding层测试示例
 * 
 * @author leavesfly
 * @version 0.01
 * 
 * 该示例演示如何使用Embedding层将词汇索引转换为词向量表示。
 */
public class EmbeddingTest {
    
    /**
     * 主函数，执行Embedding层测试
     * 
     * @param args 命令行参数
     */
    public static void main(String[] args) {
        System.out.println("=== 测试Embedding层 ===");
        
        // 定义参数
        int vocabSize = 10;      // 词汇表大小
        int embedSize = 4;       // 嵌入维度
        
        // 创建Embedding层
        Embedding embedding = new Embedding("embedding", vocabSize, embedSize);
        
        System.out.println("Embedding层参数:");
        System.out.println("- 词汇表大小: " + embedding.getVocabSize());
        System.out.println("- 嵌入维度: " + embedding.getEmbedSize());
        
        // 测试一维输入 (单个序列)
        test1DInput(embedding);
        
        // 测试二维输入 (批次序列)
        test2DInput(embedding);
    }
    
    /**
     * 测试一维输入
     */
    private static void test1DInput(Embedding embedding) {
        System.out.println("\n--- 测试一维输入 ---");
        
        // 创建输入（词汇索引）
        // 假设我们有一个包含3个词的序列
        float[][] inputData = {{1, 3, 5}};  // 一个序列，包含3个词
        NdArray input = new NdArray(inputData);
        System.out.println("输入索引: ");
        System.out.println(input);
        System.out.println("输入形状: " + input.getShape());
        
        try {
            // 前向传播
            Variable output = embedding.layerForward(new Variable(input));
            System.out.println("\n输出词向量: ");
            System.out.println(output.getValue());
            System.out.println("输出形状: " + output.getValue().getShape());
            
            System.out.println("一维输入测试成功！");
        } catch (Exception e) {
            System.err.println("一维输入测试失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 测试二维输入
     */
    private static void test2DInput(Embedding embedding) {
        System.out.println("\n--- 测试二维输入 ---");
        
        // 创建输入（词汇索引）
        // 假设我们有两个序列，每个序列有3个词
        float[][] inputData = {
            {1, 3, 5},  // 第一个序列的词索引
            {2, 4, 6}   // 第二个序列的词索引
        };
        NdArray input = new NdArray(inputData);
        System.out.println("输入索引: ");
        System.out.println(input);
        System.out.println("输入形状: " + input.getShape());
        
        try {
            // 前向传播
            Variable output = embedding.layerForward(new Variable(input));
            System.out.println("\n输出词向量: ");
            System.out.println(output.getValue());
            System.out.println("输出形状: " + output.getValue().getShape());
            
            System.out.println("二维输入测试成功！");
            
            // 显示权重矩阵
            System.out.println("\n嵌入权重矩阵: ");
            System.out.println(embedding.getParamBy("wIn").getValue());
            
        } catch (Exception e) {
            System.err.println("二维输入测试失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
}