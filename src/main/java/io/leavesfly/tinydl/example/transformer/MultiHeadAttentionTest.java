package io.leavesfly.tinydl.example.transformer;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.layer.transformer.MultiHeadAttention;

/**
 * 多头注意力层测试示例
 * 
 * @author leavesfly
 * @version 0.01
 * 
 * 该示例演示如何使用多头注意力层，这是Transformer架构的核心组件。
 * 多头注意力机制允许模型在不同表示子空间中并行关注信息的不同方面。
 */
public class MultiHeadAttentionTest {
    
    /**
     * 主函数，执行多头注意力层测试
     * 
     * @param args 命令行参数
     */
    public static void main(String[] args) {
        System.out.println("=== 测试多头注意力层 ===");
        
        int dModel = 128;
        int numHeads = 8;
        int batchSize = 2;
        int seqLen = 10;
        
        // 创建多头注意力层
        MultiHeadAttention mha = new MultiHeadAttention("mha", dModel, numHeads, false);
        
        // 创建输入
        NdArray input = NdArray.likeRandom(0.0f, 1.0f, new Shape(batchSize, seqLen, dModel));
        System.out.println("输入形状: " + input.getShape());
        
        try {
            Variable output = mha.layerForward(new Variable(input));
            System.out.println("输出形状: " + output.getValue().getShape());
            System.out.println("多头注意力测试成功！");
        } catch (Exception e) {
            System.err.println("多头注意力测试失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
}