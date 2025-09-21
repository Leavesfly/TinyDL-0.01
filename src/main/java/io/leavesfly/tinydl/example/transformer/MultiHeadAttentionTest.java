package io.leavesfly.tinydl.example.transformer;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.layer.transformer.MultiHeadAttention;

/**
 * 测试多头注意力层
 */
public class MultiHeadAttentionTest {
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