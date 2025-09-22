package io.leavesfly.tinydl.example.transformer;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.layer.transformer.PositionalEncoding;

/**
 * 位置编码层测试示例
 * 
 * @author leavesfly
 * @version 0.01
 * 
 * 该示例演示如何使用位置编码层，这是Transformer架构中用于注入序列位置信息的重要组件。
 * 由于Transformer模型本身不包含循环或卷积结构，需要位置编码来为模型提供序列中词的位置信息。
 */
public class PositionalEncodingTest {
    
    /**
     * 主函数，执行位置编码层测试
     * 
     * @param args 命令行参数
     */
    public static void main(String[] args) {
        System.out.println("=== 测试位置编码层 ===");
        
        int dModel = 128;
        int maxSeqLength = 50;
        int batchSize = 2;
        int seqLen = 10;
        
        // 创建位置编码层
        PositionalEncoding posEnc = new PositionalEncoding("pos_enc", dModel, maxSeqLength, 0.1);
        
        // 创建输入
        NdArray input = NdArray.likeRandom(0.0f, 1.0f, new Shape(batchSize, seqLen, dModel));
        System.out.println("输入形状: " + input.getShape());
        
        try {
            Variable output = posEnc.layerForward(new Variable(input));
            System.out.println("输出形状: " + output.getValue().getShape());
            System.out.println("位置编码测试成功！");
        } catch (Exception e) {
            System.err.println("位置编码测试失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
}