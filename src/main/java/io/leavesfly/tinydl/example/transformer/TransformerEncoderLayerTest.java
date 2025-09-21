package io.leavesfly.tinydl.example.transformer;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.layer.transformer.TransformerEncoderLayer;

/**
 * 测试Transformer编码器层
 */
public class TransformerEncoderLayerTest {
    public static void main(String[] args) {
        System.out.println("=== 测试Transformer编码器层 ===");
        
        int dModel = 128;
        int numHeads = 8;
        int dFF = 512;
        int batchSize = 2;
        int seqLen = 10;
        
        // 创建编码器层
        TransformerEncoderLayer encoderLayer = new TransformerEncoderLayer("encoder_layer", dModel, numHeads, dFF, 0.1);
        
        // 创建输入
        NdArray input = NdArray.likeRandom(0.0f, 1.0f, new Shape(batchSize, seqLen, dModel));
        System.out.println("输入形状: " + input.getShape());
        
        try {
            Variable output = encoderLayer.layerForward(new Variable(input));
            System.out.println("输出形状: " + output.getValue().getShape());
            System.out.println("Transformer编码器层测试成功！");
        } catch (Exception e) {
            System.err.println("Transformer编码器层测试失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
}