package io.leavesfly.tinydl.example.transformer;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.block.transformer.Transformer;
import io.leavesfly.tinydl.nnet.block.transformer.TransformerDecoder;
import io.leavesfly.tinydl.nnet.block.transformer.TransformerEncoder;

/**
 * Transformer使用示例
 * 
 * 这个示例展示了如何使用完整的Transformer模型进行序列到序列的转换任务，
 * 例如机器翻译、文本摘要等。
 */
public class TransformerExample {
    
    public static void main(String[] args) {
        // 模型参数
        int dModel = 512;      // 模型维度
        int numHeads = 8;      // 注意力头数
        int numLayers = 6;     // 编码器和解码器层数
        int dFF = 2048;        // 前馈网络隐藏维度
        int maxSeqLength = 100; // 最大序列长度
        double dropoutRate = 0.1; // dropout比率
        
        int batchSize = 2;     // 批次大小
        int srcSeqLen = 10;    // 源序列长度
        int tgtSeqLen = 8;     // 目标序列长度
        
        System.out.println("=== Transformer模型示例 ===");
        
        // 方法1：使用预定义的编码器和解码器
        System.out.println("\n1. 使用自定义编码器和解码器创建Transformer");
        TransformerEncoder encoder = new TransformerEncoder(
            "encoder", dModel, numHeads, numLayers, dFF, maxSeqLength, dropoutRate
        );
        TransformerDecoder decoder = new TransformerDecoder(
            "decoder", dModel, numHeads, numLayers, dFF, maxSeqLength, dropoutRate
        );
        Transformer transformer1 = new Transformer("transformer", encoder, decoder);
        
        // 方法2：使用便捷构造函数
        System.out.println("2. 使用便捷构造函数创建Transformer");
        Transformer transformer2 = new Transformer(
            "transformer2", dModel, numHeads, numLayers, dFF, maxSeqLength, dropoutRate
        );
        
        // 方法3：使用默认参数的便捷构造函数
        System.out.println("3. 使用默认参数创建Transformer");
        Transformer transformer3 = new Transformer(
            "transformer3", dModel, numHeads, numLayers, maxSeqLength
        );
        
        // 创建示例输入数据
        System.out.println("\\n=== 创建输入数据 ===");
        
        // 编码器输入（源序列）
        NdArray encoderInput = NdArray.likeRandom(0.0f, 1.0f, 
            new Shape(batchSize, srcSeqLen, dModel));
        Variable encInput = new Variable(encoderInput);
        System.out.println("编码器输入形状: " + encoderInput.getShape());
        
        // 解码器输入（目标序列）
        NdArray decoderInput = NdArray.likeRandom(0.0f, 1.0f, 
            new Shape(batchSize, tgtSeqLen, dModel));
        Variable decInput = new Variable(decoderInput);
        System.out.println("解码器输入形状: " + decoderInput.getShape());
        
        // 前向传播
        System.out.println("\\n=== 前向传播 ===");
        try {
            Variable output = transformer1.layerForward(encInput, decInput);
            System.out.println("Transformer输出形状: " + output.getValue().getShape());
            System.out.println("前向传播成功！");
            
            // 测试仅编码功能
            System.out.println("\\n=== 测试仅编码功能 ===");
            Variable encoderOutput = transformer1.encodeOnly(encInput);
            System.out.println("编码器输出形状: " + encoderOutput.getValue().getShape());
            
            // 测试仅解码功能
            System.out.println("\\n=== 测试仅解码功能 ===");
            Variable decoderOutput = transformer1.decodeOnly(decInput, encoderOutput);
            System.out.println("解码器输出形状: " + decoderOutput.getValue().getShape());
            
            // 验证结果一致性
            System.out.println("\\n=== 验证结果一致性 ===");
            boolean isConsistent = compareOutputs(output.getValue(), decoderOutput.getValue());
            System.out.println("完整前向传播与分步骤结果一致: " + isConsistent);
            
        } catch (Exception e) {
            System.err.println("前向传播失败: " + e.getMessage());
            e.printStackTrace();
        }
        
        // 展示模型结构信息
        System.out.println("\\n=== 模型结构信息 ===");
        System.out.println("模型维度: " + dModel);
        System.out.println("注意力头数: " + numHeads);
        System.out.println("编码器层数: " + ((TransformerEncoder)transformer1.getEncoder()).getNumLayers());
        System.out.println("解码器层数: " + ((TransformerDecoder)transformer1.getDecoder()).getNumLayers());
        System.out.println("前馈网络隐藏维度: " + ((TransformerEncoder)transformer1.getEncoder()).getDFF());
        System.out.println("最大序列长度: " + ((TransformerEncoder)transformer1.getEncoder()).getMaxSeqLength());
        
        System.out.println("\\n=== 示例完成 ===");
    }
    
    /**
     * 比较两个输出是否一致（简单版本）
     */
    private static boolean compareOutputs(NdArray output1, NdArray output2) {
        if (!output1.shape.equals(output2.shape)) {
            return false;
        }
        
        float tolerance = 1e-6f;
        for (int i = 0; i < output1.buffer.length; i++) {
            if (Math.abs(output1.buffer[i] - output2.buffer[i]) > tolerance) {
                return false;
            }
        }
        return true;
    }
}