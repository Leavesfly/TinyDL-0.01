package io.leavesfly.tinydl.example.seq2seq;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;
import io.leavesfly.tinydl.nnet.block.seq2seq.*;

/**
 * 序列到序列模型使用示例
 * 
 * 这个示例展示了如何使用改进后的seq2seq代码构建和训练一个完整的序列到序列模型。
 * 示例涵盖了模型构建、数据准备、前向传播等关键步骤。
 * 
 * 使用场景：
 * - 机器翻译（例如中文到英文）
 * - 文本摘要生成
 * - 对话系统
 * - 问答系统
 * 
 * @author TinyDL
 * @version 0.01
 * @since 2025-01-01
 */
public class Seq2SeqExample {
    
    public static void main(String[] args) {
        System.out.println("=== 序列到序列模型示例 ===");
        
        try {
            // 方法1：使用EncoderDecoder组合
            System.out.println("\n1. 使用EncoderDecoder组合构建模型");
            demonstrateEncoderDecoderUsage();
            
            // 方法2：分别使用编码器和解码器
            System.out.println("\n2. 分别使用编码器和解码器");
            demonstrateIndividualUsage();
            
            System.out.println("\n=== 示例执行完成 ===");
            
        } catch (Exception e) {
            System.err.println("示例执行失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 演示使用EncoderDecoder组合的方式
     */
    private static void demonstrateEncoderDecoderUsage() {
        // 模型参数配置
        int sourceVocabSize = 5000;    // 源语言词汇表大小
        int targetVocabSize = 4000;    // 目标语言词汇表大小
        int embeddingDim = 128;        // 词嵌入维度
        int hiddenSize = 256;          // LSTM隐藏层大小
        double dropoutRate = 0.1;      // Dropout比率
        
        // 序列长度
        int sourceSeqLen = 20;         // 源序列长度
        int targetSeqLen = 15;         // 目标序列长度
        int batchSize = 4;             // 批次大小
        
        System.out.println("模型参数:");
        System.out.printf("  源词汇表大小: %d, 目标词汇表大小: %d%n", sourceVocabSize, targetVocabSize);
        System.out.printf("  嵌入维度: %d, 隐藏层大小: %d%n", embeddingDim, hiddenSize);
        System.out.printf("  序列长度: 源=%d, 目标=%d, 批次大小=%d%n", sourceSeqLen, targetSeqLen, batchSize);
        
        // 1. 创建编码器
        Seq2SeqEncoder encoder = new Seq2SeqEncoder(
            "encoder",
            new Shape(batchSize, sourceSeqLen),         // 输入形状
            new Shape(batchSize, sourceSeqLen, hiddenSize), // 输出形状
            sourceVocabSize,
            embeddingDim,
            hiddenSize,
            dropoutRate
        );
        
        // 2. 创建解码器
        Seq2SeqDecoder decoder = new Seq2SeqDecoder(
            "decoder",
            new Shape(batchSize, targetSeqLen),         // 输入形状
            new Shape(batchSize, targetSeqLen, targetVocabSize), // 输出形状
            targetVocabSize,
            embeddingDim,
            hiddenSize,
            targetVocabSize
        );
        
        // 3. 创建完整的seq2seq模型
        EncoderDecoder seq2seqModel = new EncoderDecoder("seq2seq_model", encoder, decoder);
        
        // 4. 准备测试数据
        System.out.println("\n准备测试数据...");
        
        // 源序列数据（模拟输入）
        NdArray sourceSequence = createRandomSequence(batchSize, sourceSeqLen, sourceVocabSize);
        Variable sourceInput = new Variable(sourceSequence);
        System.out.println("源序列形状: " + sourceSequence.getShape());
        
        // 目标序列数据（模拟输入）
        NdArray targetSequence = createRandomSequence(batchSize, targetSeqLen, targetVocabSize);
        Variable targetInput = new Variable(targetSequence);
        System.out.println("目标序列形状: " + targetSequence.getShape());
        
        // 5. 执行前向传播
        System.out.println("\n执行前向传播...");
        Variable output = seq2seqModel.layerForward(sourceInput, targetInput);
        
        System.out.println("模型输出形状: " + output.getValue().getShape());
        System.out.println("前向传播成功完成!");
        
        // 6. 展示模型信息
        System.out.println("\n模型信息:");
        System.out.println("  " + seq2seqModel.toString());
        System.out.println("  编码器初始化状态: " + encoder.isLayersInitialized());
        System.out.println("  解码器初始化状态: " + decoder.isLayersInitialized());
    }
    
    /**
     * 演示分别使用编码器和解码器的方式
     */
    private static void demonstrateIndividualUsage() {
        // 创建独立的编码器和解码器进行更精细的控制
        int vocabSize = 3000;
        int embeddingDim = 64;
        int hiddenSize = 128;
        int seqLen = 10;
        int batchSize = 2;
        
        System.out.println("创建独立的编码器和解码器:");
        System.out.printf("  参数: 词汇=%d, 嵌入=%d, 隐藏=%d%n", vocabSize, embeddingDim, hiddenSize);
        
        // 1. 创建编码器
        Seq2SeqEncoder encoder = new Seq2SeqEncoder(
            "standalone_encoder",
            new Shape(batchSize, seqLen),
            new Shape(batchSize, seqLen, hiddenSize),
            vocabSize, embeddingDim, hiddenSize, 0.0
        );
        
        // 2. 创建解码器
        Seq2SeqDecoder decoder = new Seq2SeqDecoder(
            "standalone_decoder",
            new Shape(batchSize, seqLen),
            new Shape(batchSize, seqLen, vocabSize),
            vocabSize, embeddingDim, hiddenSize, vocabSize
        );
        
        // 3. 准备数据
        NdArray inputSeq = createRandomSequence(batchSize, seqLen, vocabSize);
        Variable input = new Variable(inputSeq);
        
        // 4. 编码阶段
        System.out.println("\n执行编码阶段...");
        Variable encoderOutput = encoder.layerForward(input);
        System.out.println("编码器输出形状: " + encoderOutput.getValue().getShape());
        
        // 5. 初始化解码器状态
        System.out.println("初始化解码器状态...");
        decoder.initState(encoderOutput.getValue());
        System.out.println("解码器状态初始化完成: " + decoder.isStateInitialized());
        
        // 6. 解码阶段
        System.out.println("执行解码阶段...");
        NdArray targetSeq = createRandomSequence(batchSize, seqLen, vocabSize);
        Variable decoderOutput = decoder.layerForward(new Variable(targetSeq));
        System.out.println("解码器输出形状: " + decoderOutput.getValue().getShape());
        
        System.out.println("分别使用编码器和解码器完成!");
    }
    
    /**
     * 创建随机序列数据（用于测试）
     * 
     * @param batchSize 批次大小
     * @param seqLen 序列长度
     * @param vocabSize 词汇表大小
     * @return 随机序列数据
     */
    private static NdArray createRandomSequence(int batchSize, int seqLen, int vocabSize) {
        NdArray sequence = NdArray.zeros(new Shape(batchSize, seqLen));
        
        // 填充随机词汇索引
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < seqLen; j++) {
                int randomToken = (int) (Math.random() * vocabSize);
                sequence.set(randomToken, i, j);
            }
        }
        
        return sequence;
    }
}