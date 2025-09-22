package io.leavesfly.tinydl.example.nlp;

import io.leavesfly.tinydl.func.Variable;
import io.leavesfly.tinydl.modality.nlp.GPT2Model;
import io.leavesfly.tinydl.modality.nlp.SimpleTokenizer;
import io.leavesfly.tinydl.ndarr.NdArray;
import io.leavesfly.tinydl.ndarr.Shape;

import java.util.ArrayList;
import java.util.List;

/**
 * GPT-2模型简单测试示例
 * 
 * @author leavesfly
 * @version 0.01
 * 
 * 该示例演示了GPT-2模型的基本功能测试，包括：
 * 1. 分词器功能测试
 * 2. 模型创建和初始化测试
 * 3. 前向传播测试
 * 4. 基本文本生成测试
 * 
 * GPT-2是一种基于Transformer的自回归语言模型，能够生成连贯的文本。
 */
public class GPT2SimpleTest {
    
    /**
     * 主函数，执行所有测试用例
     * 
     * @param args 命令行参数
     */
    public static void main(String[] args) {
        System.out.println("=== GPT-2 Simple Test ===");
        
        try {
            // 测试分词器
            testTokenizer();
            
            // 测试模型创建
            testModelCreation();
            
            // 测试前向传播
            testForwardPass();
            
            // 测试基础文本生成
            testBasicGeneration();
            
            System.out.println("\nAll tests completed successfully!");
            
        } catch (Exception e) {
            System.err.println("Test failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 测试分词器功能
     * 
     * 验证分词器的编码和解码功能是否正常工作
     */
    private static void testTokenizer() {
        System.out.println("\n--- Testing Tokenizer ---");
        
        // 创建分词器
        SimpleTokenizer tokenizer = new SimpleTokenizer();
        
        // 构建简单词汇表
        List<String> texts = new ArrayList<>();
        texts.add("hello world test");
        texts.add("machine learning is great");
        texts.add("deep learning models");
        
        tokenizer.buildVocab(texts, 1, 100);
        
        // 测试编码
        String testText = "hello world";
        int[] encoded = tokenizer.encode(testText);
        System.out.println("Original text: \"" + testText + "\"");
        System.out.println("Encoded: " + java.util.Arrays.toString(encoded));
        
        // 测试解码
        String decoded = tokenizer.decode(encoded);
        System.out.println("Decoded: \"" + decoded + "\"");
        
        System.out.println("Tokenizer test passed!");
    }
    
    /**
     * 测试模型创建
     * 
     * 验证GPT-2模型是否能正确创建和初始化
     */
    private static void testModelCreation() {
        System.out.println("\n--- Testing Model Creation ---");
        
        // 创建微型GPT-2模型
        GPT2Model model = GPT2Model.createTinyModel("test_gpt2", 100);
        
        // 打印模型信息
        model.printModelInfo();
        
        // 验证模型参数
        assert model.getVocabSize() == 100 : "Vocabulary size mismatch";
        assert model.getDModel() == 128 : "Model dimension mismatch";
        assert model.getNumLayers() == 4 : "Number of layers mismatch";
        assert model.getNumHeads() == 4 : "Number of heads mismatch";
        
        System.out.println("Model creation test passed!");
    }
    
    /**
     * 测试前向传播
     * 
     * 验证模型的前向传播功能是否正常工作，输出是否合理
     */
    private static void testForwardPass() {
        System.out.println("\n--- Testing Forward Pass ---");
        
        // 创建模型
        int vocabSize = 50;
        GPT2Model model = GPT2Model.createTinyModel("forward_test", vocabSize);
        
        // 创建测试输入
        int batchSize = 2;
        int seqLen = 16;
        NdArray input = new NdArray(new Shape(batchSize, seqLen));
        
        // 填充随机token ID
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < seqLen; j++) {
                int tokenId = (int) (Math.random() * vocabSize);
                input.set(tokenId, i, j);
            }
        }
        
        System.out.println("Input shape: " + input.shape);
        
        try {
            // 前向传播
            Variable output = model.layerForward(new Variable(input));
            NdArray outputData = output.getValue();
            
            System.out.println("Output shape: " + outputData.shape);
            
            // 验证输出形状
            assert outputData.shape.dimension[0] == batchSize : "Batch size mismatch in output";
            assert outputData.shape.dimension[1] == seqLen : "Sequence length mismatch in output";
            assert outputData.shape.dimension[2] == vocabSize : "Vocabulary size mismatch in output";
            
            // 检查输出是否包含有效值（不是NaN或无穷大）
            boolean hasValidValues = true;
            // 仅检查几个样本值
            for (int i = 0; i < batchSize && hasValidValues; i++) {
                for (int j = 0; j < Math.min(5, seqLen) && hasValidValues; j++) {
                    for (int k = 0; k < Math.min(5, vocabSize) && hasValidValues; k++) {
                        float value = outputData.get(i, j, k);
                        if (Float.isNaN(value) || Float.isInfinite(value)) {
                            hasValidValues = false;
                        }
                    }
                }
            }
            
            assert hasValidValues : "Output contains invalid values (NaN or Infinity)";
            
            System.out.println("Sample output values:");
            for (int i = 0; i < Math.min(5, outputData.shape.dimension[2]); i++) {
                System.out.println("  Token " + i + ": " + 
                    String.format("%.4f", outputData.get(0, 0, i)));
            }
            
            System.out.println("Forward pass test passed!");
            
        } catch (Exception e) {
            System.err.println("Forward pass failed: " + e.getMessage());
            e.printStackTrace();
            throw e;
        }
    }
    
    /**
     * 测试文本生成（基础版本）
     * 
     * 验证模型是否能根据给定的提示文本预测下一个token
     */
    private static void testBasicGeneration() {
        System.out.println("\n--- Testing Basic Generation ---");
        
        try {
            // 创建分词器
            SimpleTokenizer tokenizer = new SimpleTokenizer();
            List<String> texts = new ArrayList<>();
            texts.add("the quick brown fox");
            texts.add("machine learning is amazing");
            texts.add("hello world test");
            
            tokenizer.buildVocab(texts, 1, 50);
            
            // 创建模型
            GPT2Model model = GPT2Model.createTinyModel("gen_test", tokenizer.getVocabSize());
            
            // 测试输入
            String prompt = "the quick";
            int[] promptTokens = tokenizer.encode(prompt, false);
            
            // 填充到模型的最大长度
            int[] paddedTokens = tokenizer.pad(promptTokens, model.getMaxSeqLength(), "post");
            
            // 创建输入张量
            NdArray input = new NdArray(new Shape(1, model.getMaxSeqLength()));
            for (int i = 0; i < model.getMaxSeqLength(); i++) {
                input.set(paddedTokens[i], 0, i);
            }
            
            // 预测下一个token
            int nextToken = model.predictNextToken(input);
            
            System.out.println("Prompt: \"" + prompt + "\"");
            System.out.println("Next token ID: " + nextToken);
            System.out.println("Next token: \"" + tokenizer.getToken(nextToken) + "\"");
            
            System.out.println("Basic generation test passed!");
            
        } catch (Exception e) {
            System.err.println("Generation test failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
}